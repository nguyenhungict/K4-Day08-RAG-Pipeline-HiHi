"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options (chọn 1, cân nhắc đánh đổi cài đặt nặng vs cần API key):
    - sentence-transformers/all-MiniLM-L6-v2 hoặc BAAI/bge-m3 — chạy local, không
      cần API key, nhưng cài nặng (~1-2GB vì kéo theo torch)
    - Google models/text-embedding-004 (768 dim) — nhẹ, cần GEMINI_API_KEY
    - OpenAI text-embedding-3-small (1536 dim) — nhẹ, cần OPENAI_API_KEY
    Gợi ý: đọc EMBEDDING_PROVIDER từ .env (os.getenv("EMBEDDING_PROVIDER", "sentence_transformers"))
    để cả nhóm có thể đổi provider mà không sửa code — nhớ đổi provider phải xoá
    chroma_db/ cũ và reindex vì dimension khác nhau (1024/768/1536) không tương thích ngược.

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

import hashlib
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# TODO: Chọn chunking strategy và giải thích vì sao
CHUNK_SIZE = 500        # Đủ một ý/chính sách nhưng vẫn gọn khi đưa vào LLM.
CHUNK_OVERLAP = 50      # Giữ ngữ cảnh ở ranh giới đoạn.
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# TODO: Chọn embedding model và giải thích
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_DIM = 384

# TODO: Chọn vector store
VECTOR_STORE = "chromadb"  # "chromadb" | "weaviate" | "faiss"
COLLECTION_NAME = f"ecommerce_support_docs_{EMBEDDING_DIM}d"
INDEX_FILE = CHROMA_DIR / "chunks.json"

SEARCH_ALIASES = {
    "payment-methods.md": "payment methods ecommerce payment",
    "seller-listing-regulations.md": "seller listing regulations prohibited products",
    "returns-refund-policy.md": "return refund policy evidence",
    "article_01.md": "order tracking guide shipping status",
    "article_02.md": "change payment method guide",
}


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        if md_file.name.startswith("."):
            continue
        content = md_file.read_text(encoding="utf-8").strip()
        if content:
            doc_type = "legal" if "legal" in md_file.parts else "news"
            documents.append({"content": content, "metadata": {"source": md_file.name, "type": doc_type, "search_aliases": SEARCH_ALIASES.get(md_file.name, "")}})
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    chunks = []
    for doc in documents:
        text, start, index = doc["content"], 0, 0
        while start < len(text):
            end = min(len(text), start + CHUNK_SIZE)
            if end < len(text):
                # Ưu tiên kết thúc ở cuối câu. Nguồn Help Center thường bị
                # flatten thành một dòng dài nên chỉ tách theo newline sẽ làm
                # chunk bắt đầu/kết thúc giữa câu.
                sentence_boundary = max(
                    text.rfind(". ", start, end),
                    text.rfind("? ", start, end),
                    text.rfind("! ", start, end),
                )
                boundary = sentence_boundary + 1 if sentence_boundary >= 0 else text.rfind(" ", start, end)
                if boundary > start + CHUNK_SIZE // 2:
                    end = boundary
            part = text[start:end].strip()
            if part:
                chunks.append({"content": part, "metadata": {**doc["metadata"], "chunk_index": index}})
                index += 1
            if end >= len(text):
                break
            overlap_start = max(end - CHUNK_OVERLAP, start + 1)
            # Không bao giờ mở chunk mới giữa một từ. Nếu có dấu kết câu trong
            # vùng overlap, bắt đầu từ câu hoàn chỉnh gần nhất.
            sentence_starts = [text.rfind(mark, overlap_start, end) for mark in (". ", "? ", "! ")]
            last_sentence = max(sentence_starts)
            if last_sentence >= overlap_start:
                start = last_sentence + 2
            else:
                next_space = text.find(" ", overlap_start, end)
                start = next_space + 1 if next_space >= 0 else end
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    for chunk, vector in zip(chunks, embed_texts([c["content"] for c in chunks])):
        chunk["embedding"] = vector
    return chunks


def _tokens(text: str) -> list[str]:
    # Bỏ token một ký tự ("i", "a", ...) vì tạo false-positive mạnh trong
    # corpus tiếng Việt lớn và không mang đủ tín hiệu retrieval.
    return [token for token in re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE) if len(token) > 1]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Multilingual sentence embedding; hash chỉ là fallback khi model chưa có."""
    try:
        model = get_embedding_model()
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
    except Exception as error:
        # Giữ pipeline chạy offline trên máy chưa tải model; log rõ để demo không
        # nhầm fallback này với embedding chính.
        print(f"⚠ Embedding model unavailable, using hash fallback: {error}")
        return _hash_embed_texts(texts)


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def _hash_embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        vector = [0.0] * EMBEDDING_DIM
        for token in _tokens(text):
            slot = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % EMBEDDING_DIM
            vector[slot] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        vectors.append([x / norm for x in vector])
    return vectors


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    # JSON index là portable fallback, giúp search chạy cả khi Chroma chưa được cài.
    INDEX_FILE.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        collection.upsert(ids=[f"{c['metadata']['source']}_{c['metadata']['chunk_index']}" for c in chunks], documents=[c["content"] for c in chunks], embeddings=[c["embedding"] for c in chunks], metadatas=[c["metadata"] for c in chunks])
    except ImportError:
        pass


def load_index() -> list[dict]:
    """Đọc index local; tự tạo nếu dữ liệu đã được chuẩn hoá nhưng chưa index."""
    if INDEX_FILE.exists():
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    chunks = chunk_documents(load_documents())
    return embed_chunks(chunks) if chunks else []


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
