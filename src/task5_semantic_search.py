"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _load_documents_for_search() -> list[dict]:
    """Đọc markdown files từ data/standardized/ làm corpus cho search."""
    if not STANDARDIZED_DIR.exists():
        return []

    docs = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception:
            continue
        if not content.strip():
            continue
        docs.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "doc_type": md_file.parent.name,
                    "chunk_index": 0,
                },
            }
        )
    return docs


def _semantic_fallback(query: str, top_k: int = 10) -> list[dict]:
    """Fallback nhẹ khi ChromaDB hoặc embedding model không sẵn sàng."""
    docs = _load_documents_for_search()
    if not docs:
        return []

    import math
    import re

    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    doc_tokens = [tokenize(doc["content"]) for doc in docs]
    vocab = sorted(set(query_tokens) | {token for tokens in doc_tokens for token in tokens})
    doc_freq = {term: 0 for term in vocab}
    for tokens in doc_tokens:
        for term in set(tokens):
            doc_freq[term] += 1

    idf = {term: math.log((1 + len(doc_tokens)) / (1 + doc_freq[term])) + 1.0 for term in vocab}
    query_vector = {term: query_tokens.count(term) * idf[term] for term in set(query_tokens)}

    ranked = []
    for idx, tokens in enumerate(doc_tokens):
        doc_vector = {term: tokens.count(term) * idf[term] for term in set(tokens)}
        dot = sum(query_vector.get(term, 0.0) * doc_vector.get(term, 0.0) for term in set(query_vector) | set(doc_vector))
        norm_q = math.sqrt(sum(value * value for value in query_vector.values()))
        norm_d = math.sqrt(sum(value * value for value in doc_vector.values()))
        score = dot / (norm_q * norm_d) if norm_q and norm_d else 0.0
        if score > 0:
            ranked.append(
                {
                    "content": docs[idx]["content"],
                    "score": round(float(score), 4),
                    "metadata": docs[idx]["metadata"],
                }
            )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not query or not query.strip():
        return []

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except Exception:
        chromadb = None
        SentenceTransformer = None

    if chromadb is not None and SentenceTransformer is not None:
        try:
            from src.task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            collection = client.get_collection(name=COLLECTION_NAME)
            if collection.count() > 0:
                model = SentenceTransformer(EMBEDDING_MODEL)
                query_vector = model.encode(query, show_progress_bar=False).tolist()
                results = collection.query(
                    query_embeddings=[query_vector],
                    n_results=min(max(1, top_k), collection.count()),
                    include=["documents", "metadatas", "distances"],
                )

                output = []
                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]

                for doc, meta, dist in zip(documents, metadatas, distances):
                    score = max(0.0, 1.0 - dist)
                    output.append({
                        "content": doc,
                        "score": round(score, 4),
                        "metadata": meta or {},
                    })

                output.sort(key=lambda item: item["score"], reverse=True)
                return output[:top_k]
        except Exception:
            pass

    return _semantic_fallback(query, top_k)


if __name__ == "__main__":
    results = semantic_search("quy định trả hàng hoàn tiền shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
