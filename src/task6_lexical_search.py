"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _load_corpus() -> list[dict]:
    """Load toàn bộ file .md từ data/standardized/ làm corpus."""
    corpus = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            if content.strip():  # Bỏ qua file rỗng
                corpus.append(
                    {
                        "content": content,
                        "metadata": {
                            "source": md_file.name,
                            "doc_type": md_file.parent.name,  # "legal" hoặc "news"
                            "chunk_index": 0,
                            "customer_role": "both",
                        },
                    }
                )
        except Exception as e:
            print(f"⚠ Không đọc được {md_file.name}: {e}")
    return corpus


# Load corpus và build index khi module được import
CORPUS: list[dict] = _load_corpus()


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    if not corpus:
        return None
    # Tokenize — split theo khoảng trắng
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)


# Singleton index
_BM25_INDEX = build_bm25_index(CORPUS) if CORPUS else None


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if not CORPUS or _BM25_INDEX is None:
        return []

    tokenized_query = query.lower().split()
    scores = _BM25_INDEX.get_scores(tokenized_query)

    # Lấy top_k index có điểm cao nhất
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append(
                {
                    "content": CORPUS[idx]["content"],
                    "score": float(scores[idx]),
                    "metadata": CORPUS[idx]["metadata"],
                }
            )

    # Đảm bảo sort descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


if __name__ == "__main__":
    print(f"Corpus size: {len(CORPUS)} documents")
    # Test
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    if results:
        for r in results:
            print(
                f"[{r['score']:.3f}] ({r['metadata']['source']}) {r['content'][:100]}..."
            )
    else:
        print(
            "⚠ Không có kết quả — corpus rỗng hoặc chưa convert markdown (chạy task3 trước)."
        )
