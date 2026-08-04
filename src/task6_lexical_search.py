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

import re

from .task4_chunking_indexing import load_index

# TODO: Load corpus từ data/standardized/ hoặc từ vector store
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE) if len(token) > 1]


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi
    return BM25Okapi([
        _tokenize(f"{doc.get('metadata', {}).get('search_aliases', '')} {doc['content']}")
        for doc in corpus
    ])


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
    if not query.strip() or top_k <= 0:
        return []
    corpus = load_index()
    if not corpus:
        return []
    try:
        scores = build_bm25_index(corpus).get_scores(_tokenize(query))
    except ImportError:  # fallback simple term-frequency nếu dependency chưa cài
        terms = _tokenize(query)
        scores = [sum(_tokenize(item["content"]).count(term) for term in terms) for item in corpus]
    ranked = sorted(enumerate(scores), key=lambda pair: float(pair[1]), reverse=True)
    return [{"content": corpus[i]["content"], "score": float(score), "metadata": corpus[i]["metadata"]}
            for i, score in ranked[:top_k] if score > 0]


if __name__ == "__main__":
    # Test
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
