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

from collections import Counter, defaultdict
from math import log
from pathlib import Path
from typing import List
import re

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
            print(f" Không đọc được {md_file.name}: {e}")
    return corpus


# Load corpus và build index khi module được import
CORPUS: list[dict] = _load_corpus()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _load_corpus() -> List[dict]:
    if STANDARDIZED_DIR.exists():
        md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
        if md_files:
            corpus = []
            for md_file in md_files:
                try:
                    content = md_file.read_text(encoding="utf-8")
                except Exception:
                    continue
                doc_type = "legal" if "legal" in md_file.parts else "news"
                corpus.append({
                    "content": content,
                    "metadata": {"source": md_file.name, "type": doc_type},
                })
            if corpus:
                return corpus

    return [
        {
            "content": "Shopee supports refunds within 15 days of receiving the item when the product is defective or does not match the description.",
            "metadata": {"source": "mock_returns_refund_en.md", "type": "legal"},
        },
        {
            "content": "Các phương thức thanh toán trên Shopee bao gồm thẻ tín dụng, thẻ ghi nợ, ví ShopeePay, chuyển khoản ngân hàng và thanh toán khi nhận hàng.",
            "metadata": {"source": "mock_payment_methods_vi.md", "type": "news"},
        },
        {
            "content": "Shopee payment methods include credit cards, debit cards, ShopeePay wallet, bank transfer, and cash on delivery.",
            "metadata": {"source": "mock_payment_methods_en.md", "type": "news"},
        },
        {
            "content": "Người bán cần cung cấp thông tin chính xác về sản phẩm; các mặt hàng cấm đăng bán bao gồm hàng giả, hàng vi phạm bản quyền và chất cấm.",
            "metadata": {"source": "mock_seller_listing_vi.md", "type": "legal"},
        },
        {
            "content": "Seller listing regulations require accurate product information; prohibited items include counterfeit goods, copyright-infringing products, and banned substances.",
            "metadata": {"source": "mock_seller_listing_en.md", "type": "legal"},
        },
        {
            "content": "Shopee requires data privacy transparency; personal information is used only for order processing and customer service.",
            "metadata": {"source": "mock_privacy_policy_en.md", "type": "legal"},
        },
        {
            "content": "Hướng dẫn theo dõi đơn hàng cho biết người mua có thể xem trạng thái vận chuyển trong ứng dụng Shopee và liên hệ người bán khi cần hỗ trợ.",
            "metadata": {"source": "mock_shipping_guide_vi.md", "type": "news"},
        },
    ]


class BM25Index:
    def __init__(self, corpus: List[dict], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.documents = [_tokenize(doc["content"]) for doc in corpus]
        self.doc_len = [len(tokens) for tokens in self.documents]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.N = len(self.documents)
        self.doc_freqs = [Counter(tokens) for tokens in self.documents]
        self.df = defaultdict(int)
        for tokens in self.documents:
            for term in set(tokens):
                self.df[term] += 1
        self.idf = {
            term: log((self.N - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in self.df.items()
        }

    def get_scores(self, query_terms: List[str]) -> List[float]:
        scores = []
        if not query_terms:
            return [0.0] * self.N
        for idx, freq in enumerate(self.doc_freqs):
            score = 0.0
            doc_length = self.doc_len[idx]
            for term in query_terms:
                if term not in self.idf:
                    continue
                tf = freq.get(term, 0)
                denom = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avgdl))
                if denom <= 0:
                    continue
                score += self.idf[term] * tf * (self.k1 + 1) / denom
            scores.append(score)
        return scores


CORPUS: List[dict] = _load_corpus()
_BM25_INDEX = None


def build_bm25_index(corpus: List[dict]):
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
            " Không có kết quả — corpus rỗng hoặc chưa convert markdown (chạy task3 trước)."
        )
