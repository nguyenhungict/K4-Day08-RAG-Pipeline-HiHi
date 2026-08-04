# RAG Evaluation Results

## Framework sử dụng

RAGAS 0.1.21 với OpenRouter LLM judge và sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.

## RAGAS Scores — Config A

| Metric | Score |
|---|---:|
| Faithfulness | 0.817 |
| Answer Relevance | 0.519 |
| Context Recall | 0.667 |
| Context Precision | 0.797 |

## A/B Retrieval Proxy Scores

| Metric | Config A (hybrid + RRF) | Config B (dense-only) | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.799 | 0.783 | +0.016 |
| Answer Relevance | 0.281 | 0.281 | +0.000 |
| Context Recall | 0.799 | 0.783 | +0.016 |
| Context Precision | 1.000 | 0.987 | +0.013 |

## A/B Comparison Analysis

Config A dùng dense retrieval + BM25 và Reciprocal Rank Fusion. Config B chỉ dùng dense semantic retrieval.

## Worst Performers (Bottom 3)

| Question | Context Recall | Failure Stage |
|---|---:|---|
| Shopee hỗ trợ những phương thức thanh toán nào? | 0.600 | Retrieval/context coverage |
| Sau khi đặt hàng có đổi phương thức thanh toán được không? | 0.615 | Retrieval/context coverage |
| Người bán không được đăng bán những sản phẩm nào? | 0.619 | Retrieval/context coverage |

## Recommendations

1. Benchmark BAAI/bge-m3 với model multilingual MiniLM hiện tại để tiếp tục cải thiện Answer Relevance.
2. Chunk theo heading thay vì chỉ theo ký tự để tăng Context Recall trên các chính sách dài.
3. Calibrate lại threshold fallback và thử cross-encoder reranker để cải thiện Context Precision.
