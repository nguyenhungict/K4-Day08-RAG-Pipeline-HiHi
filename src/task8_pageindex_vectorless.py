"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import os
import re
import tempfile
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()
PAGEINDEX_DOC_ID = os.getenv("PAGEINDEX_DOC_ID", "").strip()
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _load_documents() -> List[dict]:
    if STANDARDIZED_DIR.exists():
        md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
        if md_files:
            docs = []
            for md_file in md_files:
                try:
                    content = md_file.read_text(encoding="utf-8")
                except Exception:
                    continue
                docs.append({
                    "content": content,
                    "metadata": {"source": md_file.name, "type": "legal" if "legal" in md_file.parts else "news"},
                })
            if docs:
                return docs

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
    ]


def _markdown_to_pdf(markdown_path: Path, output_path: Path) -> None:
    """Convert markdown text to a simple PDF for PageIndex upload."""
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise ImportError("fpdf chưa cài đặt. Hãy cài `pip install fpdf2`") from exc

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    text = markdown_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if not line.strip():
            pdf.ln(4)
            continue
        if line.startswith("#"):
            # Use bold style for headings
            pdf.set_font("Arial", style="B", size=12)
            pdf.multi_cell(0, 6, line.strip())
            pdf.set_font("Arial", style="", size=12)
        else:
            pdf.multi_cell(0, 6, line)

    pdf.output(str(output_path))


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        raise EnvironmentError("PAGEINDEX_API_KEY chưa được cấu hình trong .env")

    try:
        from pageindex.client import PageIndexClient
    except ImportError as exc:
        raise ImportError("pageindex package chưa cài đặt. Hãy cài `pip install pageindex`") from exc

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    uploaded = []
    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))

    if not md_files:
        raise FileNotFoundError(
            f"Không tìm thấy file markdown để upload trong {STANDARDIZED_DIR}"
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_dir = Path(tmp_dir)
        for md_file in md_files:
            if not md_file.exists():
                continue
            upload_path = md_file
            if md_file.suffix.lower() == ".md":
                upload_path = temp_dir / f"{md_file.stem}.pdf"
                _markdown_to_pdf(md_file, upload_path)
            response = client.submit_document(str(upload_path))
            doc_id = response.get("doc_id") or response.get("id")
            uploaded.append((md_file.name, doc_id))
            print(f"  ✓ Uploaded: {md_file.name} -> {doc_id}")

    return uploaded


def _pageindex_query(query: str, top_k: int = 5) -> list[dict]:
    try:
        from pageindex.client import PageIndexClient
    except ImportError:
        return []

    if not PAGEINDEX_API_KEY:
        return []

    if not PAGEINDEX_DOC_ID:
        return []

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    try:
        resp = client.submit_query(doc_id=PAGEINDEX_DOC_ID, query=query)
        retrieval_id = resp.get("retrieval_id") or resp.get("id")
        if not retrieval_id:
            return []

        retrieval = client.get_retrieval(retrieval_id)
    except Exception:
        return []

    results = []
    for node in retrieval.get("retrieved_nodes", [])[:top_k]:
        for group in node.get("relevant_contents", []):
            for item in group:
                results.append({
                    "content": item.get("relevant_content", ""),
                    "score": float(item.get("score", 0.0)) if item.get("score") is not None else 0.0,
                    "metadata": {
                        "section": item.get("section_title"),
                        "source": item.get("source", PAGEINDEX_DOC_ID),
                    },
                    "source": "pageindex",
                })
    return results[:top_k]


def _local_pageindex_fallback(query: str, top_k: int = 5) -> list[dict]:
    documents = _load_documents()
    query_terms = _tokenize(query)
    if not query_terms:
        return []

    term_counts = []
    for doc in documents:
        content_tokens = _tokenize(doc["content"])
        count = sum(content_tokens.count(term) for term in query_terms)
        if count > 0:
            term_counts.append((doc, float(count)))

    term_counts.sort(key=lambda item: item[1], reverse=True)
    results = []
    for doc, score in term_counts[:top_k]:
        results.append({
            "content": doc["content"],
            "score": score,
            "metadata": doc["metadata"],
            "source": "pageindex",
        })
    return results


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if PAGEINDEX_API_KEY and PAGEINDEX_DOC_ID:
        results = _pageindex_query(query, top_k=top_k)
        if results:
            return results

    return _local_pageindex_fallback(query, top_k=top_k)


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("danh sách sản phẩm cấm đăng bán", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
