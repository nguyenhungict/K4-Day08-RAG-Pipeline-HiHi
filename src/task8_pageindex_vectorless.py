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
import json
import re
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PROJECT_ROOT = Path(__file__).parent.parent
DOC_IDS_PATH = PROJECT_ROOT / "pageindex_doc_ids.json"
PAGEINDEX_PDF_DIR = PROJECT_ROOT / "pageindex_pdfs"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong .env")
    from pageindex import PageIndexClient

    pdf_path = _build_corpus_pdf()
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    response = client.submit_document(str(pdf_path))
    doc_id = response.get("doc_id") or response.get("id")
    if not doc_id:
        raise RuntimeError(f"PageIndex không trả doc_id: {response}")
    payload = {"documents": [{"source": pdf_path.name, "doc_id": doc_id}]}
    DOC_IDS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Uploaded {pdf_path.name} -> {doc_id}")
    return payload


def _build_corpus_pdf() -> Path:
    """Gộp corpus Markdown thật thành một PDF Unicode để upload PageIndex."""
    from fpdf import FPDF

    files = [path for path in sorted(STANDARDIZED_DIR.rglob("*.md")) if not path.name.startswith(".")]
    if not files:
        raise RuntimeError("Không có Markdown trong data/standardized")
    PAGEINDEX_PDF_DIR.mkdir(parents=True, exist_ok=True)
    output = PAGEINDEX_PDF_DIR / "shopee-support-corpus.pdf"
    font_path = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if not font_path.exists():
        raise RuntimeError("Không tìm thấy font Unicode để tạo PDF PageIndex")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("Unicode", fname=str(font_path))
    for path in files:
        pdf.add_page()
        pdf.set_font("Unicode", size=14)
        pdf.multi_cell(0, 8, text=f"SOURCE: {path.name}")
        pdf.ln(2)
        pdf.set_font("Unicode", size=10)
        text = path.read_text(encoding="utf-8")
        # Loại control chars không hợp lệ trong PDF text stream.
        text = "".join(char for char in text if char in "\n\t" or ord(char) >= 32)
        pdf.multi_cell(0, 5, text=text)
    pdf.output(str(output))
    return output


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
    if PAGEINDEX_API_KEY and DOC_IDS_PATH.exists():
        from pageindex import PageIndexClient

        cached = json.loads(DOC_IDS_PATH.read_text(encoding="utf-8"))
        doc_ids = [item["doc_id"] for item in cached.get("documents", [])]
        if doc_ids:
            client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
            response = client.chat_completions(
                messages=[{"role": "user", "content": query}],
                doc_id=doc_ids[0] if len(doc_ids) == 1 else doc_ids,
                temperature=0,
                enable_citations=True,
            )
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                return [{"content": content, "score": 1.0, "metadata": {"source": "PageIndex Cloud", "doc_ids": ",".join(doc_ids)}, "source": "pageindex"}][:top_k]

    # Local structural fallback: mỗi Markdown là một page/document. Nó giữ được
    # section/header thay vì phụ thuộc vector, đồng thời cho demo offline.
    terms = {token for token in re.findall(r"[\wÀ-ỹ]+", query.lower(), flags=re.UNICODE) if len(token) > 1}
    results = []
    for path in STANDARDIZED_DIR.rglob("*.md"):
        content = path.read_text(encoding="utf-8").strip()
        words = {token for token in re.findall(r"[\wÀ-ỹ]+", content.lower(), flags=re.UNICODE) if len(token) > 1}
        overlap = len(terms & words)
        if overlap:
            results.append({"content": content, "score": round(overlap / max(1, len(terms)), 4),
                            "metadata": {"source": path.name, "type": "legal" if "legal" in path.parts else "news", "section": content.splitlines()[0].lstrip("# ")}, "source": "pageindex"})
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]
    #
    # client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    # resp = client.submit_query(doc_id=doc_id, query=query)
    # retrieval_id = resp.get("retrieval_id") or resp.get("id")
    #
    # # Poll cho đến khi status == "completed"
    # retrieval = client.get_retrieval(retrieval_id)
    #
    # # Parse retrieval["retrieved_nodes"] — mỗi node có "relevant_contents"
    # results = []
    # for node in retrieval.get("retrieved_nodes", [])[:2]:
    #     for group in node.get("relevant_contents", []):
    #         for item in group:
    #             results.append({
    #                 "content": item.get("relevant_content", ""),
    #                 "score": ...,  # PageIndex không trả score trực tiếp — tự gán theo rank
    #                 "metadata": {"section": item.get("section_title")},
    #                 "source": "pageindex",
    #             })
    # return results[:top_k]


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
