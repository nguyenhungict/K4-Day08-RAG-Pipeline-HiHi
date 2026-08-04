"""
Task 1 — Thu thập văn bản chính sách thương mại điện tử / hỗ trợ khách hàng.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản chính sách (PDF/DOCX) từ trang chính thức của một sàn TMĐT.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, mô tả đúng nội dung.

Gợi ý nguồn (ví dụ trang công khai Shopee Vietnam — help.shopee.vn):
    - https://help.shopee.vn/portal/4/article/77251 (Chính sách trả hàng và hoàn tiền)
    - https://help.shopee.vn/portal/4/article/79198 (Phương thức thanh toán)
    - https://help.shopee.vn/portal/4/article/77244 (Chính sách bảo mật)

Gợi ý văn bản (chủ đề chính sách thương mại điện tử):
    - Chính sách đổi trả/hoàn tiền (Returns/Refund Policy)
    - Phương thức thanh toán (Payment Methods)
    - Chính sách bảo mật (Privacy Policy)
    - Quy định đăng bán sản phẩm cho người bán (Seller Listing Regulations)

Nhớ gắn metadata `customer_role` (`buyer`/`seller`/`both`) cho từng tài liệu — yêu cầu riêng
của K4 Variant (kế thừa từ Lab 07), cần thiết để viết benchmark query dùng metadata_filter.

Lưu ý: một số trang help center dùng JavaScript render nội dung (SPA) — crawl về chỉ thấy
tiêu đề mà không có nội dung thật. Đổi sang bài viết khác cùng domain thay vì cố xử lý,
và chỉ dùng nguồn công khai/được phép chia sẻ.
"""

from pathlib import Path
from datetime import datetime
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


SAMPLE_POLICIES = {
    "returns-refund-policy.docx": """Returns and Refund Policy — Chính sách Trả hàng và Hoàn tiền\n\nNgười mua có thể gửi yêu cầu Trả hàng/Hoàn tiền trong vòng 15 ngày kể từ khi đơn hàng được giao thành công. Đối với thực phẩm tươi sống hoặc đông lạnh, thời hạn là 24 giờ. Người mua cần chọn lý do phù hợp và cung cấp ảnh hoặc video làm bằng chứng khi hệ thống yêu cầu. Hoàn tiền được xử lý theo phương thức thanh toán ban đầu sau khi yêu cầu được chấp thuận.""",
    "payment-methods.docx": """Payment Methods Overview — Phương thức thanh toán\n\nNgười mua có thể thanh toán bằng ShopeePay, thẻ tín dụng hoặc thẻ ghi nợ, trả góp qua thẻ, QR Code, chuyển khoản qua ứng dụng ngân hàng, thẻ NAPAS, Apple Pay, Google Pay hoặc Thanh toán khi nhận hàng (COD). Phương thức có thể thay đổi theo đơn hàng và khu vực giao hàng.""",
    "seller-listing-regulations.docx": """Seller Listing Regulations — Quy định đăng bán sản phẩm\n\nNgười bán không được đăng hàng giả, hàng nhái, sản phẩm bất hợp pháp, nội dung vi phạm quyền sở hữu trí tuệ hoặc mặt hàng thuộc danh sách cấm của sàn. Người bán phải cung cấp mô tả, hình ảnh và thông tin sản phẩm chính xác; vi phạm có thể khiến sản phẩm bị gỡ hoặc tài khoản bị xử lý.""",
}

LEGAL_SOURCES = {
    "returns-refund-policy.docx": "https://help.shopee.vn/portal/4/article/77251?seo=1",
    "payment-methods.docx": "https://help.shopee.vn/portal/4/article/79198-",
    "seller-listing-regulations.docx": "https://help.shopee.vn/portal/4/article/77246",
}


def _write_docx(path: Path, text: str) -> None:
    """Tạo DOCX tối giản, tự chứa để bộ dữ liệu mẫu có thể dùng offline."""
    paragraphs = "".join(
        f"<w:p><w:r><w:t xml:space='preserve'>{escape(line)}</w:t></w:r></w:p>"
        for line in text.split("\n") if line
    )
    document = ("<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
                "<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>"
                f"<w:body>{paragraphs}</w:body></w:document>")
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", """<?xml version='1.0' encoding='UTF-8'?><Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'><Default Extension='rels' ContentType='application/vnd.openxmlformats-package.relationships+xml'/><Default Extension='xml' ContentType='application/xml'/><Override PartName='/word/document.xml' ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml'/></Types>""")
        archive.writestr("_rels/.rels", """<?xml version='1.0' encoding='UTF-8'?><Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'><Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' Target='word/document.xml'/></Relationships>""")
        archive.writestr("word/document.xml", document)


def collect_sample_legal_docs() -> None:
    """Tạo ba tài liệu chính sách mẫu khi crawl nguồn ngoài không khả dụng."""
    setup_directory()
    for filename, content in SAMPLE_POLICIES.items():
        destination = DATA_DIR / filename
        _write_docx(destination, content)
        print(f"✓ Đã tạo dữ liệu mẫu: {destination.name}")


def _fetch_visible_text(url: str) -> tuple[str, str]:
    """Tải nội dung đang hiển thị từ trang help center công khai của Shopee."""
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(url, timeout=45, headers={"User-Agent": "Mozilla/5.0 (RAG educational crawler)"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else "Shopee Policy"
    visible = "\n".join(part.strip() for part in soup.stripped_strings if part.strip())
    if len(visible) < 1000:
        raise ValueError(f"Nội dung crawl quá ngắn: {len(visible)} ký tự")
    return title, visible


def collect_legal_docs() -> None:
    """Thu thập ba chính sách thật; fallback mẫu riêng từng file nếu nguồn lỗi."""
    setup_directory()
    for filename, url in LEGAL_SOURCES.items():
        try:
            title, visible = _fetch_visible_text(url)
            content = (
                f"{title}\n\nSource URL: {url}\n"
                f"Crawled at: {datetime.now().astimezone().isoformat()}\n"
                "Source type: Official Shopee Vietnam Help Center\n\n"
                f"{visible}"
            )
            _write_docx(DATA_DIR / filename, content)
            print(f"✓ Đã tải nguồn thật: {filename} ({len(visible)} ký tự)")
        except Exception as error:
            _write_docx(DATA_DIR / filename, SAMPLE_POLICIES[filename])
            print(f"⚠ Không tải được {url}; dùng fallback cho {filename}: {error}")


# TODO: Tải file PDF/DOCX về DATA_DIR
# Có thể tải thủ công hoặc viết script download nếu có direct link.
#
# Ví dụ nếu có direct link:
#
# import requests
#
# def download_file(url: str, filename: str):
#     response = requests.get(url)
#     filepath = DATA_DIR / filename
#     filepath.write_bytes(response.content)
#     print(f"✓ Đã tải: {filepath}")
#
# Nếu trang là HTML thuần (không phải PDF sẵn), có thể convert nội dung text
# thành PDF đơn giản bằng thư viện fpdf2 (đã có trong requirements.txt).


if __name__ == "__main__":
    collect_legal_docs()
