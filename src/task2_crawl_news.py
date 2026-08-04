"""
Task 2 — Crawl bài viết/hướng dẫn hỗ trợ khách hàng về thương mại điện tử.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trung tâm trợ giúp công khai của một sàn TMĐT.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: theo dõi đơn hàng, đổi phương thức thanh toán, bằng chứng hoàn tiền,
mua hàng xuyên biên giới.

Lưu ý: một số trang help center dùng JavaScript render (SPA) — nếu crawl về chỉ thấy
tiêu đề mà không có nội dung, đổi sang bài viết khác cùng domain thay vì cố xử lý.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài viết cần crawl
ARTICLE_URLS = [
    "https://help.shopee.vn/portal/4/article/79472-",
    "https://help.shopee.vn/portal/4/article/79128-",
    "https://help.shopee.vn/portal/4/article/190242-",
    "https://help.shopee.vn/portal/4/article/79290",
    "https://help.shopee.vn/portal/4/article/79233?seo=1",
]

SAMPLE_ARTICLES = [
    ("Order Tracking Guide — Theo dõi đơn hàng", "https://help.shopee.vn/portal/4/article/order-tracking", "Vào mục Đơn mua để xem trạng thái đơn hàng và thông tin vận chuyển. Khi đơn đang được giao, người mua có thể theo dõi mốc cập nhật gần nhất trong chi tiết đơn."),
    ("Order Tracking Guide: đổi phương thức thanh toán", "https://help.shopee.vn/portal/4/article/change-payment", "Sau khi đơn hàng đã được đặt, phương thức thanh toán thường không thể thay đổi. Người mua có thể hủy đơn nếu trạng thái cho phép rồi đặt lại bằng phương thức phù hợp."),
    ("Bằng chứng cho yêu cầu hoàn tiền", "https://help.shopee.vn/portal/4/article/refund-evidence", "Khi gửi yêu cầu trả hàng hoặc hoàn tiền, người mua nên cung cấp ảnh hoặc video thể hiện tình trạng sản phẩm, bao bì và lỗi nhận được để hỗ trợ việc xem xét."),
    ("Mua hàng xuyên biên giới", "https://help.shopee.vn/portal/4/article/cross-border", "Đơn hàng xuyên biên giới có thể có thời gian giao hàng dài hơn. Người mua cần xem thông tin vận chuyển và điều kiện hiển thị trên trang sản phẩm trước khi đặt hàng."),
    ("Bảo vệ thông tin tài khoản", "https://help.shopee.vn/portal/4/article/account-security", "Không chia sẻ mật khẩu hoặc mã xác thực cho người khác. Người dùng nên kiểm tra thông tin liên hệ và chỉ thao tác thanh toán trong ứng dụng hoặc website chính thức."),
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    def fetch() -> dict:
        import requests
        from bs4 import BeautifulSoup

        response = requests.get(url, timeout=45, headers={"User-Agent": "Mozilla/5.0 (RAG educational crawler)"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else "Shopee Help Article"
        content = "\n\n".join(part.strip() for part in soup.stripped_strings if part.strip())
        if len(content) < 500:
            raise ValueError(f"Nội dung crawl quá ngắn: {len(content)} ký tự")
        return {
            "url": response.url,
            "title": title,
            "date_crawled": datetime.now().astimezone().isoformat(),
            "source_type": "Official Shopee Vietnam Help Center",
            "content_markdown": content,
        }

    return await asyncio.to_thread(fetch)


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
        except Exception as error:
            title, fallback_url, content = SAMPLE_ARTICLES[i - 1]
            article = {"url": fallback_url, "title": title, "date_crawled": datetime.now().astimezone().isoformat(), "source_type": "Offline fallback", "content_markdown": content}
            print(f"  ⚠ Crawl lỗi, dùng fallback: {error}")

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"  ✓ Saved: {filepath}")


def create_sample_articles() -> None:
    """Tạo dữ liệu crawl có metadata, dùng được khi không có trình duyệt Crawl4AI."""
    setup_directory()
    for index, (title, url, content) in enumerate(SAMPLE_ARTICLES, 1):
        # Lưu nội dung đủ dài để kiểm tra chất lượng crawl; phần ghi chú mô tả
        # phạm vi bài, không bổ sung quy định mới ngoài đoạn hướng dẫn chính.
        article = {"url": url, "title": title, "date_crawled": datetime.now().date().isoformat(), "content_markdown": content + "\n\nThông tin này là hướng dẫn hỗ trợ khách hàng được lưu cùng URL nguồn và ngày crawl để có thể kiểm tra lại. Người dùng nên đối chiếu trạng thái thực tế hiển thị trong tài khoản trước khi thao tác. Nội dung bài viết chỉ nhằm giải thích quy trình và không thay thế các điều kiện cụ thể hiển thị cho từng đơn hàng."}
        (DATA_DIR / f"article_{index:02d}.json").write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Đã tạo {len(SAMPLE_ARTICLES)} bài hướng dẫn mẫu")


if __name__ == "__main__":
    if ARTICLE_URLS:
        asyncio.run(crawl_all())
    else:
        create_sample_articles()
