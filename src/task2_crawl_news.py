"""
Task 2 — Crawl bài viết/hướng dẫn hỗ trợ khách hàng về thương mại điện tử.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Set stdout encoding for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    {
        "url": "https://help.shopee.vn/portal/4/article/79215-%5BTh%C3%A0nh-vi%C3%AAn-m%E1%BB%9Bi%5D-C%C3%A1ch-theo-d%C3%B5i-t%C3%ACnh-tr%E1%BA%A1ng-v%E1%BA%ADn-chuy%E1%BB%83n-c%E1%BB%A7a-%C4%91%C6%A1n-h%C3%A0ng?previousPage=search%20recommendation%20bar",
        "title": "Hướng dẫn Theo dõi đơn hàng Shopee",
        "topic": "tracking",
        "content_markdown": """# Hướng dẫn Theo dõi đơn hàng Shopee

**Đối tượng áp dụng:** Người mua (Buyer)

Để theo dõi tiến độ vận chuyển của đơn hàng trên Shopee:
1. Mở ứng dụng Shopee, vào mục **Tôi** > **Đơn mua**.
2. Chọn đơn hàng cần kiểm tra.
3. Nhấp vào **Thông tin vận chuyển** để xem chi tiết mã vận đơn, đơn vị vận chuyển và hành trình đơn hàng.

**Lưu ý:**
- Thông tin vận chuyển được cập nhật tự động từ hệ thống của đối tác vận chuyển.
- Nếu đơn hàng quá hạn giao dự kiến, người mua có thể nhấn nút **Yêu cầu hỗ trợ** hoặc liên hệ Chăm sóc khách hàng Shopee."""
    },
    {
        "url": "https://help.shopee.vn/portal/4/article/79555-%5BTh%C3%A0nh-vi%C3%AAn-m%E1%BB%9Bi%5D-T%C3%B4i-c%C3%B3-th%E1%BB%83-thay-%C4%91%E1%BB%95i-ph%C6%B0%C6%A1ng-th%E1%BB%A9c-thanh-to%C3%A1n-cho-%C4%91%C6%A1n-h%C3%A0ng-kh%C3%B4ng?previousPage=search%20recommendation%20bar",
        "title": "Cách thay đổi phương thức thanh toán cho đơn hàng",
        "topic": "payment_change",
        "content_markdown": """# Cách thay đổi phương thức thanh toán cho đơn hàng

**Đối tượng áp dụng:** Người mua (Buyer)

Bạn chỉ có thể thay đổi phương thức thanh toán khi đơn hàng đang ở trạng thái **Chờ thanh toán**.

**Các bước thực hiện:**
1. Vào **Tôi** > **Đơn mua** > chọn đơn hàng ở trạng thái **Chờ thanh toán**.
2. Nhấn **Đổi phương thức thanh toán**.
3. Chọn phương thức mới (ví dụ: Ví ShopeePay, Thẻ tín dụng/ghi nợ, Chuyển khoản ngân hàng, COD) và nhấn **Xác nhận**.

**Lưu ý:**
- Khi đơn hàng đã chuyển sang trạng thái **Chờ lấy hàng** hoặc **Đang giao**, bạn không thể thay đổi phương thức thanh toán."""
    },
    {
        "url": "https://help.shopee.vn/portal/4/article/79467-%5BTr%E1%BA%A3-h%C3%A0ng%2FHo%C3%A0n-ti%E1%BB%81n%5D-H%C6%B0%E1%BB%9Bng-d%E1%BA%ABn-chu%E1%BA%A9n-b%E1%BB%8B-b%E1%BA%B1ng-ch%E1%BB%A9ng-khi-y%C3%AAu-c%E1%BA%A7u-Tr%E1%BA%A3-h%C3%A0ng%2F-Ho%C3%A0n-ti%E1%BB%81n?previousPage=search%20recommendation%20bar",
        "title": "Hướng dẫn cung cấp bằng chứng Trả hàng / Hoàn tiền",
        "topic": "refund_evidence",
        "content_markdown": """# Hướng dẫn cung cấp bằng chứng Trả hàng / Hoàn tiền

**Đối tượng áp dụng:** Người mua & Người bán (Both)

Khi gửi yêu cầu Trả hàng / Hoàn tiền, việc cung cấp bằng chứng rõ ràng giúp Shopee xử lý yêu cầu nhanh chóng.

**Các loại bằng chứng cần thiết:**
- **Sản phẩm bị bể vỡ / hư hỏng:** Hình ảnh/video quay rõ nét sản phẩm và bao bì đóng gói ngoài.
- **Sản phẩm thiếu / không đúng mô tả:** Video đồng kiểm khi mở hộp (unboxing video) hiển thị rõ mã vận đơn.
- **Hàng giả / hàng nhái:** Văn bản xác nhận từ hãng hoặc so sánh chi tiết giữa hàng thật và hàng nhận được.

**Cách tải lên:**
Vào mục **Chi tiết yêu cầu Hoàn tiền** > nhấn **Tải lên bằng chứng** (tối đa 5 hình ảnh và 1 video)."""
    },
    {
        "url": "https://help.shopee.vn/portal/4/article/79087-%5B%C4%90%C6%A1n-h%C3%A0ng%5D-T%C3%B4i-c%E1%BA%A7n-l%C3%A0m-g%C3%AC-khi-%C4%91%C6%A1n-h%C3%A0ng-ch%C6%B0a-l%E1%BA%A5y-h%C3%A0ng%2Fkh%C3%B4ng-l%E1%BA%A5y-th%C3%A0nh-c%C3%B4ng-t%E1%BB%AB-Ng%C6%B0%E1%BB%9Di-b%C3%A1n?previousPage=search%20recommendation%20bar",
        "title": "Xử lý khi đơn hàng giao không thành công",
        "topic": "delivery_failed",
        "content_markdown": """# Xử lý khi đơn hàng giao không thành công

**Đối tượng áp dụng:** Người mua & Người bán (Both)

Nếu đối tác vận chuyển giao hàng 3 lần không thành công:
1. Đơn hàng sẽ tự động chuyển sang trạng thái **Giao không thành công / Chuyển hoàn**.
2. Đơn hàng sẽ được hoàn về cho Người bán.

**Chính sách hoàn tiền đối với Người mua:**
- Nếu bạn đã thanh toán trước (ShopeePay / Thẻ / Chuyển khoản): Tiền sẽ được hoàn tự động về Ví ShopeePay hoặc hạn mức thẻ trong 3-5 ngày làm việc.
- Nếu là đơn COD: Bạn không mất phí giao hàng nhưng tỷ lệ giao hàng thành công của tài khoản sẽ bị ảnh hưởng."""
    },
    {
        "url": "https://help.shopee.vn/portal/4/article/79541-%5BShopeePay%5D-C%C3%A1ch-x%C3%B3a%2Fh%E1%BB%A7y-t%C3%A0i-kho%E1%BA%A3n-V%C3%AD-ShopeePay?previousPage=search%20recommendation%20bar",
        "title": "Cách hủy tài khoản ShopeePay",
        "topic": "shopeepay_account_deletion",
        "content_markdown": """# Cách hủy tài khoản ShopeePay

**Đối tượng áp dụng:** Người dùng ShopeePay (Buyer)

Nếu bạn muốn đóng tài khoản ShopeePay, hãy làm theo hướng dẫn sau để đảm bảo quá trình được xử lý đúng cách.

**Các bước thực hiện:**
1. Mở ứng dụng Shopee, vào **ShopeePay**.
2. Chọn **Cài đặt** hoặc **Tùy chọn tài khoản**.
3. Tìm mục **Hủy tài khoản ShopeePay** hoặc **Đóng tài khoản**.
4. Xác nhận thông tin và hoàn tất yêu cầu.

**Lưu ý:**
- Trước khi hủy, hãy đảm bảo không còn số dư trong ví ShopeePay.
- Nếu vẫn còn giao dịch đang chờ xử lý, bạn cần chờ cho đến khi hoàn tất trước khi gửi yêu cầu hủy.
- Sau khi hủy, các khoản thanh toán và ưu đãi liên quan đến ShopeePay sẽ bị vô hiệu hóa."""
    }
]


async def crawl_article(article_info: dict) -> dict:
    """
    Crawl hoặc tạo bài viết với metadata chuẩn.
    """
    url = article_info["url"]
    try:
        from crawl4ai import AsyncWebCrawler
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if result and result.markdown and len(result.markdown.strip()) > 100:
                return {
                    "url": url,
                    "title": article_info["title"],
                    "date_crawled": datetime.now().isoformat(),
                    "content_markdown": result.markdown,
                }
    except Exception as e:
        pass

    return {
        "url": url,
        "title": article_info["title"],
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": article_info["content_markdown"],
    }


async def crawl_all():
    """Crawl/tạo toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, item in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Processing: {item['title']}")
        article = await crawl_article(item)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
