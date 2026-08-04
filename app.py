"""
RAG Chatbot UI — Frontend skeleton for CP2.

Đây là giao diện chat tĩnh của R3, được thiết kế như một khung frontend chuyên nghiệp.
UI hiện có:
- Sidebar câu hỏi gợi ý và cấu hình top_k
- Chat history và message bubbles
- Panel hiển thị nguồn tham khảo khi có
- Khung thông báo trạng thái và hướng dẫn

Task 10 / pipeline vẫn có thể chưa hoàn thiện; phần app.py chỉ chịu trách nhiệm dựng UI khung.
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Thêm project root vào sys.path để import các task từ src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Page settings
# =============================================================================

st.set_page_config(
    page_title="E-commerce Support RAG Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar() -> int:
    st.title("E-commerce Support RAG")
    st.caption(
        "Giao diện hỗ trợ truy vấn chính sách thương mại điện tử "
        "cho các lĩnh vực như đổi trả, thanh toán, bảo mật và quy định người bán."
    )

    st.divider()
    st.subheader("Câu hỏi gợi ý")
    suggestions = [
        "Thời hạn yêu cầu trả hàng/hoàn tiền là bao lâu?",
        "Shopee hỗ trợ những phương thức thanh toán nào?",
        "Làm sao để đổi phương thức thanh toán đơn hàng?",
        "Quy định về đăng bán sản phẩm cho người bán?",
        "Cách mua hàng trên Shopee của quốc gia khác?",
    ]
    for row in range(0, len(suggestions), 2):
        cols = st.columns(2)
        for idx, suggestion in enumerate(suggestions[row:row + 2]):
            if cols[idx].button(suggestion, use_container_width=True, key=f"suggestion_{row + idx}"):
                st.session_state.pending_query = suggestion

    st.divider()
    st.subheader("Thiết lập")
    top_k = st.slider("Số chunk retrieval", 3, 10, 5)
    st.caption("Điều chỉnh số lượng chunks tham gia retrieval để cân bằng độ rộng bằng chứng và hiệu suất.")

    st.divider()
    st.subheader("Kiến trúc hệ thống")
    st.info(
        "Hybrid Retrieval (Semantic + BM25) → RRF Rerank → PageIndex Fallback → LLM Generation với trích dẫn."
    )

    st.divider()
    st.caption("Giao diện này đang ở chế độ khung UI CP2, chưa yêu cầu kết nối hoàn chỉnh với backend.")
    return top_k


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Xin chào! Tôi là trợ lý hỗ trợ chính sách thương mại điện tử. "
                    "Hãy nhập câu hỏi của bạn về đổi trả, thanh toán, bảo mật hoặc quy định người bán."
                ),
                "sources": [],
            }
        ]
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None


def append_message(role: str, content: str, sources: list[dict] | None = None) -> None:
    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
            "sources": sources or [],
        }
    )


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander(f"Nguồn tham khảo ({len(sources)} chunks)"):
        for idx, src in enumerate(sources, start=1):
            meta = src.get("metadata", {})
            source_name = meta.get("source", "Unknown")
            doc_type = meta.get("type", "unknown")
            score = src.get("score", 0)
            st.markdown(f"**[{idx}] {source_name}** `{doc_type}` | score: `{score:.4f}`")
            st.write(src.get("content", "")[:300] + "...")
            if idx < len(sources):
                st.markdown("---")


# =============================================================================
# Application
# =============================================================================

init_session_state()

top_k = render_sidebar()

st.title("E-commerce Support RAG Chatbot")
st.markdown(
    "Hệ thống hỏi đáp chính sách thương mại điện tử và hỗ trợ khách hàng. "
    "Giao diện này là khung frontend CP2, thiết kế rõ ràng và dễ quan sát."
)

chat_col, info_col = st.columns([3, 1], gap="large")

with chat_col:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])

    user_input = st.chat_input("Nhập câu hỏi của bạn về chính sách/hỗ trợ thương mại điện tử...", placeholder="Nhập câu hỏi hoặc chọn từ gợi ý")
    query = user_input or st.session_state.pending_query

    if query:
        st.session_state.pending_query = None
        append_message("user", query)

        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."):
                try:
                    from src.task10_generation import generate_with_citation

                    response = generate_with_citation(query, top_k=top_k)
                    answer = response.get("answer", "Xin lỗi, tôi chưa thể trả lời câu hỏi này.")
                    sources = response.get("sources", [])
                except NotImplementedError:
                    answer = (
                        "Task 10 chưa được implement. Hãy hoàn thành `src/task10_generation.py` "
                        "để tích hợp pipeline vào UI."
                    )
                    sources = []
                except Exception as error:
                    answer = f"Lỗi khi chạy RAG Pipeline: {error}"
                    sources = []

                st.markdown(answer)
                render_sources(sources)

        append_message("assistant", answer, sources)

with info_col:
    st.subheader("Hướng dẫn sử dụng")
    st.markdown(
        "Chọn câu hỏi gợi ý hoặc nhập câu hỏi tùy ý."
        "\n- `top_k` điều chỉnh số lượng chunks retrieval sẽ được sử dụng cho pipeline."
        "\n- Khi Task 10 chưa hoàn thiện, UI vẫn hoạt động như một khung frontend để chuẩn bị tích hợp."
    )

    st.divider()
    st.subheader("Trạng thái hiện tại")
    st.info(
        "Giao diện đã sẵn sàng cho R3: chat history, câu hỏi gợi ý, input và panel nguồn tham khảo."
    )
    st.markdown("- Lịch sử chat được lưu trong phiên làm việc.")
    st.markdown("- Các nguồn tham khảo sẽ hiển thị khi pipeline trả về `sources`.")
    st.markdown("- Có thể mở rộng thêm tab tài liệu hoặc bộ lọc metadata.")

    st.divider()
    st.subheader("Mở rộng giao diện")
    st.markdown("1. Thêm nút Xóa lịch sử chat để làm sạch phiên và thử nghiệm nhanh.")
    st.markdown("2. Thêm panel preview document / source list để giảm mật độ text.")
    st.markdown("3. Tối ưu responsive cho layout khi hiển thị trên màn hình nhỏ.")

