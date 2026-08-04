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

def reset_chat_history() -> None:
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
    st.session_state.pending_query = None


def render_sidebar() -> int:
    with st.sidebar:
        st.title("E-commerce Policy Support")
        st.caption(
            "Tìm kiếm câu trả lời chính sách thương mại điện tử dựa trên văn bản tham khảo thực tế."
        )

        st.divider()
        st.subheader("Câu hỏi gợi ý")
        suggestions = [
            "Thời hạn yêu cầu trả hàng/hoàn tiền là bao lâu?",
            "Làm sao để hủy tài khoản ShoppePay?",
            "Cách trả hàng khi đơn phát sai kích thước?",
            "Quy định đăng bán sản phẩm cho người bán?",
            "Chính sách bảo mật thông tin khách hàng",
        ]
        for row in range(0, len(suggestions), 2):
            cols = st.columns(2)
            for idx, suggestion in enumerate(suggestions[row:row + 2]):
                if cols[idx].button(suggestion, use_container_width=True, key=f"suggestion_{row + idx}"):
                    st.session_state.pending_query = suggestion

        st.divider()
        st.subheader("Thiết lập truy vấn")
        top_k = st.slider("Số chunk retrieval", 3, 10, 5)
        st.caption("Điều chỉnh số lượng tài liệu tham khảo được lấy về để cân bằng độ chính xác và độ chi tiết.")

        st.divider()
        if st.button("Xóa lịch sử chat", use_container_width=True):
            reset_chat_history()

        st.divider()
        st.subheader("Kiến trúc hệ thống")
        st.markdown(
            "Hybrid Retrieval → RRF Rerank → PageIndex Fallback → LLM Generation với trích dẫn."
        )
        st.caption("Luồng này giúp đảm bảo câu trả lời vừa đúng ngữ cảnh vừa có nguồn tham khảo rõ ràng.")

        st.divider()
        st.caption(
            "Giao diện này được thiết kế chuyên nghiệp cho CP5, không dùng icon trang trí, chỉ dùng cấu trúc rõ ràng và trực quan."
        )
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

st.header("E-commerce Support RAG Chatbot")
st.markdown(
    "Chatbot hỗ trợ chính sách thương mại điện tử với câu trả lời dẫn chứng từ tài liệu. "
    "Giao diện được thiết kế rõ ràng, chuyên nghiệp và tối ưu cho quy trình hỏi đáp CP5."
)

chat_col, info_col = st.columns([3, 1], gap="large")

with chat_col:
    st.subheader("Lịch sử hội thoại")
    for msg in st.session_state.messages:
        role_label = "Trợ lý" if msg["role"] == "assistant" else "Bạn"
        st.markdown(f"**{role_label}:**")
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])
        st.markdown("---")

    with st.form("query_form"):
        user_input = st.text_area(
            "Nhập câu hỏi của bạn về chính sách/hỗ trợ thương mại điện tử...",
            placeholder="Nhập câu hỏi hoặc chọn một gợi ý",
            height=120,
        )
        submit = st.form_submit_button("Gửi câu hỏi")

    query = None
    if submit and user_input:
        query = user_input
    elif st.session_state.pending_query:
        query = st.session_state.pending_query

    if query:
        st.session_state.pending_query = None
        append_message("user", query)

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
    st.markdown(
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

