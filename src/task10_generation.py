"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"

Gợi ý LLM: OpenRouter có nhiều model gắn hậu tố ":free" không tính phí — xem
https://openrouter.ai/models?max_price=0 — phù hợp nếu chưa có credit trả phí.
Base URL: "https://openrouter.ai/api/v1", dùng chung interface với OpenAI SDK.
"""

import os
import re
import sys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .task9_retrieval_pipeline import retrieve
from .task4_chunking_indexing import load_index


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

# TODO: Chọn LLM model (OpenRouter model ID)
LLM_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
LLM_FALLBACK_MODELS = (
    LLM_MODEL,
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-20b:free",
)


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về chính sách thương mại điện tử và hỗ trợ
khách hàng (thanh toán, đổi trả, giao hàng, quyền riêng tư, quy định người bán).

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. Trả lời đúng ý định câu hỏi; không thay một câu hướng dẫn bằng danh sách thông tin liên quan
3. Mỗi khẳng định phải có trích dẫn bằng đúng tên Source trong context, ví dụ: [returns-refund-policy.md, 2026]
4. Nếu context không trả lời trực tiếp câu hỏi → trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có"
5. Trả lời bằng tiếng Việt, tối đa 250 từ; ưu tiên các bước hoặc tối đa 6 gạch đầu dòng
6. Không chép nguyên context, không suy luận hay mở rộng ngoài nguồn"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return list(chunks)
    return chunks[::2] + chunks[1::2][::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    return "\n---\n".join(
        f"[Document {i} | Source: {chunk.get('metadata', {}).get('source', f'Source {i}')} | Type: {chunk.get('metadata', {}).get('type', 'unknown')}]\n{chunk['content']}"
        for i, chunk in enumerate(chunks, 1)
    )


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {"answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.", "sources": [], "retrieval_source": "none"}
    reordered = reorder_for_llm(chunks)
    expanded = [
        {**chunk, "content": text}
        for chunk, text in _evidence_by_source(reordered)[:3]
    ]
    context = format_context(expanded)
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=20.0, max_retries=0)
            answer = None
            last_error = None
            used_model = None
            models = dict.fromkeys(LLM_FALLBACK_MODELS)
            for model in models:
                try:
                    response = client.chat.completions.create(model=model, messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}], temperature=TEMPERATURE, top_p=TOP_P, max_tokens=1000)
                    answer = response.choices[0].message.content
                    if answer and answer.strip():
                        used_model = model
                        break
                    last_error = RuntimeError(f"{model} returned empty content")
                except Exception as error:
                    last_error = error
            if not answer or not answer.strip():
                raise last_error or RuntimeError("LLM provider returned no answer")
        except Exception as error:
            print(f"All generation providers failed: {error}", file=sys.stderr)
            answer = _extractive_answer(query, chunks)
    else:
        answer = _extractive_answer(query, chunks)
    # Nếu LLM xác nhận context không đủ, các chunks truy xuất chỉ là candidates
    # không liên quan và không được trình bày như "nguồn tham khảo đã dùng".
    refusal_markers = (
        "tôi không thể xác minh",
        "không thể xác minh thông tin này",
        "i cannot verify this information",
    )
    is_refusal = any(marker in answer.lower() for marker in refusal_markers)
    return {
        "answer": answer,
        "sources": [] if is_refusal else chunks,
        "retrieval_source": "none" if is_refusal else chunks[0].get("source", "hybrid"),
    }
    # chunks = retrieve(query, top_k=top_k)
    #
    # # Step 2: Reorder
    # reordered = reorder_for_llm(chunks)
    #
    # # Step 3: Format context
    # context = format_context(reordered)
    #
    # # Step 4: Build prompt
    # user_message = f"""Context:\n{context}\n\n---\n\nQuestion: {query}"""
    #
    # # Step 5: Call LLM (OpenRouter — OpenAI-compatible API)
    # from openai import OpenAI
    # api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    # client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    #
    # response = client.chat.completions.create(
    #     model=LLM_MODEL,
    #     messages=[
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": user_message}
    #     ],
    #     temperature=TEMPERATURE,
    #     top_p=TOP_P,
    # )
    #
    # answer = response.choices[0].message.content
    #
    # # Step 6: Return
    # return {
    #     "answer": answer,
    #     "sources": chunks,
    #     "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
    # }


def _normalized_text(text: str) -> str:
    """Loại khoảng trắng/metadata thừa từ nội dung Help Center đã crawl."""
    return re.sub(r"\s+", " ", text).strip()


def _citation(chunk: dict) -> str:
    source = chunk.get("metadata", {}).get("source", "Nguồn nội bộ")
    return f"[{source}, 2026]"


def _evidence_by_source(chunks: list[dict]) -> list[tuple[dict, str]]:
    """Ghép chunk liền kề để khôi phục câu/danh sách bị cắt ở ranh giới."""
    index = load_index()
    grouped = []
    seen_sources = set()
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        source = metadata.get("source")
        if not source or source in seen_sources:
            continue
        seen_sources.add(source)
        retrieved_indices = {
            int(item.get("metadata", {}).get("chunk_index", -999))
            for item in chunks
            if item.get("metadata", {}).get("source") == source
        }
        wanted = {near for idx in retrieved_indices for near in (idx - 1, idx, idx + 1)}
        related = sorted(
            (
                item for item in index
                if item.get("metadata", {}).get("source") == source
                and int(item.get("metadata", {}).get("chunk_index", -999)) in wanted
            ),
            key=lambda item: int(item.get("metadata", {}).get("chunk_index", 0)),
        )
        text = _normalized_text(" ".join(item["content"] for item in related))
        grouped.append((chunk, text or _normalized_text(chunk["content"])))
    return grouped


def _extractive_answer(query: str, chunks: list[dict]) -> str:
    """Provider lỗi thì từ chối an toàn, không đoán bằng một chunk rời rạc."""
    return "Tôi không thể xác minh thông tin này từ nguồn hiện có."


if __name__ == "__main__":
    test_queries = [
        "Shopee hỗ trợ những phương thức thanh toán nào?",
        "Làm sao để yêu cầu đổi trả hay hoàn tiền?",
        "Cần chuẩn bị bằng chứng gì khi yêu cầu hoàn tiền?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
