"""Regression tests cho truy vấn ngoài phạm vi tài liệu."""

from src.task10_generation import generate_with_citation
from src.task9_retrieval_pipeline import retrieve


def test_out_of_domain_query_returns_no_sources():
    assert retrieve("who am i", top_k=3) == []


def test_out_of_domain_generation_refuses_to_guess():
    result = generate_with_citation("who am i", top_k=3)
    assert result["sources"] == []
    assert result["answer"] == "Tôi không thể xác minh thông tin này từ nguồn hiện có."


def test_refusal_does_not_present_irrelevant_sources(monkeypatch):
    """Có retrieval candidates nhưng LLM từ chối thì UI không được gắn nguồn giả."""
    import src.task10_generation as generation

    monkeypatch.setattr(generation, "retrieve", lambda *_args, **_kwargs: [{
        "content": "Payment methods do not mention Bitcoin.",
        "score": 0.8,
        "metadata": {"source": "payment-methods.md", "type": "legal"},
        "source": "hybrid",
    }])
    monkeypatch.setattr(generation, "_extractive_answer", lambda _query, _chunks: "Tôi không thể xác minh thông tin này từ nguồn hiện có.")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = generation.generate_with_citation("Có hỗ trợ Bitcoin không?")
    assert result["sources"] == []
    assert result["retrieval_source"] == "none"
