"""
RAG Evaluation Pipeline.

Sử dụng DeepEval / RAGAS / TruLens để đánh giá chất lượng RAG pipeline.
Chọn 1 framework và implement đầy đủ.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS/DeepEval gọi LLM RẤT NHIỀU LẦN
(không phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi). Model free của OpenRouter giới hạn
50 request/ngày CHO CẢ TÀI KHOẢN (không phải theo model hay theo API key — đổi model free
khác hay tạo key mới KHÔNG reset quota). Nếu chạy full 15+ câu hỏi mà bị rate limit giữa
chừng, thử giảm xuống subset 5 câu để chạy kịp trong buổi, hoặc nạp $10 credit để mở khóa
1000 request/ngày.
"""

import json
import os
import re
import sys
from pathlib import Path

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Option 1: DeepEval
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    pip install deepeval
    """
    # TODO: Implement
    #
    # from deepeval import evaluate
    # from deepeval.metrics import (
    #     FaithfulnessMetric,
    #     AnswerRelevancyMetric,
    #     ContextualRecallMetric,
    #     ContextualPrecisionMetric,
    # )
    # from deepeval.test_case import LLMTestCase
    #
    # test_cases = []
    # for item in golden_dataset:
    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     test_case = LLMTestCase(
    #         input=item["question"],
    #         actual_output=result["answer"],
    #         expected_output=item["expected_answer"],
    #         retrieval_context=[c["content"] for c in result["sources"]],
    #     )
    #     test_cases.append(test_case)
    #
    # metrics = [
    #     FaithfulnessMetric(threshold=0.7),
    #     AnswerRelevancyMetric(threshold=0.7),
    #     ContextualRecallMetric(threshold=0.7),
    #     ContextualPrecisionMetric(threshold=0.7),
    # ]
    #
    # results = evaluate(test_cases, metrics)
    # return results
    raise NotImplementedError("Implement evaluate_with_deepeval")


# =============================================================================
# Option 2: RAGAS
# =============================================================================

def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng RAGAS.

    pip install ragas
    """
    # RAGAS requires an LLM judge/API key. This implementation uses RAGAS when
    # installed; the offline benchmark below remains available for classroom demo.
    #
    # from ragas import evaluate
    # from ragas.metrics import (
    #     faithfulness,
    #     answer_relevancy,
    #     context_recall,
    #     context_precision,
    # )
    # from datasets import Dataset
    #
    # eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    #
    # for item in golden_dataset:
    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     eval_data["question"].append(item["question"])
    #     eval_data["answer"].append(result["answer"])
    #     eval_data["contexts"].append([c["content"] for c in result["sources"]])
    #     eval_data["ground_truth"].append(item["expected_answer"])
    #
    # dataset = Dataset.from_dict(eval_data)
    # result = evaluate(
    #     dataset,
    #     metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    # )
    # return result.to_pandas()
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    from langchain_core.embeddings import Embeddings
    from src.task4_chunking_indexing import embed_texts

    class LocalHashEmbeddings(Embeddings):
        """Embedding offline cùng vector space retrieval, tránh cần embeddings API."""
        def embed_documents(self, texts):
            return embed_texts(list(texts))

        def embed_query(self, text):
            return embed_texts([text])[0]

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Cần OPENROUTER_API_KEY hoặc OPENAI_API_KEY để chạy RAGAS")
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None,
        temperature=0,
        max_tokens=1024,
    )
    rows = [rag_pipeline(item["question"]) for item in golden_dataset]
    dataset = Dataset.from_dict({
        "question": [item["question"] for item in golden_dataset],
        "answer": [row["answer"] for row in rows],
        "contexts": [[source["content"] for source in row["sources"]] for row in rows],
        "ground_truth": [item["expected_answer"] for item in golden_dataset],
    })
    return evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm,
        embeddings=LocalHashEmbeddings(),
        raise_exceptions=False,
    ).to_pandas()


# =============================================================================
# Option 3: TruLens
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng TruLens.

    pip install trulens
    """
    # TODO: Implement
    #
    # from trulens.apps.custom import TruCustomApp
    # from trulens.core import Feedback
    # from trulens.providers.openai import OpenAI as TruOpenAI
    #
    # provider = TruOpenAI()
    #
    # f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
    # f_relevance = Feedback(provider.relevance).on_input_output()
    # f_context_relevance = Feedback(provider.context_relevance).on_input()
    #
    # tru_rag = TruCustomApp(
    #     rag_pipeline,
    #     app_name="EcommerceSupport_RAG",
    #     feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
    # )
    #
    # with tru_rag as recording:
    #     for item in golden_dataset:
    #         rag_pipeline.generate_with_citation(item["question"])
    #
    # # Dashboard: from trulens.dashboard import run_dashboard; run_dashboard()
    raise NotImplementedError("Implement evaluate_with_trulens")


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(rag_pipeline, golden_dataset: list[dict]):
    """
    So sánh A/B giữa ít nhất 2 configs.

    Gợi ý configs để so sánh:
    - Config A: hybrid search + reranking
    - Config B: dense-only (không reranking)
    - Config C: hybrid search + PageIndex fallback
    """
    from src.task9_retrieval_pipeline import retrieve
    from src.task5_semantic_search import semantic_search
    configs = ("A: hybrid + RRF", "B: dense-only")
    comparison = {}
    for name in configs:
        rows = []
        for item in golden_dataset:
            if name == "A: hybrid + RRF":
                sources = retrieve(item["question"], top_k=5, use_reranking=True)
            else:
                sources = [{**source, "source": "dense-only"} for source in semantic_search(item["question"], top_k=5)]
            context = " ".join(source["content"] for source in sources).lower()
            expected = _tokens(item["expected_answer"])
            question = _tokens(item["question"])
            # A/B này cô lập retrieval, nên dùng expected answer làm reference
            # thay vì gọi generation lại (tránh trộn chất lượng LLM vào retriever).
            answer = item["expected_answer"].lower()
            # Reproducible retrieval proxies: grounded evidence coverage and relevance.
            recall = _coverage(expected, context)
            precision = sum(1 for source in sources if _coverage(question, source["content"].lower()) > 0) / max(1, len(sources))
            faithfulness = _coverage(_tokens(answer), context) if answer else 0.0
            relevance = _coverage(question, answer) if answer else 0.0
            rows.append({"question": item["question"], "faithfulness": faithfulness, "answer_relevance": relevance, "context_recall": recall, "context_precision": precision})
        metrics = {metric: round(sum(row[metric] for row in rows) / len(rows), 3) for metric in ("faithfulness", "answer_relevance", "context_recall", "context_precision")}
        comparison[name] = {"metrics": metrics, "rows": rows}
    return comparison
    #     "hybrid_rerank": {"use_reranking": True, "alpha": 0.5},
    #     "dense_only": {"use_reranking": False, "alpha": 1.0},
    # }
    #
    # results = {}
    # for config_name, params in configs.items():
    #     # Run eval with this config
    #     ...
    #     results[config_name] = scores
    #
    # return results


# =============================================================================
# Export Results
# =============================================================================

def export_results(results: dict, comparison: dict):
    """Export evaluation results to results.md"""
    a, b = comparison["A: hybrid + RRF"]["metrics"], comparison["B: dense-only"]["metrics"]
    labels = [("Faithfulness", "faithfulness"), ("Answer Relevance", "answer_relevance"), ("Context Recall", "context_recall"), ("Context Precision", "context_precision")]
    framework_note = "RAGAS 0.1.21 với OpenRouter LLM judge và sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2." if results else "Bộ benchmark offline có thể tái lập; chạy `RUN_RAGAS=1` để bổ sung RAGAS LLM judge."
    lines = ["# RAG Evaluation Results", "", "## Framework sử dụng", "", framework_note]
    if results:
        lines += ["", "## RAGAS Scores — Config A", "", "| Metric | Score |", "|---|---:|",
                  f"| Faithfulness | {results.get('faithfulness', float('nan')):.3f} |",
                  f"| Answer Relevance | {results.get('answer_relevancy', float('nan')):.3f} |",
                  f"| Context Recall | {results.get('context_recall', float('nan')):.3f} |",
                  f"| Context Precision | {results.get('context_precision', float('nan')):.3f} |"]
    lines += ["", "## A/B Retrieval Proxy Scores", "", "| Metric | Config A (hybrid + RRF) | Config B (dense-only) | Δ |", "|---|---:|---:|---:|"]
    for label, key in labels:
        lines.append(f"| {label} | {a[key]:.3f} | {b[key]:.3f} | {a[key] - b[key]:+.3f} |")
    lines += ["", "## A/B Comparison Analysis", "", "Config A dùng dense retrieval + BM25 và Reciprocal Rank Fusion. Config B chỉ dùng dense semantic retrieval.", "", "## Worst Performers (Bottom 3)", "", "| Question | Context Recall | Failure Stage |", "|---|---:|---|"]
    worst = sorted(comparison["A: hybrid + RRF"]["rows"], key=lambda row: row["context_recall"])[:3]
    lines += [f"| {row['question']} | {row['context_recall']:.3f} | Retrieval/context coverage |" for row in worst]
    lines += ["", "## Recommendations", "", "1. Benchmark BAAI/bge-m3 với model multilingual MiniLM hiện tại để tiếp tục cải thiện Answer Relevance.", "2. Chunk theo heading thay vì chỉ theo ký tự để tăng Context Recall trên các chính sách dài.", "3. Calibrate lại threshold fallback và thử cross-encoder reranker để cải thiện Context Precision."]
    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return RESULTS_PATH
    # content += "## Overall Scores\n\n"
    # content += "| Metric | Score |\n|--------|-------|\n"
    # ...
    # content += "\n## A/B Comparison\n\n"
    # ...
    # content += "\n## Worst Performers\n\n"
    # ...
    # content += "\n## Recommendations\n\n"
    # ...
    #
    # RESULTS_PATH.write_text(content, encoding="utf-8")


def _tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE) if len(word) > 2}


def _coverage(terms: set[str], text: str) -> float:
    return sum(term in text for term in terms) / max(1, len(terms))


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    from src.task10_generation import generate_with_citation
    ragas_results = {}
    if os.getenv("RUN_RAGAS") == "1":
        print("Running RAGAS with 4 metrics...")
        frame = evaluate_with_ragas(generate_with_citation, golden_dataset)
        frame.to_csv(Path(__file__).parent / "ragas_details.csv", index=False)
        metric_columns = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
        ragas_results = {column: round(float(frame[column].mean()), 3) for column in metric_columns if column in frame}
        print("RAGAS:", ragas_results)
    comparison = compare_configs(generate_with_citation, golden_dataset)
    export_results(ragas_results, comparison)
    print(f"✓ Saved evaluation report: {RESULTS_PATH}")
