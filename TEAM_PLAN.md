# Kế Hoạch Phân Công Nhóm — Day 8 RAG Pipeline v2

**Nhóm 4 người** — bám theo 7 checkpoint (180 phút) trong `checkpoint_timer.html`.

| Thành viên | Role | Phụ trách chính |
|---|---|---|
| Nguyễn Duy Hưng | **R1** — Team Leader & RAG Architect | `supervisor.py`, Task 7, Task 9 (tích hợp pipeline) |
| Nguyễn Hoàng Thảo Tiên | **R2** — Data & Retrieval Specialist | Task 1, 2, 3, 4, 5 |
| Đoàn Duy Chiến | **R3** — Frontend & Chatbot Developer | Task 6, Task 8, Task 10, `app.py` |
| Trần Bảo Phúc | **R4** — Evaluation & QA Engineer | `golden_dataset.json`, `eval_pipeline.py`, `results.md`, QA hỗ trợ |

---

## 1. Nguyên tắc chống chờ đợi

Điểm nghẽn lớn nhất của pipeline là chuỗi phụ thuộc **Task 1→2→3→4→5**, toàn bộ do R2 phụ trách. Nếu các role khác ngồi chờ R2 xong mới bắt đầu, cả nhóm sẽ mất ~50 phút đầu (CP1+CP2). Để tránh việc này:

1. **Dùng contract (function signature) đã có sẵn trong code, không chờ implementation thật.** Tất cả các file `taskN_*.py` đã định nghĩa sẵn signature + docstring format trả về (xem mục 4 bên dưới). Ai cần gọi hàm của người khác thì code trước theo đúng format đó, test bằng **mock data** tự tạo.
2. **R3 không phụ thuộc R2 để bắt đầu Task 6 và Task 8**:
   - Task 6 (BM25) chỉ cần **văn bản thô** (list các đoạn text) — có thể tự lấy 5-10 đoạn mẫu từ 1-2 file Shopee tải thủ công trong 5 phút, không cần đợi R2 convert xong toàn bộ `data/standardized/`.
   - Task 8 (PageIndex) dùng SDK/API riêng, tự upload 1-2 file PDF mẫu của chính mình — hoàn toàn tách biệt khỏi ChromaDB của R2.
3. **R1 code Task 7 (RRF) bằng dữ liệu giả lập** ngay từ CP2 (không cần đợi Task 5/6 thật xong) — vì RRF chỉ cần input là 2 list `[{"content", "score", "metadata"}]`, tự tạo list giả 5-10 phần tử là đủ để viết và test hàm `rerank_rrf()`.
4. **R4 không phụ thuộc ai để bắt đầu** — soạn 15 câu hỏi golden dataset dựa trên *chủ đề* (chính sách Shopee) chứ không cần chunk thật. Chỉ cần biết chủ đề tài liệu (đã chốt ngay từ đầu) là viết được câu hỏi + đáp án kỳ vọng.
5. **Điểm đồng bộ bắt buộc (sync point)** — mọi người dừng code, báo cáo nhanh 2 phút:
   - Cuối CP1 (0:35): R2 xác nhận `data/standardized/` đã có file thật → R3 (Task 6) và R1 (Task 7) chuyển từ mock data sang data thật.
   - Cuối CP2 (1:00): R2 xác nhận Task 4+5 chạy được → R1 bắt đầu ráp Task 9 thật (thay vì mock).
   - Cuối CP3 (1:20): R1 xác nhận Task 7+8 sẵn sàng → R1 ráp `retrieve()` hoàn chỉnh.
   - Cuối CP4 (1:45): `pytest tests/test_individual.py` phải xanh 35/35 trước khi qua bài nhóm.

---

## 2. Sơ đồ phụ thuộc (dependency)

```
Task1 ─┐
Task2 ─┴→ Task3 ─→ Task4 ─→ Task5 (R2) ──┐
                      │                    ├─→ Task7 (R1) ─┐
                Task3 ─┴───────────────────┤                ├─→ Task9 (R1) ─→ Task10 (R3) ─→ app.py (R3)
                                     Task6 (R3) ────────────┘        ↑                              │
                                                                       │                              │
                                                       Task8 (R3, độc lập) ─────────────────────┘                              │
                                                                                                                                 ↓
                                                                                          golden_dataset.json (R4, độc lập từ đầu)
                                                                                                                                 ↓
                                                                                          eval_pipeline.py + results.md (R4, cần Task9+10 xong)
```

**Chỉ 2 người thực sự phải chờ nhau:** R1 (Task 9) phải chờ Task 5/6/7/8 xong thật; R4 (chạy eval thật) phải chờ Task 9/10 xong thật. Mọi việc khác làm song song được.

---

## 3. Lịch trình chi tiết theo checkpoint

### 🟦 CP0 — Setup (0:00–0:10, 10 phút)

| Ai | Việc | Output |
|---|---|---|
| R1 | Tạo repo chung, share `.env` (điền `OPENROUTER_API_KEY`, `PAGEINDEX_API_KEY`) | `.env` mọi người có |
| R2 | `python -m venv .venv`, `pip install -r requirements.txt`, test `import chromadb, sentence_transformers` | venv sẵn sàng |
| R3 | Test `streamlit run app.py` chạy được (dù chưa có nội dung) | Streamlit OK |
| R4 | `pip install ragas datasets`, test import | Lib sẵn sàng |

**Chốt chung 2 phút cuối:** cả nhóm thống nhất 4 nguồn PDF (Task 1) + 5 bài viết (Task 2) để R2 không phải tự chọn một mình → tránh phải làm lại.

---

### 🟩 CP1 — Thu thập & chuẩn hoá dữ liệu (0:10–0:35, 25 phút)

| Ai | Việc chính | Việc song song (không chờ) |
|---|---|---|
| **R2** | Task 1 (tải ≥3 PDF Shopee → `data/landing/legal/`) → Task 2 (crawl ≥5 bài → `data/landing/news/`) → Task 3 (`python src/task3_convert_markdown.py`) | — |
| **R1** | Đọc kỹ bẫy fallback trong `task9_retrieval_pipeline.py` (đã có sẵn comment chi tiết), phác thảo `supervisor.py` skeleton, viết trước hàm `rerank_rrf()` (Task 7) dùng 2 list mock (tự tay gõ 5 dict giả) | Không đụng file của R2 |
| **R3** | Tự tải 1-2 file PDF Shopee bất kỳ (không trùng R2) về máy cá nhân → tự tay copy 5-10 đoạn text ngắn làm corpus giả → code `build_bm25_index()` + `lexical_search()` (Task 6) với corpus giả này. Song song: đăng ký tài khoản pageindex.ai, đọc SDK | Không cần `data/standardized/` thật |
| **R4** | Đọc 4 chủ đề chính sách (đổi trả, thanh toán, bảo mật, seller listing) đã chốt ở CP0 → bắt đầu soạn câu hỏi cho `golden_dataset.json` (question + expected_answer + expected_context) dựa trên hiểu biết chung về chính sách TMĐT | Không cần chờ file thật |

✅ **Cuối CP1:** R2 báo "đã có file trong `data/standardized/`" → R3 và R1 chuyển sang dùng data thật ở CP2.

---

### 🟩 CP2 — Chunking, Indexing & Search cơ bản (0:35–1:00, 25 phút)

| Ai | Việc chính | Ghi chú |
|---|---|---|
| **R2** | Task 4 (`python src/task4_chunking_indexing.py` — chunk_size=800, overlap=100, gắn `customer_role`, embed `BAAI/bge-m3`, lưu ChromaDB) → Task 5 (`semantic_search()`) | Việc nặng nhất, ưu tiên không bị làm phiền |
| **R1** | Hoàn thiện Task 7 với data thật (đổi từ mock sang gọi `semantic_search`/`lexical_search` thật nếu R2/R3 đã xong; nếu chưa, tiếp tục dùng mock, chỉ đổi input source sau) | Có thể test độc lập bằng `pytest tests/test_individual.py::TestTask7` |
| **R3** | Đổi corpus giả Task 6 → corpus thật (đọc từ `data/standardized/` bằng `pathlib.Path.rglob("*.md")`) ngay khi R2 báo đã convert xong. Sau đó rảnh → bắt đầu code khung `app.py` (giao diện chat tĩnh, chưa nối logic) | |
| **R4** | Tiếp tục soạn golden dataset, mục tiêu xong 15 câu trước CP3 | |

✅ **Cuối CP2:** chạy thử `pytest tests/test_individual.py -k "task4 or task5 or task6"` — Pass Criteria chính thức của checkpoint.

---

### 🟧 CP3 — Reranking & Vectorless Fallback (1:00–1:20, 20 phút)

| Ai | Việc chính |
|---|---|
| **R1** | Chốt `rerank_rrf()` (công thức $\sum \frac{1}{60+rank}$, k=60) — chạy `python src/task7_reranking.py` xác nhận output re-sorted đúng |
| **R2** | Đã xong Task 4-5 → hỗ trợ debug nếu R1/R3 gặp lỗi liên quan embedding/ChromaDB; tự thử vài query semantic vs lexical để cảm nhận sự khác biệt (chuẩn bị demo CP6) |
| **R3** | Hoàn thiện Task 8 (`upload_documents()` + `pageindex_search()`) — **nhớ `print(json.dumps(...))` response thật trước khi parse**, đừng đoán schema |
| **R4** | Test câu hỏi **ngoài domain** (vd: "thời tiết Hà Nội hôm nay") qua `semantic_search()` để tự đo điểm cosine — giúp R1 calibrate `SCORE_THRESHOLD` ở Task 9 (đừng copy nguyên 0.48, phải tự đo trên corpus của nhóm) |

✅ **Cuối CP3:** R1 đã có đủ Task 7 (RRF) + Task 8 (PageIndex) hoạt động độc lập để ráp vào Task 9 ở CP4.

---

### 🟧 CP4 — Pipeline hoàn chỉnh & Generation (1:20–1:45, 25 phút) — MỐC 50đ CÁ NHÂN

| Ai | Việc chính |
|---|---|
| **R1** | Hoàn thiện `retrieve()` trong `task9_retrieval_pipeline.py`: gọi `semantic_search` + `lexical_search` song song → `rerank_rrf` merge → `rerank` → **so ngưỡng với `dense_results[0]["score"]` (cosine gốc), KHÔNG so với điểm RRF đã fuse** → fallback `pageindex_search` nếu dưới ngưỡng |
| **R2** | Chạy full `pytest tests/test_individual.py -v`, hỗ trợ fix bug cho cả nhóm (đây là lúc R2 rảnh nhất vì Task 1-5 đã xong từ CP2-CP3) |
| **R3** | Task 10 (`task10_generation.py`): viết `reorder_for_llm()` (`front + back[::-1]`), format context + `SYSTEM_PROMPT` yêu cầu citation `[Nguồn, Năm]`, gọi LLM |
| **R4** | Rà soát format citation output của R3 có đúng `[Nguồn]` không; hoàn thiện nốt golden dataset (đủ 15 câu) |

✅ **Cuối CP4 — Pass Criteria bắt buộc:** `python -m pytest tests/test_individual.py -v` → **35/35 passed**. Nếu chưa xanh, ưu tiên fix trước khi sang bài nhóm.

---

### 🟨 CP5 — Bài nhóm: Chatbot UI & Đánh giá RAGAS (1:45–2:15, 30 phút)

| Ai | Việc chính |
|---|---|
| **R1** | Ghép code tối ưu nhất của cả nhóm vào nhánh chính, review tổng thể, theo dõi tiến độ báo cáo |
| **R2** | Nối `generate_with_citation()` (Task 10) vào luồng xử lý câu hỏi trong `app.py` cùng R3 |
| **R3** | Hoàn thiện Streamlit `app.py`: chat UI, thanh chỉnh `top_k`, khung hiển thị source documents, câu hỏi gợi ý |
| **R4** | Chạy `python -m group_project.evaluation.eval_pipeline` lấy 4 chỉ số (Faithfulness, Answer Relevancy, Context Recall, Context Precision) → so sánh A/B (vd: hybrid có rerank vs dense-only không rerank) → viết `group_project/evaluation/results.md` |

✅ **Cuối CP5:** Chatbot trả lời kèm nguồn; `results.md` có đủ bảng điểm + A/B comparison.

---

### 🟦 CP6 — Demo & Nộp bài (2:15–3:00, 45 phút)

| Ai | Việc chính |
|---|---|
| **R1** | Thuyết trình tổng quan kiến trúc RAG Pipeline (5-8 phút) |
| **R2** | Trả lời câu hỏi kỹ thuật về Hybrid Search / RRF / Fallback logic |
| **R3** | Live demo Streamlit trên màn hình |
| **R4** | Báo cáo kết quả RAGAS, phân tích Hybrid vs Dense-only |
| Cả nhóm | `git push origin main` trước khi hết giờ |

---

## 4. Contract giữa các module (để code song song không cần chờ)

Tất cả hàm retrieval trả về cùng 1 format để cắm lẫn nhau được ngay cả khi module khác chưa xong (dùng mock trả về đúng format này):

```python
list[dict]  # mỗi dict:
{
    "content": str,     # nội dung chunk
    "score": float,     # điểm liên quan (thang đo khác nhau theo module — semantic: [0,1], BM25: [0,∞))
    "metadata": dict,   # {"source": ..., "doc_type": ..., "chunk_index": ..., "customer_role": "buyer"/"seller"/"both"}
}
```

Riêng `pageindex_search()` và output cuối của `retrieve()` có thêm field `"source": "hybrid" | "pageindex"`.

**Mock nhanh để test khi module upstream chưa xong** (copy dùng ngay):
```python
MOCK_RESULTS = [
    {"content": "Shopee hỗ trợ hoàn tiền trong 15 ngày...", "score": 0.82, "metadata": {"source": "returns-refund-policy-shopee.md", "customer_role": "buyer"}},
    {"content": "Người bán chịu phí sàn 5% trên mỗi đơn...", "score": 0.65, "metadata": {"source": "seller-listing-regulations-shopee.md", "customer_role": "seller"}},
]
```

---

## 5. Rủi ro & cách xử lý khi bị block

| Tình huống | Xử lý |
|---|---|
| R2 trễ tiến độ Task 1-2 (trang bị chặn 403) | Dùng ngay bộ data mẫu có sẵn trong repo (đã đề cập trong `LAB_GUIDE.md` mục Troubleshooting) thay vì cố crawl tiếp |
| R1 xong Task 7 nhưng Task 5/6 vẫn chưa có data thật | Tiếp tục dùng mock, không đứng chờ — chuyển sang review code Task 9 comment sẵn, viết `supervisor.py` skeleton |
| R3 bị lỗi `crawl4ai`/Playwright khi hỗ trợ crawl | `playwright install chromium`, nếu vẫn lỗi thì bỏ qua, tập trung Task 6/8 |
| R4 chưa có Task 9/10 để chạy eval thật ở CP5 | Vẫn hoàn thiện `golden_dataset.json` trước, viết sẵn khung `eval_pipeline.py` gọi hàm `retrieve()` + `generate_with_citation()` qua import — chỉ cần chạy thử khi 2 hàm đó sẵn sàng |
| Cuối CP4 pytest chưa xanh hết | Ưu tiên tuyệt đối: cả 4 người dừng việc khác, mỗi người nhận 2-3 test fail gần task mình nhất để fix song song 5 phút cuối |
