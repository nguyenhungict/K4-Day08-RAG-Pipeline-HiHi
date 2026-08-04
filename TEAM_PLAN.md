# Kế Hoạch Phân Công Nhóm — Day 8 RAG Pipeline v2

**Nhóm 4 người** — bám sát đúng phân vai theo checkpoint trong `LAB_GUIDE.md`, tổng 180 phút / 7 checkpoint (`checkpoint_timer.html`).

| Thành viên | Role | Phụ trách chính theo checkpoint |
|---|---|---|
| Nguyễn Duy Hưng | 👑 **R1** — Team Leader & RAG Architect | Điều phối, kiểm tra, ghép code tổng hợp cuối cùng (không tự làm 1 task riêng, xem mục 3) |
| Nguyễn Hoàng Thảo Tiên | ⚙️ **R2** — Data & Pipeline Specialist | Task 1 → Task 4 → Task 7 → Task 9 → nối `generate_with_citation()` vào `app.py` |
| Đoàn Duy Chiến | 🎨 **R3** — Frontend & Chatbot Dev | Task 2 → Task 5 → Task 8 → Task 10 → xây `app.py` |
| Trần Bảo Phúc | 📊 **R4** — Evaluation & QA Engineer | Task 3 → Task 6 → test fallback → rà soát citation → `golden_dataset.json` + RAGAS + `results.md` |

⚠️ **Thay đổi so với bản trước:** Task 9 (tích hợp pipeline) nay do **R2** làm (không phải R1); R1 không có task code riêng mà tập trung kiểm tra chất lượng/thông số ở từng checkpoint — đúng vai trò "Team Leader & RAG Architect" thuần điều phối. Nếu R1 rảnh tay ở checkpoint nào, ưu tiên nhảy vào hỗ trợ người đang là nút thắt (xem mục 4).

---

## 1. Lịch trình 7 Checkpoint (180 phút)

| Checkpoint | Thời gian | Mục tiêu phải đạt | File nộp / Kiểm tra |
|---|---|---|---|
| **CP0** 🟦 | 0:00–0:10 (10m) | Cài xong môi trường venv, có file `.env` chứa API Key | `pip install -r requirements.txt` |
| **CP1** 🟦 | 0:10–0:35 (25m) | Có ≥3 PDF trong `legal/`, ≥5 JSON trong `news/` và convert sang `.md` | `python -m src.task3_convert_markdown` |
| **CP2** 🟩 | 0:35–1:00 (25m) | Cắt đoạn văn bản, lưu ChromaDB, chạy thử Semantic & BM25 | `python -m src.task4_chunking_indexing` |
| **CP3** 🟩 | 1:00–1:20 (20m) | Viết thuật toán RRF Rerank gộp thứ hạng & tích hợp PageIndex | `python -m src.task7_reranking` |
| **CP4** 🟩 | 1:20–1:45 (25m) | **Mốc cá nhân 50đ**: Chạy Pytest đạt 35/35 PASSED | `python -m pytest tests/test_individual.py -v` |
| **CP5** 🟧 | 1:45–2:15 (30m) | **Mốc bài nhóm 50đ**: Chạy Chatbot Streamlit + Đánh giá RAGAS | `streamlit run app.py` |
| **CP6** 🟦 | 2:15–3:00 (45m) | Thuyết trình Live Demo (45 phút) & Push code GitHub | `git push origin main` |

Mỗi checkpoint đã có sẵn vài phút review/demo ngẫu nhiên cuối chặng (coach gọi bất kỳ nhóm nào demo) — **không cần thêm slot review riêng**, cứ bám khung giờ trên.

---

## 2. Phân công chi tiết theo từng checkpoint

### 🟦 CP0 — Setup Môi Trường & Khởi Tạo Project (0:00–0:10)

| Role | Việc |
|---|---|
| 👑 **R1** | Kiểm tra cả nhóm clone thành công repo, khởi tạo repository chung, chia sẻ `.env` (`OPENROUTER_API_KEY`, `PAGEINDEX_API_KEY`) |
| ⚙️ **R2** | `python -m venv .venv`, `pip install -r requirements.txt`, kiểm tra `import chromadb, sentence_transformers` |
| 🎨 **R3** | Kiểm tra cài Streamlit: `streamlit run app.py` chạy được (dù chưa có nội dung) |
| 📊 **R4** | Kiểm tra tồn tại/cài đặt `ragas`, `datasets` |

✅ **Pass Criteria:** Tất cả khởi tạo xong môi trường không lỗi import.

**Chốt chung 2 phút cuối:** cả nhóm thống nhất nguồn PDF (Task 1) + bài viết (Task 2) để R2/R3 không chọn trùng.

---

### 🟦 CP1 — Thu Thập & Chuẩn Hoá Dữ Liệu — Task 1..3 (0:10–0:35)

| Role | Việc |
|---|---|
| 👑 **R1** | Kiểm tra phân công nguồn dữ liệu, tránh trùng lặp tài liệu giữa R2/R3 |
| ⚙️ **R2** | **Task 1** — Tải ≥3 tài liệu chính sách gốc (PDF/DOCX) → `data/landing/legal/` |
| 🎨 **R3** | **Task 2** — Crawl ≥5 bài viết/thông báo hướng dẫn → `data/landing/news/` |
| 📊 **R4** | **Task 3** — `python -m src.task3_convert_markdown` chuyển đổi toàn bộ sang `data/standardized/` |

⚠️ **Lưu ý thứ tự:** Task 3 (R4) **phụ thuộc trực tiếp** vào output của Task 1 (R2) và Task 2 (R3) — R4 không có gì để convert nếu R2/R3 chưa tải/crawl xong file nào. Để tránh R4 ngồi chờ hết 25 phút:
- R2/R3 ưu tiên tải/crawl xong **1 file đầu tiên trong 5-7 phút đầu** rồi báo ngay cho R4, thay vì làm xong hết 3-5 file mới báo.
- R4 chạy thử `task3_convert_markdown.py` ngay khi có 1 file mẫu để bắt lỗi sớm (vd: thiếu `markitdown[pdf]`), sau đó chạy lại lần cuối khi R2/R3 đã có đủ file.

✅ **Pass Criteria:** ≥3 file trong `legal/`, ≥5 file trong `news/`, có `.md` tương ứng trong `standardized/`.

---

### 🟩 CP2 — Chunking, Indexing & Search Cơ Bản — Task 4..6 (0:35–1:00)

| Role | Việc |
|---|---|
| 👑 **R1** | Kiểm tra tham số chunking (`CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`) và xác nhận dùng embedding model `BAAI/bge-m3` |
| ⚙️ **R2** | **Task 4** — Chunking + gọi embedding model + tạo ChromaDB (`chroma_db/`) |
| 🎨 **R3** | **Task 5** — Hoàn thiện `semantic_search()` trong `task5_semantic_search.py` |
| 📊 **R4** | **Task 6** — Hoàn thiện `lexical_search()` trong `task6_lexical_search.py` (BM25) |

⚠️ **Lưu ý thứ tự:** Task 5 (R3) cần **collection ChromaDB đã tồn tại** (output Task 4 của R2) mới test được thật, nên R3 có thể bị chờ đầu chặng. Ngược lại, **Task 6 (R4) không phụ thuộc Task 4/5** — chỉ cần đọc trực tiếp text từ `data/standardized/` (`Path.rglob("*.md")`), nên R4 bắt tay làm được ngay từ phút đầu CP2, không cần chờ ai.
- R2 ưu tiên chạy xong bước embed + index (bỏ qua việc tối ưu tham số) trong ~10-12 phút đầu để R3 sớm có ChromaDB dùng thật.
- Trong lúc chờ, R3 viết trước phần code gọi `collection.query(...)` dựa theo comment mẫu có sẵn trong `task5_semantic_search.py`, chỉ cần cắm collection thật vào khi R2 xong.

✅ **Pass Criteria:** `chroma_db/` có data; `pytest tests/test_individual.py -k "task4 or task5 or task6"` pass.

---

### 🟩 CP3 — Reranking & Vectorless Fallback — Task 7..8 (1:00–1:20)

| Role | Việc |
|---|---|
| 👑 **R1** | Kiểm tra công thức RRF ($k=60$) đảm bảo cân bằng giữa kết quả Semantic và BM25 |
| ⚙️ **R2** | **Task 7** — Hoàn thiện `rerank_rrf()` trong `task7_reranking.py` |
| 🎨 **R3** | **Task 8** — Tích hợp PageIndex SDK trong `task8_pageindex_vectorless.py` |
| 📊 **R4** | Thử nghiệm câu hỏi **ngoài domain** qua `semantic_search()` (Task 5 của R3) để đo điểm cosine, chuẩn bị số liệu giúp R2 calibrate ngưỡng fallback ở Task 9 |

✅ **Không ai phải chờ nhau ở checkpoint này:**
- Task 7 (R2) chỉ cần input là 2 list `[{"content","score","metadata"}]` đúng format — R2 có thể tự gõ list mock 5 phần tử để code/test `rerank_rrf()` mà không cần Task 5/6 thật (dù thực tế cả hai đã xong từ CP2).
- Task 8 (R3) hoàn toàn độc lập — dùng SDK/API riêng, tự upload file PDF của chính mình lên PageIndex.

✅ **Pass Criteria:** RRF gộp thành công kết quả từ 2 ranker; PageIndex trả kết quả phù hợp.

---

### 🟩 CP4 — Pipeline Hoàn Chỉnh & Generation — Task 9..10 (1:20–1:45) — MỐC 50đ CÁ NHÂN

| Role | Việc |
|---|---|
| 👑 **R1** | Kiểm tra toàn bộ mã nguồn bài cá nhân, chạy `pytest tests/test_individual.py` xác nhận cả nhóm đạt điểm |
| ⚙️ **R2** | **Task 9** (`task9_retrieval_pipeline.py`) — Nối Semantic + BM25 + RRF + PageIndex Fallback khi cosine gốc `< 0.48` |
| 🎨 **R3** | **Task 10** (`task10_generation.py`) — Reordering (`front + back[::-1]`) + gọi LLM sinh câu trả lời có citation |
| 📊 **R4** | Rà soát định dạng citation trong câu trả lời từ LLM |

⚠️ **Đây là điểm nghẽn thật sự duy nhất trong ngày:** Task 9 (R2) cần Task 5 (R3), 6 (R4), 7 (R2 tự làm), 8 (R3) đều đã chạy được thật — nhưng tất cả đã xong từ CP2-CP3 nên R2 không phải chờ gì thêm ở CP4, chỉ việc ráp lại.
- **Bẫy quan trọng:** so ngưỡng `0.48` với **`dense_results[0]["score"]`** (điểm cosine gốc từ Task 5), **KHÔNG so với điểm RRF đã fuse** (RRF luôn ≈0.016 dù có liên quan hay không) → nếu so nhầm, fallback không bao giờ kích hoạt.
- Task 10 (R3) không cần chờ Task 9 xong mới code — có thể viết `generate_with_citation()` và test bằng list context giả trước, chỉ cắm `retrieve()` thật của R2 vào khi cả hai đều sẵn sàng.

✅ **Pass Criteria bắt buộc:** `python -m pytest tests/test_individual.py -v` → **35/35 passed**. Nếu chưa xanh, dừng mọi việc khác, cả 4 người chia nhau fix nốt test fail trong 5 phút cuối trước khi qua CP5.

---

### 🟧 CP5 — Bài Tập Nhóm: Chatbot UI & Đánh Giá RAGAS (1:45–2:15) — MỐC 30đ NHÓM

| Role | Việc |
|---|---|
| 👑 **R1** | Phân công tổng hợp đoạn code tối ưu nhất của nhóm vào `app.py`, theo dõi tiến độ báo cáo |
| ⚙️ **R2** | Kết nối `generate_with_citation()` (Task 10 của R3) vào luồng xử lý câu hỏi của `app.py` |
| 🎨 **R3** | Hoàn thiện Streamlit `app.py`: chat UI chuyên nghiệp, thanh chỉnh `top_k`, khung hiển thị source documents, câu hỏi gợi ý, không dùng icon trang trí |
| 📊 **R4** | Xây `group_project/evaluation/golden_dataset.json` (15–20 câu) → `python -m group_project.evaluation.eval_pipeline` lấy 4 chỉ số RAGAS → viết `results.md` (kèm so sánh A/B, vd hybrid+rerank vs dense-only) |

✅ **Pass Criteria:** Chatbot UI phản hồi kèm nguồn; `results.md` có đủ bảng điểm A/B testing.

**Lưu ý:** R4 nên soạn `golden_dataset.json` **từ trước** (ngay từ CP1-CP2, tận dụng lúc chờ Task 3 xong) để đến CP5 chỉ cần chạy eval thật, không mất thời gian soạn câu hỏi vào lúc gấp.

---

### 🟦 CP6 — Thuyết Trình Demo Live & Nộp Bài (2:15–3:00)

| Role | Việc |
|---|---|
| 👑 **R1** | Thuyết trình tổng quan kiến trúc RAG Pipeline & Chatbot (5-8 phút) |
| ⚙️ **R2** | Trả lời câu hỏi kỹ thuật về Hybrid Search, RRF, Fallback logic |
| 🎨 **R3** | Live demo Streamlit trên màn hình chiếu |
| 📊 **R4** | Báo cáo kết quả RAGAS, phân tích Hybrid Search vs Dense-only |
| Cả nhóm | `git push origin main` trước khi hết giờ |

---

## 3. Vai trò điều phối của R1 (không có task code riêng)

R1 không sở hữu 1 file `taskN_*.py` cụ thể nào — nhiệm vụ xuyên suốt là **kiểm tra + gỡ nghẽn**:

| Checkpoint | Việc kiểm tra cụ thể |
|---|---|
| CP0 | `.env` đủ key, mọi người import thư viện không lỗi |
| CP1 | R2/R3 không chọn trùng nguồn tài liệu |
| CP2 | `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`, embedding model `BAAI/bge-m3` đúng như thống nhất |
| CP3 | Công thức RRF đúng $k=60$, không lệch giữa 2 ranker |
| CP4 | Chạy `pytest -v` xác nhận 35/35 trước khi cho phép sang bài nhóm |
| CP5 | Review code trước khi merge vào `app.py`, đảm bảo không có 2 bản logic đá nhau |

**Thời gian rảnh của R1** (khi không có gì để kiểm tra ngay) nên dùng để: nhảy vào hỗ trợ role đang là nút thắt của checkpoint đó (xem cột "⚠️ Lưu ý thứ tự" ở mỗi checkpoint), hoặc viết `supervisor.py` (pattern nâng cao, không bắt buộc chấm điểm) nếu còn dư thời gian.

---

## 4. Bảng tổng hợp các điểm có thể bị chờ & cách né

| Checkpoint | Ai có thể bị chờ | Chờ ai / cái gì | Cách né |
|---|---|---|---|
| CP1 | R4 (Task 3) | File từ R2 (Task 1) / R3 (Task 2) | R2/R3 báo ngay khi có 1 file đầu tiên, không đợi xong hết |
| CP2 | R3 (Task 5) | ChromaDB từ R2 (Task 4) | R2 ưu tiên embed+index trước, tối ưu tham số sau; R3 viết code trước bằng comment mẫu, cắm data thật sau |
| CP3 | Không ai | — | Task 7/8 độc lập hoàn toàn với nhau |
| CP4 | Không ai (nếu CP2-CP3 đúng tiến độ) | — | Task 9 chỉ ráp lại, Task 10 code song song bằng mock context |
| CP5 | R4 (chạy eval thật) | Task 9 (R2) + Task 10 (R3) xong | R4 soạn `golden_dataset.json` từ sớm (CP1-CP2) để không mất thời gian ở CP5 |

---

## 5. Rủi ro & xử lý sự cố

| Tình huống | Xử lý |
|---|---|
| Trang nguồn chặn bot (403) khi R2/R3 tải/crawl | Dùng ngay bộ data mẫu có sẵn trong repo (xem `LAB_GUIDE.md` mục Troubleshooting) |
| `MissingDependencyException` khi R4 convert PDF (Task 3) | `pip install "markitdown[pdf]"` |
| `BrowserType.launch: Executable doesn't exist` khi R3 crawl (Task 2) | `playwright install chromium` |
| `UnicodeEncodeError` khi in tiếng Việt trên Windows | `$env:PYTHONIOENCODING="utf-8"` hoặc `python -X utf8` |
| Đổi dữ liệu nhưng quên xoá `chroma_db/` cũ | `Remove-Item -Recurse -Force chroma_db` rồi chạy lại Task 4 |
| Fallback ở Task 9 không bao giờ kích hoạt | Kiểm tra R2 đang so với `dense_results[0]["score"]` (cosine gốc), không phải điểm RRF |
| `429 Too Many Requests` khi R4 chạy RAGAS | Giảm tạm số câu hỏi trong `golden_dataset.json` xuống 5 câu khi chạy thử |
| Cuối CP4 pytest chưa xanh hết | Dừng mọi việc khác, R1 chia mỗi người 2-3 test fail gần task của mình để fix song song |

---

## 6. Contract giữa các module (để code song song không cần chờ)

Tất cả hàm retrieval trả về cùng 1 format — ai cần gọi hàm người khác thì cứ mock đúng format này để code trước:

```python
list[dict]  # mỗi dict:
{
    "content": str,     # nội dung chunk
    "score": float,     # điểm liên quan (semantic: [0,1], BM25: [0,∞))
    "metadata": dict,   # {"source": ..., "doc_type": ..., "chunk_index": ..., "customer_role": "buyer"/"seller"/"both"}
}
```

Riêng `pageindex_search()` và output cuối của `retrieve()` (Task 9) có thêm field `"source": "hybrid" | "pageindex"`.

**Mock nhanh dùng ngay khi module upstream chưa xong:**
```python
MOCK_RESULTS = [
    {"content": "Shopee hỗ trợ hoàn tiền trong 15 ngày...", "score": 0.82, "metadata": {"source": "returns-refund-policy-shopee.md", "customer_role": "buyer"}},
    {"content": "Người bán chịu phí sàn 5% trên mỗi đơn...", "score": 0.65, "metadata": {"source": "seller-listing-regulations-shopee.md", "customer_role": "seller"}},
]
```
