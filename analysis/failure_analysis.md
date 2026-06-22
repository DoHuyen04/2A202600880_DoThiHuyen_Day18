# Failure Analysis — Lab 18

**Sinh viên:** Đỗ Thị Huyền
**Ngày chạy:** 2026-06-22
**Pipeline:** M1 Hierarchical Chunking → M5 Enrichment → M2 Hybrid (BM25 + Dense + RRF) → M3 Cross-Encoder Rerank → LLM (gpt-4o-mini, temperature=0) → M4 RAGAS

> **Lưu ý môi trường:** Máy chỉ có ~7.8GB RAM (≈1.1GB trống), không giữ đồng thời được model embedding lớn + cross-encoder reranker → segfault. Đã hạ embedding xuống `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) và reranker xuống `bge-reranker-base`, đồng thời eval chạy **2-pass** (giải phóng encoder trước khi load reranker) + cache enrichment. Cả hai model vẫn đa ngôn ngữ, hỗ trợ tiếng Việt. Điểm dưới đây phản ánh cấu hình nhẹ này.

---

## RAGAS Scores (20 câu hỏi)

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.7889 | 0.7729 | −0.0160 |
| Answer Relevancy | 0.7296 | 0.7709 | **+0.0413** |
| Context Precision | 0.9417 | 0.9042 | −0.0375 |
| Context Recall | 0.8500 | 0.8500 | 0.0000 |

**Nhận xét tổng quan:**
- **Answer Relevancy ↑ và vượt ngưỡng 0.75:** baseline 0.7296 (< 0.75) → production 0.7709 (≥ 0.75). Enrichment (M5) + reranking (M3) giúp câu trả lời đúng trọng tâm hơn → **production đạt cả 4 metrics ≥ 0.75**, baseline thì không.
- **Faithfulness / Context Precision ↓ nhẹ:** production dùng `chunk_hierarchical` child-256 (107 chunks) thay vì `chunk_basic` (57 chunks). Chunk nhỏ + nhiều version tài liệu (v1/v2, v2023/v2024) làm lọt nhiều chunk gần-giống vào top-k → nhiễu, kéo precision/faithfulness xuống chút ít.
- **Biến động RAGAS:** qua nhiều lần chạy, faithfulness dao động 0.757–0.839 do RAGAS dùng LLM làm "judge" + model embedding/rerank bản nhẹ. Số trên là lần chạy ổn định (temperature=0).

---

## Bottom-5 Failures

### #1 — avg 0.375 (tệ nhất)
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành v2.0: **120 ngày** (v1.0 cũ là 90 ngày, đã thay thế).
- **Got:** "Không tìm thấy."
- **Worst metric:** faithfulness = 0.0 (answer_relevancy 0.0, context_precision 0.5, context_recall 1.0)
- **Error Tree:** Output sai (từ chối) → Context **có đủ** (recall 1.0) nhưng **lẫn cả v1 (90 ngày) lẫn v2 (120 ngày)** → precision 0.5 → model bối rối giữa 2 con số mâu thuẫn → chọn từ chối.
- **Root cause:** Nhiễu version tài liệu (mat_khau_v1 vs v2) khiến model không quyết được bản hiện hành.
- **Suggested fix:** Thêm metadata `version`/`effective_date`, filter chỉ giữ tài liệu hiện hành trước khi đưa vào LLM.

### #2 — avg 0.5625
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn > 50 triệu cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** "Cần phê duyệt theo thẩm quyền tương ứng từ phòng Mua sắm." (mơ hồ, không nêu CEO)
- **Worst metric:** answer_relevancy = 0.0 (faithfulness 0.6667, precision 0.5833, recall 1.0)
- **Error Tree:** Context có (recall 1.0) → nhưng câu trả lời **chung chung, né trả lời cụ thể** → answer_relevancy = 0.
- **Root cause:** Top-k lẫn nhiều ngưỡng phê duyệt khác nhau (precision 0.58) → model trả lời an toàn, mơ hồ.
- **Suggested fix:** `chunk_structure_aware` giữ nguyên bảng ngưỡng trong 1 chunk; tăng trọng số BM25 cho truy vấn có con số ("55 triệu").

### #3 — avg 0.6509
- **Question:** Nếu cần mua laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** 30 triệu (5–50tr) → Giám đốc phòng ban (Director); cần xác nhận cấu hình từ CNTT; đính kèm ≥3 báo giá (vì >10tr).
- **Got:** Nêu đúng phần CNTT + "trưởng phòng" nhưng **thiếu** yêu cầu 3 báo giá.
- **Worst metric:** context_recall = 0.3333 (precision 1.0, faithfulness 0.5)
- **Error Tree:** Output thiếu → Context **không gom đủ** mảnh "≥3 báo giá khi >10tr" (recall 0.33) → câu trả lời sót ý.
- **Root cause:** Quy định mua sắm bị tách thành nhiều chunk, top-k không lấy hết.
- **Suggested fix:** Tăng top_k dense + hierarchical parent retrieval để gồm đủ ngữ cảnh quy định.

### #4 — avg 0.6714
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Hạn 15 ngày; quá hạn 5 ngày × 2%/tháng trên 15 triệu ≈ 50.000 VNĐ (pro-rata).
- **Got:** Tính nhầm "20 ngày = 0,67 tháng → phạt 200.000 VNĐ" (sai mốc hạn 15 ngày + sai số ngày quá hạn).
- **Worst metric:** faithfulness = 0.125 (context_precision 1.0, recall 0.6667)
- **Error Tree:** Context **đủ** (precision 1.0) → nhưng LLM **tự bịa phép tính**, bỏ qua mốc "hạn 15 ngày" → faithfulness sụp.
- **Root cause:** Hallucination ở bước suy luận số học.
- **Suggested fix:** Prompt cấm tự tính ngoài context + tách bước "trích mốc hạn" trước khi tính; hoặc dùng tool/calculator có kiểm soát.

### #5 — avg 0.6959
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16–30 ngày → Giám đốc điều hành (CEO); lưu ý >14 ngày phải tự đóng bảo hiểm.
- **Got:** "Cần phê duyệt của Giám đốc điều hành (CEO)." (đúng phần lõi, thiếu lưu ý bảo hiểm)
- **Worst metric:** faithfulness = 0.5 (context_recall 0.5, answer_relevancy 0.78)
- **Error Tree:** Output đúng phần chính → nhưng context chỉ lấy được 1 mảnh (ngưỡng duyệt), **thiếu** mảnh "tự đóng bảo hiểm" → recall 0.5, faithfulness giảm.
- **Root cause:** Chunk lưu ý bảo hiểm tách rời, không lọt cùng top-k.
- **Suggested fix:** Hierarchical parent retrieval để trả về cả đoạn quy định liên quan.

---

## Pattern chung & hướng cải thiện ưu tiên

| Pattern lỗi | Câu liên quan | Hướng fix ưu tiên |
|-------------|---------------|-------------------|
| **Nhiễu version tài liệu** (precision thấp, model từ chối/mơ hồ) | #1, #2 | Metadata `version`/`effective_date` + filter tài liệu hiện hành |
| **Thiếu chunk** (context_recall thấp) | #3, #5 | Tăng top_k, hierarchical parent retrieval, `chunk_structure_aware` giữ bảng nguyên khối |
| **Hallucination khi tính toán** (faithfulness thấp) | #4 | Prompt cấm tự tính ngoài context; tách bước trích mốc trước khi tính |

---

## Case Study (cho presentation)

**Question chọn phân tích:** "Bao lâu phải đổi mật khẩu một lần?" (avg 0.375 — tệ nhất)

**Error Tree walkthrough:**
1. Output đúng? → Không — model trả "Không tìm thấy" dù đáp án có trong tài liệu.
2. Context đúng? → Có thông tin (recall 1.0) nhưng **lẫn 2 version mâu thuẫn**: `mat_khau_v1.md` (90 ngày) + `mat_khau_v2.md` (120 ngày) → precision 0.5.
3. Query rewrite OK? → Có, query rõ; vấn đề ở **retrieval lẫn version** khiến model bối rối.
4. Fix ở bước: **M2/M5** — gắn metadata `version` khi enrich, filter "chỉ tài liệu hiện hành" trước khi đưa vào LLM.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thêm trường `version`/`effective_date` vào metadata (M5) và filter ở M2 → xử lý cả #1, #2.
- Hierarchical parent retrieval cho #3, #5 (gom đủ ngữ cảnh quy định).
- Prompt chống tự-tính-toán cho #4.
