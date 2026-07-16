# Medical AI Assistant — Vietnamese Medical Q&A (QLoRA + RAG)

Trợ lý hỏi–đáp y học tiếng Việt, thiết kế theo nguyên tắc **an toàn trước, dẫn nguồn bắt buộc**.
Kiến trúc tách bạch: model được fine-tune (Llama 3.1 8B, QLoRA/Unsloth) để học *cách lập luận,
văn phong và trích dẫn* — còn **kiến thức thực tế đến từ RAG trên nguồn y khoa Việt Nam**, không
ghi nhớ trong trọng số model. Nhờ vậy câu trả lời luôn truy vết được về tài liệu gốc và cập nhật
được mà không phải train lại.

> ⚠️ **Chỉ phục vụ nghiên cứu / học tập.** Không dùng để chẩn đoán, kê đơn hay ra quyết định y tế
> thật. Chi tiết giới hạn và rủi ro: [`governance/model_card.md`](governance/model_card.md).

## Điểm nổi bật

- **RAG hybrid, ưu tiên độ tin cậy nguồn** — BGE-M3 sinh vector dense + sparse, Qdrant fuse bằng
  RRF, cross-encoder rerank, rồi cộng điểm ưu tiên cho nguồn Bộ Y tế khi điểm sát nhau. Nếu không
  tài liệu nào đủ liên quan (dưới ngưỡng) thì trả về rỗng — thà nói "không tìm thấy" còn hơn bịa.
- **Safety plane nhiều lớp** — phát hiện cấp cứu và định tuyến, che PII, từ chối câu ngoài phạm vi,
  guard đầu vào/đầu ra, chèn disclaimer theo chính sách. Kết quả đo (MVP): định tuyến cấp cứu
  **recall 100% / false-positive 0%**, che PII **100%** không che nhầm nội dung y khoa.
- **Xử lý tiếng Việt low-resource** — không có sẵn dataset reasoning y khoa tiếng Việt, nên dữ liệu
  đi theo hướng lai: dịch medical-o1 làm seed + tăng cường dữ liệu bản địa. Base model và embedding
  đều phải benchmark riêng cho tiếng Việt trước khi chốt.
- **Quyết định kỹ thuật được ghi lại** — mỗi lựa chọn lớn (RAG vs fine-tune, chunking, embedding,
  serving, base model) có một ADR kèm lý do và đánh đổi trong [`governance/adr/`](governance/adr/).

## Kiến trúc — 3 tầng

| Tầng | Vai trò | Code |
|------|---------|------|
| **Knowledge** (offline) | Nạp nguồn → chunk → embed → vector DB (Qdrant, hybrid + RRF). Encode/rerank query có thể offload sang model server remote (Colab GPU) qua client có fallback + retry. | [`src/knowledge/`](src/knowledge/) |
| **Serving** (online) | input guard → retrieval → build prompt → generation → output guard → policy → citation | [`src/serving/`](src/serving/), [`src/prompting/`](src/prompting/), [`src/generation/`](src/generation/) |
| **Observability** | tracing (Langfuse) → hàng đợi human review → feedback | [`src/monitoring/`](src/monitoring/) |

Chi tiết đầy đủ: [`docs/architecture.md`](docs/architecture.md).

## Trạng thái triển khai

Dự án phân biệt rõ phần đã chạy thật và phần mới ở mức thiết kế — để người đọc biết đâu là code
kiểm chứng được, đâu là interface chờ hạ tầng cloud:

- **Đã chạy thật** (kể cả trên Kaggle/Colab): data pipeline, knowledge/RAG, prompting, training,
  evaluation, RAG demo offline. Observability đã hoạt động: trace Langfuse (tự tắt thành no-op nếu
  thiếu key) và hàng đợi feedback 👍/👎 chờ người duyệt trước khi vào train.
- **Thiết kế + stub** (cần cloud mới deploy thật): phần cloud của serving (Redis, autoscale) và
  phần live của monitoring — `alerts.py`, `retrieval_drift.py` hiện là placeholder.

## Chạy thử

```bash
pip install -e .
pip install -r requirements.txt        # runtime deps (qdrant-client, BGE-M3, ...)
# chỉ cần chạy phần serving: pip install -r requirements-serve.txt

make data     # Phase 1 — ingest → validate → clean → PII scrub → split
make train    # Phase 2 — QLoRA fine-tune (Unsloth)
make eval     # Phase 3 — benchmark + quality + retrieval + safety
make serve    # Phase 4 — FastAPI + RAG (demo offline)
```

Kết quả evaluation mới nhất: [`reports/eval_report.md`](reports/eval_report.md).

## Ghi chú kỹ thuật

- **Ba khái niệm "prompt" tách biệt:** [`prompts/`](prompts/) là template text có version (asset),
  [`configs/prompt.yaml`](configs/prompt.yaml) là tham số (temperature, max_tokens, citation), còn
  [`src/prompting/`](src/prompting/) là code dựng prompt.
- **Dữ liệu nặng không nằm trong git:** `data/` và `artifacts/` được quản lý bằng DVC.
