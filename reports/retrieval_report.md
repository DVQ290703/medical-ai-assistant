# Retrieval Report (TIẾNG VIỆT)

Benchmark chất lượng retrieval trên golden set **tiếng Việt** (tài liệu + query VN) để
**justify lựa chọn embedding ở [ADR-0003](../governance/adr/0003-embedding-bge-vs-e5.md)**.
Đo bằng [`src/evaluation/retrieval.py`](../src/evaluation/retrieval.py) trên
`evaluation_sets/retrieval/v1/goldset.jsonl`.

> ⚠️ **Trạng thái: CHƯA có số liệu.** Chạy được cần: (1) golden set tiếng Việt
> (`make_retrieval_goldset.py` — hiện mới có `example.jsonl`), (2) Qdrant đã index corpus,
> (3) model server remote để encode/rerank. Eval hiện chạy `--safety-only` nên retrieval bị bỏ qua.

## So sánh embedding

| Embedding | Recall@5 | MRR | Context Precision | Ghi chú |
|---|---|---|---|---|
| **BGE-M3** (đang dùng) | TODO | TODO | TODO | Đa ngôn ngữ, hybrid dense+sparse, tới 8192 token |
| multilingual-e5-large | TODO | TODO | TODO | Ứng viên so sánh (chỉ dense) |
| ~~bge-large-en~~ | — | — | — | **Loại**: embedding tiếng Anh không ăn tiếng Việt |

## Phương pháp

- **Recall@k**: gold doc có nằm trong top-k kết quả không (1/0), trung bình toàn set.
- **MRR**: 1/(thứ hạng gold đầu tiên) — thưởng cho việc xếp gold lên cao.
- **Context Precision**: tỉ lệ tài liệu trả về thực sự liên quan (cần label thủ công hoặc LLM-judge).
- Golden set hiện dựng theo **self-retrieval sanity-check** (query sinh từ chính chunk → chunk đó
  là gold). Đây là kiểm tra "pipeline có hoạt động không", **chưa phải benchmark human-labeled** —
  cần bổ sung query người viết + gold người gán để đánh giá thật.

## Ý nghĩa với ADR-0003

ADR-0003 (embedding cho tiếng Việt) đang ở trạng thái **Proposed — chờ benchmark**. Bảng số ở
trên chính là bằng chứng để chốt: BGE-M3 vs multilingual-e5-large trên corpus VN. Khi có số,
cập nhật cả file này lẫn status của ADR-0003.
