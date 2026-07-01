# Golden Evaluation Sets (TIẾNG VIỆT, versioned — deliverable hạng nhất)

Quy tắc: **không overwrite**. Bản mới = folder mới (`v1/ -> v2/`).

| Set | Nội dung | Ghi chú |
|---|---|---|
| vm14k/        | benchmark y khoa VN (chính) | VM14K — benchmark y khoa tiếng Việt đầu tiên |
| medqa/        | MedQA dịch (đối chiếu) | KÈM caveat dịch thuật + shuffle đáp án |
| drug_safety/  | liều thuốc (VN) | target hallucination < 5%, ≥300 case |
| emergency/    | tình huống cấp cứu (VN) | routing recall > 95%, ≥300 case |
| retrieval/    | query -> gold_docs (VN) | BẮT BUỘC để tính Recall@k, MRR |
| pii/          | rò rỉ PII | red-team |
| jailbreak/    | prompt injection / jailbreak | red-team, ≥200 case |

### Lưu ý eval MCQ tiếng Việt
- MCQ dễ thổi phồng năng lực -> cân nhắc thêm free-response.
- Có positional bias (model thiên chọn B) -> **shuffle đáp án ngẫu nhiên** khi eval.

### Format `retrieval/` (JSONL) — tài liệu là tiếng Việt
```json
{"query": "Liều paracetamol tối đa/ngày cho người lớn?", "gold_doc_ids": ["duocthu_042"]}
```
