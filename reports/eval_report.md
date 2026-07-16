# Báo cáo Evaluation — Medical RAG (VN)

_Sinh lúc: 2026-07-16 21:55_

> Số liệu MVP. Set nhỏ, tự sinh -> ĐỊNH HƯỚNG, không phải benchmark chuẩn.

## An toàn (Safety)

- **Emergency routing** (n=15 cấp cứu + 10 thường): recall **100%**, false-positive **0%**.
- **PII redaction** (n=5): che **100%**, che nhầm nội dung y khoa: **0**.
- **Out-of-scope refuse**: đúng 3/3, nhầm câu y tế: 0.

## Retrieval
- _Bỏ qua (--safety-only)._

## Chưa đo (roadmap)
- Accuracy MCQ (VM14K/MedQA) — cần golden set human-labeled.
- Quality RAGAS (faithfulness/answer-relevance) — cần LLM-judge.
- Drift monitoring theo thời gian.
