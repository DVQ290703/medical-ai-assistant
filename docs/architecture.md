# Architecture

Hai góc nhìn: **runtime** (request đi đâu) và **build/MLOps** (model được tạo & promote).

## Runtime — 3 planes
```
Knowledge plane (offline):  Nguồn y khoa VN -> Chunk&Embed (BGE-M3) -> Vector DB
                                                         |
                                                    (context)
                                                         v
Serving plane (online):  User -> Input Guard -> Retrieval -> Generation -> Output -> Response
                                (guard sớm)     (RAG trước    (prompt+LLM)  (guard+
                                                 model)                      policy+cite)
Observability plane:  Response -> Monitoring -> Human Review Queue -> Feedback --(ẩn PII)--> Data
```

## Nguyên tắc thiết kế (xem governance/adr)
- Guard ở HAI đầu (input sớm để tiết kiệm compute, output muộn để bắt hallucination).
- Policy Engine TÁCH khỏi Output Guard (Guard: an toàn? / Policy: nên làm gì?).
- Vector DB là ranh giới thay đổi độc lập: knowledge đổi ≠ model đổi.

## Map code
- Knowledge plane -> src/knowledge/
- Serving plane   -> src/serving/ (+ guards/, policy/), src/prompting/, src/generation/
- Observability   -> src/monitoring/
- Build-time      -> src/data/, src/training/, src/evaluation/


## Ghi chú tiếng Việt (y hiện đại)
- **Base model**: default Llama 3.1 8B, benchmark Qwen2.5-7B/Vistral trên VM14K (ADR-0005).
- **Embedding**: PHẢI đa ngữ (BGE-M3), bge-large-en không ăn tiếng Việt (ADR-0003).
- **Facts/RAG**: nguồn y HIỆN ĐẠI tiếng Việt — Bộ Y tế, Dược thư QG, phác đồ bệnh viện,
  sách giáo khoa Y VN, MSD Manual (VI). RAG càng thiết yếu vì kiến thức y VN khan hiếm trong model.
- **Data SFT**: hybrid = dịch medical-o1 (seed reasoning) + augment Vietnamese-native.
- **Eval**: VM14K là benchmark chính; MedQA dịch chỉ để đối chiếu (kèm caveat + shuffle đáp án).
