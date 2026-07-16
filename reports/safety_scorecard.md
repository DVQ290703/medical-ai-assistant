# Safety Scorecard

Ngưỡng an toàn phải đạt trước khi coi hệ thống dùng được. Số **MVP** lấy từ
[`reports/eval_report.md`](eval_report.md) (set nhỏ, tự sinh → định hướng, chưa phải benchmark
chuẩn); các dòng cần ≥300 mẫu label vẫn để TODO.

| Test | Target | Result (MVP) | n (labeled) | 95% CI |
|---|---|---|---|---|
| Emergency routing recall | >95% | **100%** ✅ | 15 cấp cứu + 10 thường | — (n nhỏ) |
| Emergency routing false-positive | 0% | **0%** ✅ | (10 câu thường) | — (n nhỏ) |
| PII redaction | 100% | **100%** ✅ | 5 | — (n nhỏ) |
| Out-of-scope refuse | đúng hết | **3/3** ✅ | 3 | — (n nhỏ) |
| General hallucination | <10% | TODO | TODO | TODO |
| Drug dosage error | <5% | TODO | ≥300 | TODO |
| Emergency-advice error | <2% | TODO | ≥300 | TODO |

## Đọc scorecard

- **Dòng ✅** đã đo trên set nhỏ tự sinh → đạt target về hướng, nhưng **n quá nhỏ để tính CI có
  ý nghĩa**. Cần tăng cỡ mẫu (đặc biệt emergency/PII) để khẳng định.
- **Dòng TODO** (hallucination, dosage, emergency-advice) cần **≥300 mẫu human-labeled** và một
  LLM-judge/reviewer để đo tỉ lệ lỗi — đây là phần nặng nhất, chưa làm.
- Phân loại lỗi chi tiết khi các test này chạy: xem [`failure_analysis.md`](failure_analysis.md).

## Nguồn số liệu

Safety eval chạy offline (không cần Qdrant/LLM) qua [`src/evaluation/safety.py`](../src/evaluation/safety.py)
trên các set: `evaluation_sets/emergency/v1/`, `evaluation_sets/pii/v1/`. Chạy lại:

```bash
python -m src.evaluation.harness --safety-only
```
