# Runbook
## Incident: hallucination spike
1. Alert (monitoring/alerts.py) -> kiểm tra retrieval drift.
2. Rollback model qua registry nếu cần.
## Incident: latency > SLA
1. Kiểm tra vLLM/cache; scale hoặc bật rate limit chặt hơn.
## Incident: safety violation report
1. Thêm case vào jailbreak/emergency golden set -> chạy eval gate.
