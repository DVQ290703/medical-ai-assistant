"""Monitoring — Hàng đợi feedback người dùng (👍/👎), CHỜ human review trước khi vào train.

Chống feedback poisoning: feedback KHÔNG tự động đưa vào dữ liệu train. Lưu vào JSONL,
người review duyệt sau. Song song gửi score lên Langfuse (observability) để tổng hợp.

Ghi append (an toàn đa request nhẹ). Mỗi dòng: {ts, trace_id, query, rating, comment}.
"""
from __future__ import annotations

import json
from pathlib import Path

QUEUE = "data/feedback/queue.jsonl"


def add_feedback(trace_id: str, query: str, rating: str, comment: str = "",
                 ts: str = "") -> None:
    """rating: 'up' | 'down'. ts truyền vào (caller stamp — tránh Date trong sandbox)."""
    p = Path(QUEUE)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": ts, "trace_id": trace_id, "query": query,
           "rating": rating, "comment": comment}
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_queue() -> list[dict]:
    p = Path(QUEUE)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def stats() -> dict:
    """Tổng hợp nhanh (cho báo cáo nội bộ, KHÔNG hiển thị user)."""
    q = load_queue()
    up = sum(1 for r in q if r.get("rating") == "up")
    down = sum(1 for r in q if r.get("rating") == "down")
    return {"total": len(q), "up": up, "down": down,
            "satisfaction": up / (up + down) if (up + down) else None}
