"""Serving — Rate limiter (per-minute/day) chống abuse/DoS/flood.

In-memory theo IP (đủ demo; production dùng redis như configs/serving.yaml ghi). Sliding
window: giữ timestamp trong 60s / 86400s, vượt ngưỡng -> False.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, per_minute: int = 30, per_day: int = 500):
        self.per_minute = per_minute
        self.per_day = per_day
        self._min: dict = defaultdict(deque)
        self._day: dict = defaultdict(deque)

    def check(self, key: str) -> tuple[bool, str]:
        """Trả (ok, lý do). ok=False nếu vượt ngưỡng. key thường là IP."""
        now = time.time()
        dq_m, dq_d = self._min[key], self._day[key]
        while dq_m and now - dq_m[0] > 60:
            dq_m.popleft()
        while dq_d and now - dq_d[0] > 86400:
            dq_d.popleft()
        if len(dq_m) >= self.per_minute:
            return False, f"Quá {self.per_minute} yêu cầu/phút. Thử lại sau."
        if len(dq_d) >= self.per_day:
            return False, f"Quá {self.per_day} yêu cầu/ngày."
        dq_m.append(now)
        dq_d.append(now)
        return True, ""
