"""Policy — Escalation: cấp cứu -> hướng dẫn gọi cấp cứu (không qua LLM/RAG).

Tái dùng thông điệp cấp cứu của input_guard (nguồn sự thật duy nhất, tránh lệch nội dung).
"""
from __future__ import annotations

from src.serving.guards.input_guard import emergency_check


def escalation_message(query: str) -> str | None:
    """Trả thông điệp cấp cứu nếu query là emergency; None nếu không."""
    return emergency_check(query)
