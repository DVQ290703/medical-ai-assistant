"""Prompting — Load system prompt (versioned) từ prompts/."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=8)
def load_system_prompt(path: str = "prompts/system_prompt_v1.txt") -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()
