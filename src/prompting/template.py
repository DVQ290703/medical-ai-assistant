"""Prompting — Load system prompt + few-shot examples (versioned) từ prompts/."""
from __future__ import annotations

import json
import os
from functools import lru_cache


@lru_cache(maxsize=8)
def load_system_prompt(path: str = "prompts/system_prompt_v1.txt") -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


@lru_cache(maxsize=4)
def load_fewshot(path: str = "prompts/fewshot_v1.jsonl") -> tuple:
    """Nạp ví dụ few-shot (asset versioned). Mỗi dòng JSONL: {user, [context], assistant}.

    Trả tuple (immutable -> lru_cache được) các dict. Thiếu file -> rỗng (few-shot tuỳ chọn).
    """
    if not os.path.exists(path):
        return ()
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return tuple(out)
