"""Serving — Citation: map [số] trong câu trả lời -> nguồn (title/url) đã retrieve.

build_sources_list: từ hits + các [số] LLM thực sự trích -> danh sách nguồn để hiển thị.
Nếu citation_required mà answer không có [số] nào -> đánh dấu (caller cảnh báo/append).
"""
from __future__ import annotations

import re

_CITE_RE = re.compile(r"\[(\d+)\]")


def cited_indices(answer: str) -> list[int]:
    """Các số [i] xuất hiện trong answer (unique, giữ thứ tự)."""
    seen, out = set(), []
    for m in _CITE_RE.finditer(answer):
        i = int(m.group(1))
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def build_sources(answer: str, hits: list) -> list[dict]:
    """Trả danh sách nguồn tương ứng các [số] được trích trong answer.

    hits index 1-based khớp [i] ở builder.format_context. [số] ngoài phạm vi -> bỏ qua.
    """
    idxs = cited_indices(answer)
    sources = []
    for i in idxs:
        if 1 <= i <= len(hits):
            h = hits[i - 1]
            sources.append({
                "n": i,
                "title": getattr(h, "title", ""),
                "url": getattr(h, "url", ""),
                "source": getattr(h, "source", ""),
            })
    return sources


def format_sources(sources: list[dict]) -> str:
    """Render danh sách nguồn để in kèm câu trả lời."""
    if not sources:
        return ""
    lines = ["\nNguồn tham khảo:"]
    for s in sources:
        line = f"  [{s['n']}] {s['title']}"
        if s.get("url"):
            line += f" — {s['url']}"
        elif s.get("source"):
            line += f" — {s['source']}"
        lines.append(line)
    return "\n".join(lines)
