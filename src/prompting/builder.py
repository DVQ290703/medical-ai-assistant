"""Prompting — Build context (đánh số nguồn) + ghép user prompt từ hits retrieve.

format_context: mỗi hit -> khối "[i] <title> (nguồn)\\n<text>". Số [i] để LLM trích dẫn.
build_user_prompt: context + câu hỏi + chỉ dẫn trích [số].
"""
from __future__ import annotations


def format_context(hits: list) -> str:
    """hits: list Hit (có .title, .text, .source, .url). Trả context đánh số [1][2]..."""
    blocks = []
    for i, h in enumerate(hits, 1):
        title = getattr(h, "title", "") or "(không tiêu đề)"
        src = getattr(h, "source", "") or ""
        text = getattr(h, "text", "")
        tag = f"[{i}] {title}"
        if src:
            tag += f" — nguồn: {src}"
        blocks.append(f"{tag}\n{text}")
    return "\n\n".join(blocks)


def build_user_prompt(query: str, hits: list) -> str:
    """Ghép THÔNG TIN THAM KHẢO (đánh số) + câu hỏi + chỉ dẫn trích dẫn."""
    context = format_context(hits)
    return (
        "THÔNG TIN THAM KHẢO (chỉ dùng những gì có ở đây, trích [số] khi dùng):\n"
        f"{context}\n\n"
        f"CÂU HỎI: {query}\n\n"
        "Trả lời bằng tiếng Việt, bám sát thông tin tham khảo, trích [số] cho mỗi ý dùng "
        "nguồn. Nếu không đủ thông tin, nói rõ và khuyên đi khám."
    )
