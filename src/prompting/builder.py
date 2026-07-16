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


def build_user_prompt(query: str, hits: list, force_answer: bool = False) -> str:
    """Ghép THÔNG TIN THAM KHẢO (đánh số) + câu hỏi + chỉ dẫn trích dẫn.

    Lượt user cuối cùng trong hội thoại — nơi đính context RAG cho câu hỏi hiện tại.
    force_answer=True: đã hỏi lại đủ số lần -> CẤM hỏi tiếp, buộc trả lời với gì đã biết.
    """
    context = format_context(hits)
    clarify_rule = (
        "ĐÃ hỏi làm rõ đủ rồi — LẦN NÀY TUYỆT ĐỐI KHÔNG hỏi lại nữa (không dùng [HỎI LẠI]). "
        "Hãy trả lời ngay dựa trên những gì đã biết trong hội thoại; nếu vẫn thiếu chi tiết, "
        "nêu thông tin tham khảo chung + khuyên đi khám."
        if force_answer else
        "Nếu câu hỏi còn MƠ HỒ (thiếu triệu chứng/thời gian/mức độ) để trả lời an toàn, hãy "
        "HỎI LẠI 1-2 câu làm rõ thay vì trả lời vội — mở đầu bằng '[HỎI LẠI]'."
    )
    return (
        "THÔNG TIN THAM KHẢO (chỉ dùng những gì có ở đây, trích [số] khi dùng):\n"
        f"{context}\n\n"
        f"CÂU HỎI: {query}\n\n"
        "Trả lời bằng tiếng Việt, bám sát thông tin tham khảo, trích [số] cho mỗi ý dùng "
        "nguồn. Trả lời ĐÚNG và ĐỦ điều được hỏi: nếu hỏi CÁCH XỬ TRÍ/ĐIỀU TRỊ và tài liệu "
        "có, phải nêu hướng xử trí cụ thể (chăm sóc tại nhà, nhóm thuốc, dấu hiệu cần đi khám "
        f"ngay) — đừng chỉ nói 'đi khám bác sĩ'. {clarify_rule} "
        "Nếu không đủ thông tin tham khảo, nói rõ và khuyên đi khám."
    )


# số lượt hội thoại tối đa giữ lại (tránh prompt phình + rò rỉ ngữ cảnh cũ)
MAX_HISTORY_TURNS = 6


def _fewshot_turns(examples) -> list[dict]:
    """Biến ví dụ few-shot (dict {user, [context], assistant}) thành cặp message user/assistant.

    Ví dụ có context -> ghép giống lượt thật (context đánh số + câu hỏi) để model học đúng
    KHUÔN cả khi có tài liệu. Đặt TRƯỚC hội thoại thật (dạy hành vi, không phải nội dung).
    """
    out: list[dict] = []
    for ex in examples or ():
        u = ex.get("user", "")
        ctx = ex.get("context", "")
        user_msg = (
            f"THÔNG TIN THAM KHẢO:\n{ctx}\n\nCÂU HỎI: {u}" if ctx else u
        )
        out.append({"role": "user", "content": user_msg})
        out.append({"role": "assistant", "content": ex.get("assistant", "")})
    return out


def build_turns(query: str, hits: list, history: list | None = None,
                force_answer: bool = False, fewshot=None) -> list[dict]:
    """Ghép lịch sử hội thoại thành list messages cho engine.generate_messages.

    fewshot: ví dụ mẫu (asset từ prompts/fewshot_*.jsonl) -> chèn ĐẦU danh sách, trước
    history thật. Dạy model KHUÔN trả lời (hỏi lại đúng cách, không né điều trị, giữ
    ranh giới) — hiệu quả hơn liệt kê quy tắc.
    history: list {role: 'user'|'assistant', content: str} các lượt TRƯỚC (không gồm
    query hiện tại). Lượt cuối (query hiện tại) mới được đính context RAG — các lượt cũ
    giữ nguyên text để model hiểu mạch, nhưng KHÔNG kèm lại context (tránh phình prompt).
    force_answer: đã hỏi lại quá nhiều -> buộc trả lời (không hỏi tiếp).
    """
    turns: list[dict] = _fewshot_turns(fewshot)
    for h in (history or [])[-MAX_HISTORY_TURNS:]:
        role = h.get("role")
        content = (h.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            turns.append({"role": role, "content": content})
    turns.append({"role": "user",
                  "content": build_user_prompt(query, hits, force_answer=force_answer)})
    return turns
