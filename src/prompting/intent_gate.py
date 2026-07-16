"""Prompting — Intent gate: chặn phỏng đoán bệnh khi triệu chứng chưa đủ chi tiết.

VÌ SAO CÓ FILE NÀY: prompt + few-shot KHÔNG đủ để ngăn model nêu tên bệnh ("có thể viêm
ruột thừa") khi người dùng mới mô tả triệu chứng sơ sài. Đây là ràng buộc CỨNG ở tầng code
(giống emergency check): gọi 1 LLM ngắn phân loại, nếu là "mô tả triệu chứng để tìm bệnh mà
chưa đủ chi tiết" -> buộc hỏi lại, model không có cơ hội phỏng đoán.

An toàn: LLM phân loại lỗi / trả JSON hỏng -> mặc định KHÔNG chặn (để luồng thường xử lý),
tránh chặn oan câu kiến thức. Đây là lớp thêm, không thay thế prompt.
"""
from __future__ import annotations

import json
import re

from src.serving.guards.input_guard import _norm

# --- Rule lọc RẺ (không gọi LLM) — chỉ gọi LLM cho phần MỜ ---
# Ý định KHÔNG phải "mô tả triệu chứng tìm bệnh" -> cho qua ngay, khỏi tốn API.
_NOT_SYMPTOM_HINTS = [
    "la gi", "la benh gi", "nghia la", "cach ", "lam sao", "lam the nao",
    "nen lam gi", "co nen", "bao nhieu", "lieu", "uong gi", "dieu tri",
    "chua ", "phong ngua", "phong tranh", "nguyen nhan", "trieu chung cua",
    "co may loai", "khac nhau", "so sanh",
]
# Từ chỉ triệu chứng người dùng tự kể (dấu hiệu ĐANG mô tả tình trạng bản thân).
_SYMPTOM_WORDS = [
    "dau", "sot", "ho", "buon non", "non", "chong mat", "met", "kho tho",
    "tieu chay", "phat ban", "ngua", "sung", "chay mau", "tuc nguc", "nhuc",
    "o e", "kho chiu", "te", "co giat", "ngat",
]

_GATE_SYSTEM = (
    "Bạn là bộ phân loại ý định cho trợ lý y tế. Đọc câu hỏi mới nhất của người dùng "
    "(kèm ngữ cảnh nếu có) và phân loại. Trả về DUY NHẤT một JSON, không giải thích:\n"
    '{"symptom_seeking": true|false, "enough_detail": true|false, "question": "..."}\n\n'
    "- symptom_seeking=true nếu người dùng MÔ TẢ TRIỆU CHỨNG của bản thân/người nhà để hỏi "
    "nguyên nhân/bệnh (vd 'đau bụng bên phải 2 ngày', 'đau đầu buồn nôn'). "
    "false nếu là câu hỏi KIẾN THỨC ('sốt virus là gì'), CÁCH CHĂM SÓC/XỬ TRÍ "
    "('trẻ sốt nên làm gì'), hỏi THUỐC/LIỀU, hay ngoài y tế.\n"
    "- enough_detail=true nếu đã đủ chi tiết để tra cứu an toàn (vị trí rõ, tính chất, "
    "triệu chứng kèm, thời gian, tuổi khi cần). Mô tả sơ sài -> false.\n"
    "- question: nếu symptom_seeking=true và enough_detail=false, viết 1-2 câu hỏi làm rõ "
    "NGẮN bằng tiếng Việt (hỏi vị trí/tính chất/triệu chứng kèm/tuổi). KHÔNG nêu tên bệnh nào. "
    "Ngược lại để chuỗi rỗng."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def clarification_question(query: str, history: list, engine) -> str | None:
    """Trả câu hỏi làm rõ (str) nếu cần chặn để hỏi thêm; None nếu cho phép trả lời.

    - Đã hỏi lại rồi (history có [HỎI LẠI]) -> không gác nữa (tránh lặp vô tận).
    - LLM lỗi/JSON hỏng -> None (không chặn oan).
    """
    already_asked = any(
        h.get("role") == "assistant" and "[HỎI LẠI]" in (h.get("content") or "")
        for h in (history or [])
    )
    if already_asked:
        return None

    # --- Lọc RẺ trước khi gọi LLM (tiết kiệm API) ---
    q = _norm(query)
    # (a) có dấu hiệu là câu hỏi kiến thức/chăm sóc/thuốc -> chắc chắn KHÔNG phải mô tả
    #     triệu chứng -> cho qua, khỏi gọi LLM.
    if any(h in q for h in _NOT_SYMPTOM_HINTS):
        return None
    # (b) không chứa TỪ TRIỆU CHỨNG nào -> khó là "kể triệu chứng tìm bệnh" -> cho qua.
    if not any(w in q for w in _SYMPTOM_WORDS):
        return None
    # (c) còn lại = MỜ (có từ triệu chứng, không có dấu hiệu hỏi kiến thức) -> mới gọi LLM.

    ctx = " ".join(h.get("content", "") for h in (history or [])[-4:]
                   if h.get("role") == "user")
    user = f"Ngữ cảnh trước: {ctx}\n\nCâu mới: {query}" if ctx else query

    try:
        raw = engine.generate(_GATE_SYSTEM, user)
        m = _JSON_RE.search(raw)
        if not m:
            return None
        d = json.loads(m.group(0))
    except Exception as e:
        print(f"[gate] phân loại lỗi ({e}); bỏ qua gate.")
        return None

    if d.get("symptom_seeking") and not d.get("enough_detail"):
        q = (d.get("question") or "").strip()
        return q or "Bạn mô tả rõ hơn triệu chứng giúp mình nhé: vị trí, tính chất, đã bao lâu, có kèm gì khác không?"
    return None
