"""Policy — Rules: phân loại query để engine quyết định hành động.

Khác Guard (kiểm tra an toàn), Rules chỉ TRẢ NHÃN. engine.py đọc nhãn -> chọn hành động.

Nhãn:
  - is_emergency   : dấu hiệu cấp cứu (tái dùng input_guard.emergency_check)
  - is_out_of_scope: câu hỏi ngoài phạm vi y tế (khi refuse_out_of_scope=true)
  - need_doctor    : chủ đề nên khuyên gặp bác sĩ mạnh hơn (liều/kê đơn/chẩn đoán cá nhân)
"""
from __future__ import annotations

import re

from src.serving.guards.input_guard import emergency_check, _norm


# Dấu hiệu CHỦ ĐỀ Y TẾ — để phát hiện out-of-scope (câu KHÔNG có gì y tế).
# Match theo TỪ (word boundary), KHÔNG substring — tránh "ho" dính "hom"/"cho".
# Cụm nhiều từ (vd "trieu chung") match nguyên cụm.
_MEDICAL_HINTS = [
    "benh", "trieu chung", "dau", "sot", "ho", "thuoc", "dieu tri", "chan doan",
    "bac si", "benh vien", "kham", "xet nghiem", "lieu", "tiem", "vaccine", "vacxin",
    "ung thu", "tim mach", "gan", "than", "phoi", "da day", "huyet ap", "tieu duong",
    "suc khoe", "nhiem", "viem", "man tinh", "cap tinh", "phau thuat",
    "kinh nguyet", "thai", "tre em", "di ung", "ngua", "buon non", "chong mat",
]
_HINT_RE = re.compile(r"\b(" + "|".join(re.escape(h) for h in _MEDICAL_HINTS) + r")\b")

# Chủ đề cần nhấn mạnh gặp bác sĩ (kê đơn/liều/chẩn đoán cá nhân).
_NEED_DOCTOR = [
    "lieu dung", "uong bao nhieu", "ke don", "don thuoc", "chan doan",
    "co phai bi", "co bi khong", "co nen uong", "co nen dung",
]


def is_emergency(query: str) -> bool:
    return emergency_check(query) is not None


def is_out_of_scope(query: str) -> bool:
    """True nếu câu hỏi KHÔNG chứa dấu hiệu y tế nào (ngoài phạm vi trợ lý)."""
    return _HINT_RE.search(_norm(query)) is None


def need_doctor(query: str) -> bool:
    q = _norm(query)
    return any(k in q for k in _NEED_DOCTOR)
