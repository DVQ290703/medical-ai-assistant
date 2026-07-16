"""Policy — Disclaimer y tế theo ngữ cảnh, chèn vào cuối câu trả lời.

Disclaimer chung LUÔN có. need_doctor -> thêm nhắc mạnh gặp bác sĩ (liều/kê đơn/chẩn đoán).
System prompt đã yêu cầu LLM tự nhắc, nhưng policy ÉP disclaimer ở tầng app -> đảm bảo
100% có (không phó mặc model).
"""
from __future__ import annotations

_BASE = ("\n\n— Lưu ý: đây là thông tin tham khảo, KHÔNG thay thế chẩn đoán/điều trị của "
         "bác sĩ. Hãy đi khám khi triệu chứng nặng, kéo dài hoặc bất thường.")

_DOCTOR = ("\n— Với liều dùng/kê đơn/chẩn đoán cụ thể cho trường hợp của bạn, cần bác sĩ "
           "trực tiếp thăm khám và chỉ định.")


def disclaimer(need_doctor: bool = False) -> str:
    return _BASE + (_DOCTOR if need_doctor else "")


def has_disclaimer(text: str) -> bool:
    """Kiểm answer đã có disclaimer chưa (tránh chèn trùng nếu LLM tự nhắc rồi)."""
    t = text.lower()
    return "tham khảo" in t and "bác sĩ" in t
