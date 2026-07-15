"""Guards — Output: kiểm tra câu trả lời SAU khi LLM sinh, trước khi trả người dùng.

3 kiểm tra rule-based (KHÔNG cần model — đúng tinh thần guard nhẹ, chạy được ở tầng app):
  1. ÉP CITATION: câu trả lời 'normal' phải trích ít nhất 1 [số]. Không có -> cờ cảnh báo.
  2. GROUNDING (nhẹ): trả lời dài + khẳng định mạnh mà KHÔNG trích nguồn nào -> nghi bịa.
  3. PII FILTER: che số điện thoại / email / CCCD lỡ lọt (bảo vệ dữ liệu cá nhân).

Nguyên tắc an toàn: guard CHỈ annotate/che PII, KHÔNG tự ý xoá nội dung y khoa (xoá nhầm
còn nguy hiểm hơn). Trả về (text_đã_xử_lý, list cảnh báo) để caller quyết hiển thị.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.serving.citation import cited_indices

# PII patterns (VN): điện thoại 10 số, email, CCCD/CMND 9-12 số đứng riêng
_PHONE_RE = re.compile(r"\b0\d{9}\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_ID_RE = re.compile(r"\b\d{9,12}\b")            # CCCD 12 / CMND 9 — chỉ cụm số dài đứng riêng


@dataclass
class GuardResult:
    text: str
    warnings: list = field(default_factory=list)
    pii_redacted: int = 0


def _redact_pii(text: str) -> tuple[str, int]:
    n = 0

    def sub(pattern, repl, s):
        nonlocal n
        s2, cnt = pattern.subn(repl, s)
        n += cnt
        return s2

    text = sub(_EMAIL_RE, "[email đã ẩn]", text)
    text = sub(_PHONE_RE, "[SĐT đã ẩn]", text)
    # CCCD/CMND: chỉ che cụm số 9-12 chữ số ĐỨNG RIÊNG (tránh nhầm liều/mã ICD).
    text = sub(_ID_RE, "[số định danh đã ẩn]", text)
    return text, n


def check_output(text: str, kind: str = "normal", citation_required: bool = True,
                 has_sources: bool = False) -> GuardResult:
    """Kiểm câu trả lời. kind: normal|emergency|no_info (chỉ soi kỹ 'normal').

    has_sources: caller đã gắn được nguồn hay chưa (từ build_sources).
    """
    warnings: list[str] = []

    # emergency / no_info: không phải câu trả lời tri thức -> chỉ che PII, bỏ qua citation.
    text, n_pii = _redact_pii(text)

    if kind == "normal":
        n_cites = len(cited_indices(text))
        if citation_required and n_cites == 0 and not has_sources:
            warnings.append("no_citation: câu trả lời không trích nguồn nào -> nghi thiếu căn cứ.")
        # grounding nhẹ: trả lời dài (>60 từ) mà 0 trích dẫn -> nghi bịa
        if len(text.split()) > 60 and n_cites == 0:
            warnings.append("low_grounding: trả lời dài nhưng không dẫn nguồn.")

    return GuardResult(text=text, warnings=warnings, pii_redacted=n_pii)
