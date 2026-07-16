"""Guards — Input: phát hiện CẤP CỨU (red flags) -> chặn trước khi gọi LLM.

emergency_check là LUẬT CỨNG ở tầng ứng dụng (KHÔNG dựa vào LLM/RAG). Với chatbot y tế,
dấu hiệu nguy hiểm tính mạng phải được chặn 100%, không phó mặc model tự nhận. Nếu khớp
red flag -> trả cảnh báo GỌI CẤP CỨU 115 NGAY, bỏ qua bước tư vấn thường.

Đây là an toàn TỐI THIỂU, không thay thế phán đoán y tế chuyên môn.
"""
from __future__ import annotations

import re
import unicodedata

# Cụm red-flag (chuẩn hoá không dấu để bắt cả gõ thiếu dấu). Mỗi cụm = 1 dấu hiệu nguy hiểm.
_RED_FLAGS = [
    "dau nguc du doi", "dau nguc lan", "dau that nguc", "tuc nguc kho tho",
    "kho tho du doi", "kho tho nang", "tim ngung dap", "ngung tho",
    "co giat", "co giat toan than", "sui bot mep",
    "ngat xiu", "ngat", "xiu", "bat tinh", "hon me", "li bi kho danh thuc", "lo mo",
    "liet nua nguoi", "meo mieng", "noi ngong dot ngot", "yeu liet tay chan dot ngot",
    "mau khong cam", "chay mau o at", "non ra mau", "di ngoai ra mau nhieu",
    "hoc mau", "hoc ra mau", "ho ra mau", "non mau", "oi mau", "mua mau",
    "ngo doc", "uong nham thuoc", "qua lieu thuoc", "uong thuoc tu tu",
    "kho tho tim tai", "tim tai", "soc phan ve", "di ung nang kho tho",
    "thop phong", "sot cao co giat", "cung co", "dau dau du doi dot ngot",
    "tu tu", "muon chet", "tu lam hai ban than",
]

_EMERGENCY_MESSAGE = (
    "⚠️ Dấu hiệu bạn mô tả CÓ THỂ là tình huống CẤP CỨU nguy hiểm tính mạng.\n\n"
    "HÃY GỌI NGAY 115 (cấp cứu) hoặc đến cơ sở y tế gần nhất NGAY LẬP TỨC. "
    "Nếu có người bên cạnh, nhờ họ hỗ trợ ngay.\n\n"
    "Đây là cảnh báo tự động, không thay thế đánh giá của nhân viên y tế — nhưng với các "
    "dấu hiệu này, KHÔNG nên chờ đợi hay tự tra cứu."
)


def _norm(s: str) -> str:
    """Bỏ dấu tiếng Việt + lowercase để so khớp cụm không phụ thuộc dấu."""
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("đ", "d")
    return re.sub(r"\s+", " ", s).strip()


def emergency_check(query: str) -> str | None:
    """Trả cảnh báo cấp cứu nếu query khớp red flag; None nếu không phải cấp cứu."""
    q = _norm(query)
    for flag in _RED_FLAGS:
        if flag in q:
            return _EMERGENCY_MESSAGE
    return None
