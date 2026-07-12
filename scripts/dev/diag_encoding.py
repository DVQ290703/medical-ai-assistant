"""Dev: soi code-point ký tự lỗi font trong PDF kcb.vn để map đúng trong kb_ingest.

Chạy: python scripts/dev/diag_encoding.py
"""
import sys
import io
import unicodedata

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import fitz  # noqa: E402

PDF = "data/raw/kb/pdf/3128_QD-BYT_Huong-dan-chan-doan-va-dieu-tri-ung-thu-vu.pdf"

doc = fitz.open(PDF)
text = "\n".join(p.get_text("text") for p in doc[:5])
doc.close()

# dòng ĐẠI ...NG đầu tiên
line = next(ln for ln in text.splitlines() if "NG" in ln and "Đ" in ln)
head = line.split("..")[0].strip()
print("cụm:", repr(head))
print("\ntừng ký tự:")
for ch in head:
    try:
        name = unicodedata.name(ch)
    except ValueError:
        name = "?"
    flag = "  <-- NGHI NGỜ" if ord(ch) > 0x1EF9 or (0x0180 <= ord(ch) <= 0x024F) else ""
    print(f"  {ch!r}  U+{ord(ch):04X}  {name}{flag}")
