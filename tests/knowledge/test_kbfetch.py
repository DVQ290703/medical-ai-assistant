"""Verify kb_fetch + pdf loader end-to-end với 1 PDF thật từ kcb.vn."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.knowledge.kb_fetch import fetch_urls, _check_host

# 0) host guard: nguồn ngoài kcb.vn phải bị từ chối
try:
    _check_host("https://www.msdmanuals.com/vi/abc.pdf")
    print("[FAIL] host-guard KHÔNG chặn msdmanuals")
except SystemExit:
    print("[ok] host-guard chặn nguồn ngoài kcb.vn")

# 1) tải 1 PDF mẫu thật (HDĐT ung thư vú) — agent đã xác nhận URL này tồn tại
url = "https://kcb.vn/upload/2005611/20210723/3128_QD-BYT_Huong-dan-chan-doan-va-dieu-tri-ung-thu-vu.pdf"
n = fetch_urls([url])
print("tải được:", n)

# 2) mở bằng pymupdf -> trích text được không (không phải scan)?
if n:
    import fitz
    from pathlib import Path
    pdfs = list(Path("data/raw/kb/pdf").glob("*.pdf"))
    doc = fitz.open(pdfs[0])
    txt = "\n".join(p.get_text("text") for p in doc)
    print(f"\n[pdf] {pdfs[0].name}: {len(doc)} trang, {len(txt):,} ký tự")
    print(f"[pdf] {len(txt)/max(1,len(doc)):.0f} ký tự/trang (>100 = có text, không scan)")
    print("\n--- 500 ký tự đầu ---")
    print(txt[:500].replace("\n", " "))
    doc.close()
