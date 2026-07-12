"""Dev: verify fix encoding (TCVN font) + gom chunk nhỏ trên PDF kcb.vn thật.

Chạy: python scripts/dev/verify_kb_clean.py
(chỉ đọc PDF local đã tải, không cần mạng/GPU/Qdrant)
"""
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src.knowledge.kb_ingest import clean_pdf_text, _fix_tcvn, _strip_toc  # noqa: E402
from src.knowledge.chunk import chunk_text, load_chunk_cfg, _n_tokens       # noqa: E402
import fitz  # noqa: E402

PDF = "data/raw/kb/pdf/3128_QD-BYT_Huong-dan-chan-doan-va-dieu-tri-ung-thu-vu.pdf"

doc = fitz.open(PDF)
raw = "\n".join(p.get_text("text") for p in doc)
doc.close()

# 1) fix encoding
fixed = _fix_tcvn(raw)
print("=== FIX ENCODING ===")
print("  'HƢỚNG' còn trong text sau fix?", "HƢỚNG" in fixed, "(mong đợi False)")
print("  'HƯỚNG' xuất hiện?", "HƯỚNG" in fixed, "(mong đợi True)")

# 2) strip mục lục
before = len(raw.splitlines())
after = len(_strip_toc(raw).splitlines())
print("\n=== STRIP MỤC LỤC ===")
print(f"  dòng: {before} -> {after} (bỏ {before-after} dòng ToC)")

# 3) chunk sau khi làm sạch + gom nhỏ
clean = clean_pdf_text(raw)
cfg = load_chunk_cfg()
print(f"\n=== CHUNK (size={cfg['size']} min={cfg['min_size']}) ===")
chunks = chunk_text(clean, cfg["size"], cfg["overlap"], cfg["min_size"])
lens = sorted(_n_tokens(c) for c in chunks)
n = len(lens)
tiny = sum(1 for x in lens if x < 20)
print(f"  {n} chunk | min={lens[0]} median={lens[n//2]} max={lens[-1]} tok")
print(f"  chunk <20 tok: {tiny}/{n} = {tiny/n*100:.0f}% (trước fix: 35%)")

# so với KHÔNG gom (min_size=0)
no_merge = chunk_text(clean, cfg["size"], cfg["overlap"], 0)
print(f"  (không gom: {len(no_merge)} chunk -> gom còn {n})")

print("\n=== 5 chunk đầu sau làm sạch ===")
for c in chunks[:5]:
    print(f"  ({_n_tokens(c)} tok) {c[:100].replace(chr(10),' ')}")
