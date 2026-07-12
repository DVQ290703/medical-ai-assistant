import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from src.knowledge.chunk import (
    chunk_text, split_by_heading, _looks_like_dosage_block, _n_tokens, load_chunk_cfg
)

cfg = load_chunk_cfg()
print("cfg:", cfg)
size, overlap = cfg["size"], cfg["overlap"]

# 1) heading-split
t1 = """1. Mục đích
Phục hồi áp lực âm khoang màng phổi.
2. Chỉ định
2.1 Tràn khí màng phổi tự phát.
2.2 Tràn máu màng phổi do chấn thương.
3. Chống chỉ định
Không có chống chỉ định tuyệt đối."""
secs = split_by_heading(t1)
print(f"\n[heading-split] {len(secs)} section:")
for s in secs:
    print("  H=", repr(s["heading"][:30]), "| body=", repr(s["body"][:40]))

# 2) dosage-guard
dose = """Liều dùng
Paracetamol 500 mg mỗi 6 giờ
Ibuprofen 400 mg mỗi 8 giờ
Amoxicillin 500 mg 3 lần/ngày
Vitamin C 1000 mg mỗi ngày"""
print("\n[dosage-guard] is_dosage:", _looks_like_dosage_block(dose))
print("  -> chunks:", len(chunk_text(dose, size, overlap)), "(mong đợi 1)")

# 3) recursive cho text dài không heading
long = "Đây là một đoạn văn y khoa dài. " * 200
ch = chunk_text(long, size, overlap)
print(f"\n[recursive] long text -> {len(ch)} chunks")
print("  token mỗi chunk:", [_n_tokens(c) for c in ch][:6], f"(<= ~{size})")
print("  max token:", max(_n_tokens(c) for c in ch))
