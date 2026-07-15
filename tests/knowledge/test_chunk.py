"""Test chunk structure-aware (không cần GPU/mạng)."""
from src.knowledge.chunk import (
    chunk_text, split_by_heading, chunk_section,
    _looks_like_dosage_block, _n_tokens,
)

SIZE, OVERLAP, MIN = 768, 96, 256


def test_heading_split_tach_dung_muc():
    t = ("1. Mục đích\nPhục hồi áp lực âm.\n"
         "2. Chỉ định\n2.1 Tràn khí màng phổi.\n"
         "3. Chống chỉ định\nKhông có tuyệt đối.")
    headings = [s["heading"] for s in split_by_heading(t)]
    assert any(h.startswith("1.") for h in headings)
    assert any(h.startswith("2.") for h in headings)
    assert any(h.startswith("3.") for h in headings)


def test_text_khong_heading_gom_1_section():
    secs = split_by_heading("Một đoạn văn không có mục đánh số nào cả.")
    assert len(secs) == 1
    assert secs[0]["heading"] == ""


def test_dosage_block_giu_nguyen_khong_tach():
    dose = ("Liều dùng\n"
            "Paracetamol 500 mg mỗi 6 giờ\n"
            "Ibuprofen 400 mg mỗi 8 giờ\n"
            "Amoxicillin 500 mg 3 lần/ngày\n"
            "Vitamin C 1000 mg mỗi ngày")
    assert _looks_like_dosage_block(dose) is True
    # dù ép size nhỏ, dosage block -> đúng 1 chunk (an toàn kê đơn)
    chunks = chunk_section({"heading": "", "body": dose}, size=20, overlap=4)
    assert len(chunks) == 1


def test_van_ban_thuong_khong_bi_coi_la_dosage():
    normal = ("Ung thư vú là bệnh thường gặp. "
              "Chẩn đoán dựa trên lâm sàng và cận lâm sàng. "
              "Điều trị tuỳ giai đoạn bệnh.")
    assert _looks_like_dosage_block(normal) is False


def test_text_ngan_giu_nguyen_1_chunk():
    chunks = chunk_text("1. Đại cương\nMột đoạn ngắn.", SIZE, OVERLAP)
    assert len(chunks) == 1


def test_text_dai_cat_nhieu_chunk_khong_vuot_size():
    long = "Đây là một câu y khoa dài. " * 400   # >> size
    chunks = chunk_text(long, SIZE, OVERLAP)
    assert len(chunks) > 1
    # _pack gộp tới khi VƯỢT size rồi mới cắt -> chunk có thể nhỉnh hơn size 1 câu.
    # Không được vượt quá nhiều (đảm bảo < max_size embed). Biên: ~1.2x size.
    assert all(_n_tokens(c) <= SIZE * 1.2 for c in chunks)


def test_gom_chunk_nho_giam_so_luong():
    t = "\n".join(f"{i}. Mục {i}\nNội dung ngắn của mục {i}." for i in range(1, 21))
    khong_gom = chunk_text(t, SIZE, OVERLAP, min_size=0)
    co_gom = chunk_text(t, SIZE, OVERLAP, min_size=MIN)
    assert len(co_gom) <= len(khong_gom)
