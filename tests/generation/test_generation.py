"""Test logic thuần tầng generation (không cần API/Qdrant/GPU)."""
from dataclasses import dataclass

from src.serving.guards.input_guard import emergency_check
from src.serving.citation import cited_indices, build_sources, format_sources
from src.prompting.builder import format_context, build_user_prompt


@dataclass
class FakeHit:
    title: str = "T"
    text: str = "nội dung"
    source: str = "byt-kcb"
    url: str = "http://x"


# ---- emergency_check (an toàn: chặn cấp cứu) ----

def test_emergency_bat_dau_nguc():
    assert emergency_check("tôi bị đau ngực dữ dội lan ra tay trái") is not None

def test_emergency_bat_co_giat():
    assert emergency_check("con tôi co giật, tím tái") is not None

def test_emergency_bat_ca_khi_thieu_dau():
    # gõ không dấu vẫn phải bắt
    assert emergency_check("kho tho du doi") is not None

def test_emergency_bo_qua_cau_thuong():
    assert emergency_check("tôi bị đau đầu nhẹ, hơi mệt") is None

def test_emergency_bo_qua_hoi_kien_thuc():
    assert emergency_check("đái tháo đường type 2 điều trị thế nào") is None


# ---- citation ----

def test_cited_indices_unique_giu_thu_tu():
    assert cited_indices("A [2] B [1] C [2] D [3]") == [2, 1, 3]

def test_build_sources_map_dung_hit():
    hits = [FakeHit(title="Sốt"), FakeHit(title="Ho"), FakeHit(title="Đau")]
    src = build_sources("theo [1] và [3]", hits)
    assert [s["n"] for s in src] == [1, 3]
    assert src[0]["title"] == "Sốt" and src[1]["title"] == "Đau"

def test_build_sources_bo_qua_so_ngoai_pham_vi():
    hits = [FakeHit()]
    assert build_sources("dùng [5]", hits) == []   # [5] vượt số hit -> bỏ

def test_format_sources_rong_khi_khong_co():
    assert format_sources([]) == ""


# ---- builder ----

def test_format_context_danh_so():
    hits = [FakeHit(title="A"), FakeHit(title="B")]
    ctx = format_context(hits)
    assert "[1] A" in ctx and "[2] B" in ctx

def test_build_user_prompt_co_cau_hoi_va_chi_dan():
    p = build_user_prompt("ho khan là gì", [FakeHit()])
    assert "ho khan là gì" in p
    assert "[số]" in p or "trích" in p
