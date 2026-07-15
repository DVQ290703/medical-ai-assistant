"""Test output_guard (rule-based, không cần model)."""
from src.serving.guards.output_guard import check_output, _redact_pii


def test_che_sdt():
    t, n = _redact_pii("Gọi cho tôi số 0912345678 nhé")
    assert "0912345678" not in t
    assert n == 1


def test_che_email():
    t, n = _redact_pii("Email: benhnhan@gmail.com")
    assert "benhnhan@gmail.com" not in t
    assert n == 1


def test_cite_ok_khong_canh_bao():
    r = check_output("Paracetamol dùng khi sốt [1].", kind="normal", has_sources=True)
    assert r.warnings == []


def test_thieu_citation_canh_bao():
    r = check_output("Bạn nên uống thuốc này thường xuyên để khỏi bệnh nhanh.",
                     kind="normal", citation_required=True, has_sources=False)
    assert any("no_citation" in w for w in r.warnings)


def test_tra_loi_dai_khong_dan_nguon_low_grounding():
    long = "Đây là một khẳng định y khoa dài không có dẫn nguồn nào cả. " * 8  # >60 từ
    r = check_output(long, kind="normal", has_sources=False)
    assert any("low_grounding" in w for w in r.warnings)


def test_emergency_khong_soi_citation():
    # câu cấp cứu không phải trả lời tri thức -> không cảnh báo citation
    r = check_output("GỌI 115 NGAY", kind="emergency")
    assert r.warnings == []


def test_khong_xoa_noi_dung_y_khoa():
    # liều thuốc có số -> KHÔNG bị nhầm là PII/ẩn (số ngắn, có đơn vị)
    t, n = _redact_pii("Paracetamol 500 mg mỗi 6 giờ")
    assert "500" in t and n == 0
