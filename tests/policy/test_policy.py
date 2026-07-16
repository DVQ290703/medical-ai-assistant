"""Test policy layer (rules + engine + disclaimer, không cần model/Qdrant)."""
from src.serving.policy import rules
from src.serving.policy.engine import decide
from src.serving.policy.disclaimer import disclaimer, has_disclaimer


# ---- rules ----
def test_emergency():
    assert rules.is_emergency("đau ngực dữ dội lan tay trái") is True
    assert rules.is_emergency("đau đầu nhẹ") is False

def test_out_of_scope():
    assert rules.is_out_of_scope("thời tiết hôm nay thế nào") is True
    assert rules.is_out_of_scope("triệu chứng sốt xuất huyết") is False

def test_need_doctor():
    assert rules.need_doctor("tôi có nên uống thuốc này không") is True
    assert rules.need_doctor("bệnh sởi là gì") is False


# ---- engine (quyết định) ----
def test_decide_escalate():
    d = decide("con tôi co giật tím tái")
    assert d.action == "escalate" and d.message

def test_decide_refuse_out_of_scope():
    d = decide("kể chuyện cười cho tôi nghe")
    assert d.action == "refuse"

def test_decide_answer_normal():
    d = decide("bệnh tiểu đường ăn uống thế nào")
    assert d.action == "answer"

def test_decide_answer_need_doctor():
    d = decide("liều dùng paracetamol cho người lớn là bao nhiêu")
    assert d.action == "answer" and d.need_doctor is True


# ---- disclaimer ----
def test_disclaimer_base():
    t = disclaimer(need_doctor=False)
    assert "tham khảo" in t and "bác sĩ" in t

def test_disclaimer_need_doctor_manh_hon():
    assert len(disclaimer(need_doctor=True)) > len(disclaimer(need_doctor=False))

def test_has_disclaimer():
    assert has_disclaimer("... thông tin tham khảo, cần bác sĩ khám ...") is True
    assert has_disclaimer("Paracetamol dùng khi sốt.") is False
