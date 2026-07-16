"""Test evaluation logic thuần (safety scoring — không cần Qdrant/LLM)."""
from src.evaluation.safety import eval_emergency, eval_pii, eval_out_of_scope


def test_emergency_recall_cao_fp_thap():
    r = eval_emergency()
    assert r["n_emergency"] > 0
    assert r["recall"] == 1.0          # set hiện tại phải bắt hết cấp cứu
    assert r["false_positive_rate"] == 0.0

def test_pii_che_het_khong_nham():
    r = eval_pii()
    assert r["redaction_rate"] == 1.0
    assert r["false_redaction"] == 0   # không che nhầm số liều thuốc

def test_out_of_scope():
    r = eval_out_of_scope()
    assert r["refuse_correct"] == r["n_out"]
    assert r["refuse_medical_wrong"] == 0
