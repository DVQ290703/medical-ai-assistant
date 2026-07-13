"""Test logic ranking của retriever (thuần, không cần Qdrant/model)."""
from dataclasses import dataclass

from src.knowledge.retriever import rank_candidates


@dataclass
class FakeCand:
    payload: dict
    point_id: str = "x"
    collection: str = "vinmec_kb"


def _c(source, text="t", title="T"):
    return FakeCand(payload={"source": source, "text": text, "title": title,
                             "url": "u", "chunk_id": "c"})


def test_threshold_loai_hit_diem_thap():
    cands = [_c("vinmec-article"), _c("vinmec-article")]
    scores = [0.9, 0.2]
    hits = rank_candidates(cands, scores, {}, min_score=0.5, top_n=10)
    assert len(hits) == 1
    assert hits[0].rerank_score == 0.9


def test_threshold_tat_ca_thap_tra_rong():
    # query không liên quan -> mọi điểm dưới ngưỡng -> [] (không bịa)
    cands = [_c("vinmec-article"), _c("byt-kcb")]
    hits = rank_candidates(cands, [0.1, 0.05], {}, min_score=0.5, top_n=10)
    assert hits == []


def test_source_priority_tie_break_khi_diem_sat():
    # 2 hit điểm rerank BẰNG nhau -> byt-kcb ưu tiên lên đầu
    cands = [_c("vinmec-article"), _c("byt-kcb")]
    scores = [0.80, 0.80]
    hits = rank_candidates(cands, scores, {"byt-kcb": 2, "vinmec-article": 1},
                           min_score=0.0, top_n=10)
    assert hits[0].source == "byt-kcb"


def test_source_priority_khong_lan_at_relevance():
    # article điểm cao hơn HẲN -> vẫn thắng dù byt-kcb có bonus
    cands = [_c("vinmec-article"), _c("byt-kcb")]
    scores = [0.95, 0.60]
    hits = rank_candidates(cands, scores, {"byt-kcb": 2, "vinmec-article": 1},
                           min_score=0.0, top_n=10)
    assert hits[0].source == "vinmec-article"   # bonus 0.02 không lấn át chênh 0.35


def test_top_n_gioi_han_so_hit():
    cands = [_c("vinmec-article") for _ in range(10)]
    scores = [0.9] * 10
    hits = rank_candidates(cands, scores, {}, min_score=0.0, top_n=4)
    assert len(hits) == 4
