"""Test logic ranking của retriever (thuần, không cần Qdrant/model)."""
from dataclasses import dataclass

from src.knowledge.retriever import rank_candidates


@dataclass
class FakeCand:
    payload: dict
    point_id: str = "x"
    collection: str = "vinmec_kb"


def _c(source, text, title="T", url=None):
    # mặc định url + text khác nhau theo text để KHÔNG bị dedup ngoài ý muốn
    return FakeCand(payload={"source": source, "text": text, "title": title,
                             "url": url if url is not None else f"u/{text}",
                             "chunk_id": text})


def test_threshold_loai_hit_diem_thap():
    cands = [_c("vinmec-article", "A"), _c("vinmec-article", "B")]
    hits = rank_candidates(cands, [0.9, 0.2], {}, min_score=0.5, top_n=10)
    assert len(hits) == 1
    assert hits[0].rerank_score == 0.9


def test_threshold_tat_ca_thap_tra_rong():
    # query không liên quan -> mọi điểm dưới ngưỡng -> [] (không bịa)
    cands = [_c("vinmec-article", "A"), _c("byt-kcb", "B")]
    hits = rank_candidates(cands, [0.1, 0.05], {}, min_score=0.5, top_n=10)
    assert hits == []


def test_source_priority_tie_break_khi_diem_sat():
    # 2 hit điểm rerank BẰNG nhau -> byt-kcb ưu tiên lên đầu
    cands = [_c("vinmec-article", "A"), _c("byt-kcb", "B")]
    hits = rank_candidates(cands, [0.80, 0.80], {"byt-kcb": 2, "vinmec-article": 1},
                           min_score=0.0, top_n=10)
    assert hits[0].source == "byt-kcb"


def test_source_priority_khong_lan_at_relevance():
    # article điểm cao hơn HẲN -> vẫn thắng dù byt-kcb có bonus
    cands = [_c("vinmec-article", "A"), _c("byt-kcb", "B")]
    hits = rank_candidates(cands, [0.95, 0.60], {"byt-kcb": 2, "vinmec-article": 1},
                           min_score=0.0, top_n=10)
    assert hits[0].source == "vinmec-article"   # bonus 0.02 không lấn át chênh 0.35


def test_top_n_gioi_han_so_hit():
    cands = [_c("vinmec-article", f"text-{i}") for i in range(10)]
    hits = rank_candidates(cands, [0.9] * 10, {}, min_score=0.0, top_n=4)
    assert len(hits) == 4


def test_dedup_cung_url_giu_diem_cao():
    # 2 chunk cùng url, điểm khác -> giữ cái cao hơn, bỏ cái thấp
    cands = [_c("vinmec-article", "phần A", url="bai1"),
             _c("vinmec-article", "phần B", url="bai1"),
             _c("vinmec-article", "khác", url="bai2")]
    hits = rank_candidates(cands, [0.9, 0.8, 0.7], {}, min_score=0.0, top_n=10)
    urls = [h.url for h in hits]
    assert urls.count("bai1") == 1      # chỉ 1 chunk từ bai1
    assert "bai2" in urls
    assert hits[0].rerank_score == 0.9  # giữ cái điểm cao


def test_dedup_text_gan_trung():
    # cùng ~80 ký tự đầu (url rỗng, vd PDF) -> coi là trùng
    same = "Cùng một đoạn mở đầu giống hệt nhau để test near duplicate theo text đầu"
    cands = [_c("byt-kcb", same + " phần 1", url=""),
             _c("byt-kcb", same + " phần 2", url="")]
    hits = rank_candidates(cands, [0.9, 0.85], {}, min_score=0.0, top_n=10)
    assert len(hits) == 1
