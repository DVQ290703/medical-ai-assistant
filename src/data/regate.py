"""Phase 1 — Re-gate: phân loại LẠI data ĐÃ DỊCH bằng gate mới, KHÔNG gọi API ($0).

Vì sao làm được không cần EN gốc: mỗi mẫu bị loại đã lưu `_reasons`. Ta áp LUẬT MỚI lên
các lý do cũ, cộng tính lại vài thứ chỉ cần bản tiếng Việt (tỷ lệ ký tự Việt, whitelist
thuốc, len_ratio đã ghi trong reason). Mẫu bị loại OAN theo luật cũ (đếm câu, từ thường bắt
nhầm làm thuốc, đáp án MCQ ngắn) sẽ được VỚT về kept.

Vẫn giữ loại đúng: rớt liều, response dài còn nguyên tiếng Anh, độ dài lệch thật, thuốc THẬT mất.

Usage (notebook):
    from src.data.regate import regate
    regate(kept="/kaggle/working/seed_vi.jsonl",
           rejected="/kaggle/working/seed_vi.rejected.jsonl",
           needs_human="/kaggle/working/seed_vi.needs_human.jsonl",
           out="/kaggle/working/seed_vi.regated.jsonl")
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .translate import _VI_CHARS, _DRUG_WHITELIST


def _load(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def redecide(rec: dict) -> tuple[list, list]:
    """Áp gate MỚI lên _reasons cũ. Trả (blocking, review)."""
    old = rec.get("_reasons", [])
    resp = rec.get("response", "") or ""
    blocking, review = [], []
    for r in old:
        if r.startswith("MẤT liều"):
            blocking.append(r)                              # rớt liều -> vẫn loại
        elif "GIỮ NGUYÊN tiếng Anh" in r or "chưa dịch" in r:
            if len(resp.strip()) >= 40:                     # dài mà chưa dịch -> loại
                blocking.append(r)
            # ngắn (đáp án MCQ) -> bỏ qua
        elif "co cụt" in r or "CoT ngắn" in r:
            pass                                            # đếm câu đã bỏ -> không chặn
        elif "CẮT CỤT" in r or "cắt cuối" in r:
            pass                                            # gate mới không loại vì lý do này
        elif "len_ratio" in r:
            m = re.search(r"len_ratio=([\d.]+)", r)
            ratio = float(m.group(1)) if m else 1.0
            if not (0.4 <= ratio <= 2.5):
                blocking.append(r)                          # lệch độ dài thật -> loại
        elif "tiếng Việt" in r:                             # "ít ký tự tiếng Việt"
            if len(resp) >= 40:
                vi = sum(c in _VI_CHARS for c in resp) / max(len(resp), 1)
                if vi < 0.03:
                    blocking.append(r)
        elif "tên thuốc" in r:          # bắt cả "Nghi mất tên thuốc" (mới) và "MẤT tên thuốc" (cũ)
            drugs = re.findall(r"'([^']+)'", r)
            real = sorted(d for d in drugs if d.lower() in _DRUG_WHITELIST)
            if real:
                review.append(f"Nghi mất tên thuốc (review): {real}")
            # thuốc "giả" (từ thường / hormone viết tắt) -> bỏ
        else:
            review.append(r)                                # lý do lạ -> để review cho an toàn
    return blocking, review


def regate(kept: str, rejected: str, needs_human: str, out: str) -> dict:
    kept_recs = _load(kept)
    to_recheck = _load(rejected) + _load(needs_human)

    new_kept = list(kept_recs)          # kept cũ vẫn pass gate mới (nới hơn)
    still_rej, still_hum, recovered = [], [], 0

    for rec in to_recheck:
        blocking, review = redecide(rec)
        clean = {k: v for k, v in rec.items() if not k.startswith("_")}
        if blocking:
            still_rej.append({**clean, "_reasons": blocking})
        elif review:
            still_hum.append({**clean, "_reasons": review})
        else:
            new_kept.append(clean)      # VỚT về kept
            recovered += 1

    out_p = Path(out); out_p.parent.mkdir(parents=True, exist_ok=True)
    _dump(out_p, new_kept)
    _dump(out_p.with_suffix(".rejected.jsonl"), still_rej)
    _dump(out_p.with_suffix(".needs_human.jsonl"), still_hum)

    stats = {"kept_cũ": len(kept_recs), "đưa_vào_xét_lại": len(to_recheck),
             "vớt_được": recovered, "kept_mới": len(new_kept),
             "vẫn_rejected": len(still_rej), "vẫn_needs_human": len(still_hum)}
    print(f"[regate] {stats}")
    return stats


def _dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")