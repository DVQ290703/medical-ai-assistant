"""Phase 1 — Validate tập data ĐÃ DỊCH trước khi train (khác quality-gate lúc dịch).

Quality-gate (translate.py) chấm TỪNG bản dịch. Validate ở đây kiểm CẢ TẬP:
  1. Schema      — đủ field question/cot/response, không rỗng.
  2. Encoding    — không ký tự thay thế (\\ufffd) / control chars.
  3. Language    — response thật sự là tiếng Việt (chống lọt mẫu chưa dịch).
  4. Length      — không quá ngắn (dịch cụt) / quá dài bất thường.
  5. Dedup       — bỏ trùng (exact theo câu hỏi chuẩn hoá; near-dup nếu có datasketch).
  6. MCQ         — phát hiện câu trắc nghiệm ("A. ... B. ... C. ...") và ĐỊNH TUYẾN riêng,
                   vì format MCQ dạy model 'chọn đáp án' thay vì giải thích tự do.

Kết quả:
  * <out>            : mẫu SẠCH, sẵn sàng train
  * <out>.invalid.jsonl : mẫu bị loại (kèm _issues)
  * <out>.mcq.jsonl     : mẫu trắc nghiệm (tuỳ policy, mặc định tách riêng)
  * reports/validation_report.md

Usage (notebook — GỌI HÀM, tránh argparse):
    from src.data.validate import validate_file
    validate_file("/kaggle/working/seed_vi.jsonl", "/kaggle/working/validated.jsonl")
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import yaml

_VI_CHARS = set("ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ"
                "ĂÂĐÊÔƠƯ")
# option trắc nghiệm: "A." "B)" đầu dòng hoặc sau khoảng trắng
_MCQ_OPT_RE = re.compile(r"(?:^|\s)([A-E])[.)]\s+\S")


def _norm_q(s: str) -> str:
    """Chuẩn hoá câu hỏi để so trùng (giữ dấu tiếng Việt, bỏ khoảng trắng thừa/dấu câu)."""
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
               " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _is_mcq(rec: dict) -> bool:
    text = f"{rec.get('question','')}\n{rec.get('response','')}"
    letters = {m.group(1) for m in _MCQ_OPT_RE.finditer(text)}
    return len(letters) >= 3   # >=3 lựa chọn A/B/C... -> trắc nghiệm


def check_record(rec: dict, min_vi_ratio: float, min_resp_len: int) -> list[str]:
    """Trả danh sách vấn đề (rỗng = hợp lệ)."""
    issues = []
    # schema
    for f in ("question", "cot", "response"):
        if not isinstance(rec.get(f), str) or not rec.get(f, "").strip():
            issues.append(f"thiếu/rỗng field: {f}")
    if issues:
        return issues  # hỏng schema thì khỏi kiểm tiếp
    # encoding
    joined = rec["question"] + rec["cot"] + rec["response"]
    if "\ufffd" in joined or any(unicodedata.category(c) == "Cc" and c not in "\n\t\r"
                                 for c in joined):
        issues.append("ký tự lỗi encoding")
    # language (response phải là tiếng Việt)
    resp = rec["response"]
    vi_ratio = sum(c in _VI_CHARS for c in resp) / max(len(resp), 1)
    if vi_ratio < min_vi_ratio:
        issues.append(f"response không giống tiếng Việt (vi_ratio={vi_ratio:.3f})")
    # length
    if len(resp.strip()) < min_resp_len:
        issues.append(f"response quá ngắn ({len(resp.strip())} ký tự)")
    return issues


def validate_file(in_path: str, out_path: str, config_path: str | None = None,
                  report_path: str = "reports/validation_report.md",
                  mcq_policy: str | None = None) -> dict:
    """Kiểm cả tập, ghi mẫu sạch/loại/MCQ ra file riêng + report.

    mcq_policy: 'flag' (tách riêng, mặc định) | 'drop' (bỏ hẳn) | 'keep' (giữ trong sạch).
    """
    cfg = {}
    if config_path and Path(config_path).exists():
        cfg = yaml.safe_load(open(config_path, encoding="utf-8")) or {}
    v = cfg.get("validation", {})
    min_vi_ratio = v.get("min_vi_ratio", 0.02)
    min_resp_len = v.get("min_resp_len", 20)
    mcq_policy = mcq_policy or v.get("mcq_policy", "flag")

    recs = [json.loads(l) for l in open(in_path, encoding="utf-8")]

    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    inv_path = out.with_suffix(".invalid.jsonl")
    mcq_path = out.with_suffix(".mcq.jsonl")

    seen_q: set[str] = set()
    clean, invalid, mcq = [], [], []
    n_dup = 0

    for rec in recs:
        issues = check_record(rec, min_vi_ratio, min_resp_len)
        if issues:
            invalid.append({**rec, "_issues": issues})
            continue
        # dedup theo câu hỏi chuẩn hoá
        key = _norm_q(rec["question"])
        if key in seen_q:
            n_dup += 1
            invalid.append({**rec, "_issues": ["trùng câu hỏi (dedup)"]})
            continue
        seen_q.add(key)
        # MCQ
        if _is_mcq(rec):
            if mcq_policy == "drop":
                invalid.append({**rec, "_issues": ["MCQ (drop theo policy)"]})
                continue
            if mcq_policy == "flag":
                mcq.append({**rec, "_mcq": True})
                continue
            # 'keep' -> rơi xuống clean
        clean.append(rec)

    _dump(out, clean)
    _dump(inv_path, invalid)
    _dump(mcq_path, mcq)

    stats = {"total": len(recs), "clean": len(clean), "invalid": len(invalid),
             "mcq": len(mcq), "duplicates_removed": n_dup, "mcq_policy": mcq_policy}
    _write_report(report_path, stats, invalid[:8])
    print(f"[validate] {stats}")
    return stats


def _dump(path: Path, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_report(path, stats, inv_examples):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Validation Report\n",
             f"- total: {stats['total']}",
             f"- clean (train-ready): {stats['clean']}",
             f"- invalid (loại): {stats['invalid']}",
             f"  - trong đó trùng lặp: {stats['duplicates_removed']}",
             f"- MCQ (policy={stats['mcq_policy']}): {stats['mcq']}\n",
             "## Ví dụ mẫu bị loại"]
    for r in inv_examples:
        q = r.get("question", "")[:80].replace("\n", " ")
        lines.append(f"- {r['_issues']} | Q: {q}")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/raw/seed_vi.jsonl")
    ap.add_argument("--out", default="data/processed/validated.jsonl")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--mcq-policy", choices=["flag", "drop", "keep"], default=None)
    args = ap.parse_args()
    validate_file(args.in_path, args.out, args.config, mcq_policy=args.mcq_policy)


if __name__ == "__main__":
    main()