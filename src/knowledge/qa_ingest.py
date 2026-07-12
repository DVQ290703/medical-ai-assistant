"""Knowledge plane — Gộp Q&A đa nguồn vào corpus RAG hiện có, có DEDUP.

Nền: src/knowledge/ingest.py đã ingest phuocsang/vinmec-medical-qa (16k) ->
data/raw/vinmec.jsonl. Module này GỘP THÊM nguồn Q&A khác (urnus11 split medical_qa)
vào cùng file đó, khử trùng bằng answer-fingerprint.

Đo thực tế (2026-07): trùng chéo urnus11 ∩ phuocsang ~0% (1 dòng), urnus11 tự trùng
nội bộ ~3.4%. Nên gộp AN TOÀN, phần trùng nhỏ được dedup lọc sạch.

Fingerprint = _norm_q(answer)[:200] (NFC + lowercase + bỏ dấu câu). Dùng lại _norm_q
từ src/data/validate.py để nhất quán với dedup của data plane.

Nguồn đọc từ configs/rag.yaml khối `qa_extra`. field_map: {question, answer, url}.

Usage:
  python -m src.knowledge.qa_ingest --source vn-healthcare-qa --inspect --limit 20
  python -m src.knowledge.qa_ingest --source vn-healthcare-qa          # gộp thật
  python -m src.knowledge.qa_ingest --source vn-healthcare-qa --dry-run # đếm, không ghi
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.data.validate import _norm_q          # reuse chuẩn hoá dedup của data plane
from src.knowledge.ingest import authenticate


CONFIG_PATH = "configs/rag.yaml"
FP_LEN = 200                                    # số ký tự đầu answer làm fingerprint


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_qa_extra(cfg: dict, source_id: str) -> dict:
    entries = cfg.get("qa_extra") or []
    entry = next((e for e in entries if e.get("id") == source_id), None)
    if entry is None:
        ids = [e.get("id") for e in entries]
        raise SystemExit(f"Không thấy qa_extra id='{source_id}'. Có: {ids}")
    return entry


def fingerprint(answer: str) -> str:
    return _norm_q(answer)[:FP_LEN]


def load_existing_fingerprints(path: str) -> set[str]:
    """Fingerprint của corpus Q&A hiện có (để dedup nguồn mới so với nó)."""
    fps = set()
    p = Path(path)
    if not p.exists():
        return fps
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ans = r.get("answer") or ""
            if ans:
                fps.add(fingerprint(ans))
    return fps


def load_new_rows(entry: dict, token: str | None, limit: int | None) -> list[dict]:
    """Nạp nguồn Q&A mới từ HF, map -> {question, answer, url}."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Thiếu `datasets`. pip install datasets.") from e

    source = entry["source"]
    split = entry.get("split", "train")
    fm = entry["field_map"]                     # {question, answer, url}
    ds = load_dataset(source, split=split, token=token)
    print(f"[ok] Loaded {len(ds):,} rows từ {source}, split='{split}'")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    rows = []
    for rec in ds:
        q = (rec.get(fm["question"]) or "").strip()
        a = (rec.get(fm["answer"]) or "").strip()
        url = (rec.get(fm.get("url", "")) or "").strip() if fm.get("url") else ""
        if not q or not a:
            continue
        rows.append({"question": q, "answer": a, "url": url})
    return rows


def dedup(new_rows: list[dict], existing_fps: set[str]) -> tuple[list[dict], dict]:
    """Bỏ dòng trùng existing corpus + trùng nội bộ. Trả (kept, stats)."""
    kept = []
    seen = set(existing_fps)
    dup_cross = dup_self = 0
    for r in new_rows:
        fp = fingerprint(r["answer"])
        if fp in existing_fps:
            dup_cross += 1
            continue
        if fp in seen and fp not in existing_fps:
            dup_self += 1
            continue
        seen.add(fp)
        kept.append(r)
    stats = {
        "new_total": len(new_rows),
        "dup_cross": dup_cross,      # trùng với corpus hiện có
        "dup_self": dup_self,        # trùng nội bộ nguồn mới
        "kept": len(kept),
    }
    return kept, stats


def append_jsonl(rows: list[dict], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:      # APPEND, giữ nguyên phuocsang
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def merge_source(source_id: str, config_path: str = CONFIG_PATH,
                 do_inspect: bool = False, limit: int | None = None,
                 dry_run: bool = False) -> None:
    cfg = load_config(config_path)
    entry = get_qa_extra(cfg, source_id)
    target = entry["merge_into"]
    dedup_against = entry.get("dedup_against", target)

    token = authenticate(os.environ.get("HF_TOKEN")) if entry.get("type") == "hf" else None
    new_rows = load_new_rows(entry, token, limit)

    existing_fps = load_existing_fingerprints(dedup_against)
    print(f"[dedup] corpus hiện có: {len(existing_fps):,} fingerprint từ {dedup_against}")

    kept, stats = dedup(new_rows, existing_fps)
    print(f"[dedup] nguồn mới={stats['new_total']:,} | trùng chéo={stats['dup_cross']:,} | "
          f"trùng nội bộ={stats['dup_self']:,} | GIỮ={stats['kept']:,}")

    if do_inspect:
        print("\n=== 3 mẫu GIỮ LẠI ===")
        for r in kept[:3]:
            print(f"  Q: {r['question'][:100]}")
            print(f"  A: {r['answer'][:150]}")
            print(f"  url: {r['url']}\n")

    if dry_run:
        print("[dry-run] không ghi.")
        return
    if limit and do_inspect:
        print("[note] Đang --inspect + --limit -> không ghi. Bỏ 2 cờ để gộp thật.")
        return

    append_jsonl(kept, target)
    print(f"[qa_ingest] APPEND {len(kept):,} Q&A -> {target}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Gộp Q&A đa nguồn + dedup -> corpus RAG")
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--source", required=True, help="id trong qa_extra của rag.yaml")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="đếm dedup, không ghi")
    args = ap.parse_args()
    merge_source(args.source, args.config, args.inspect, args.limit, args.dry_run)


if __name__ == "__main__":
    main()
