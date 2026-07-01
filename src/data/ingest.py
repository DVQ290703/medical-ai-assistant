"""Phase 1 (VN) — Ingest seed dataset (medical-o1-reasoning-SFT, EN) -> JSONL chuẩn hoá.

Chạy trên Kaggle (cần internet để tải HuggingFace). Output là seed EN, chưa dịch —
bước dịch nằm ở translate.py.

Usage:
    python -m src.data.ingest --config configs/data.yaml --out data/raw/seed_en.jsonl
    python -m src.data.ingest --config configs/data.yaml --limit 30   # probe nhỏ
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


# medical-o1 fields -> tên chuẩn nội bộ
FIELD_MAP = {"Question": "question", "Complex_CoT": "cot", "Response": "response"}


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_record(rec: dict) -> dict:
    """Đổi tên field o1 -> chuẩn nội bộ; giữ nguyên nội dung."""
    out = {}
    for src_key, dst_key in FIELD_MAP.items():
        out[dst_key] = (rec.get(src_key) or "").strip()
    return out


def ingest(config_path: str, out_path: str, limit: int | None = None) -> int:
    cfg = load_config(config_path)
    seed = cfg["seed_dataset"]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Import trong hàm để môi trường không có HF vẫn import được module này.
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "Thiếu `datasets`. Cài: pip install datasets. "
            "Bước này cần chạy nơi có internet (Kaggle)."
        ) from e

    ds = load_dataset(seed["source"], seed.get("config", "en"), split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    n = 0
    with out.open("w", encoding="utf-8") as f:
        for rec in ds:
            norm = normalize_record(rec)
            if not norm["question"] or not norm["response"]:
                continue  # bỏ mẫu rỗng (validate.py sẽ kiểm kỹ hơn)
            f.write(json.dumps(norm, ensure_ascii=False) + "\n")
            n += 1

    print(f"[ingest] {n} mẫu -> {out}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--out", default="data/raw/seed_en.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    ingest(args.config, args.out, args.limit)


if __name__ == "__main__":
    main()