"""Knowledge plane — Ingest corpus RAG (vinmec-medical-qa, gated) -> JSONL chuẩn hoá.

Bước ĐẦU TIÊN của tầng knowledge, chạy TRƯỚC chunk -> embed. Khác với
src/data/ingest.py (đó là seed để TRAIN); đây là corpus tri thức để RETRIEVE.

Mục tiêu:
  1. Xác thực HF token (xử lý gotcha: stale HF_TOKEN override login())
  2. Load dataset gated theo configs/rag.yaml -> corpus
  3. Inspect schema thật — xác định field nào là question / answer
  4. Chuẩn hoá -> data/raw/*.jsonl (field: question, answer)

Yêu cầu:
  - Đã bấm "Agree and access repository" trên trang dataset HF.
  - Có HF_TOKEN (Settings > Access Tokens, quyền read). ĐỪNG hardcode vào file.
  - Chạy nơi có internet (Kaggle). `pip install datasets huggingface_hub pyyaml`.

Usage:
    # Chỉ xem schema thật (chưa ghi file) — chạy lần đầu để biết field:
    python -m src.knowledge.ingest --inspect

    # Sau khi điền field_map trong configs/rag.yaml -> ghi JSONL:
    python -m src.knowledge.ingest --out data/raw/vinmec.jsonl

    # Probe nhỏ:
    python -m src.knowledge.ingest --inspect --limit 30
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

# Nạp .env nếu chạy local (HF_TOKEN...). Trên Kaggle không có .env -> bỏ qua.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def authenticate(token: str | None = None) -> str:
    """
    Xác thực HF. Gotcha: stale HF_TOKEN trong env có thể âm thầm override
    login() -> ưu tiên token truyền vào, cảnh báo nếu env giữ token khác.
    """
    from huggingface_hub import login, whoami

    env_token = os.environ.get("HF_TOKEN")
    if token is None:
        token = env_token
        if token is None:
            raise ValueError(
                "Chưa có token. Set HF_TOKEN hoặc truyền token=... "
                "Lấy tại https://huggingface.co/settings/tokens"
            )
    elif env_token and env_token != token:
        print("[warn] HF_TOKEN trong env khác token truyền vào -> "
              "xoá env token để tránh override.")
        del os.environ["HF_TOKEN"]

    login(token=token)
    who = whoami()
    print(f"[ok] Đăng nhập HF với user: {who.get('name', '?')}")
    return token


def load_hf_dataset(source: str, split: str, token: str | None):
    """Load dataset gated. Chưa được duyệt -> HF trả 401/403 -> báo lỗi rõ."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "Thiếu `datasets`. Cài: pip install datasets huggingface_hub. "
            "Bước này cần chạy nơi có internet (Kaggle)."
        ) from e

    try:
        ds = load_dataset(source, split=split, token=token)
    except Exception as e:
        msg = str(e).lower()
        if "401" in msg or "403" in msg or "gated" in msg:
            raise PermissionError(
                f"Chưa có quyền truy cập {source}. Vào trang dataset trên HF, "
                "bấm 'Agree and access repository', rồi thử lại."
            ) from e
        raise
    print(f"[ok] Loaded {len(ds):,} rows từ {source}, split='{split}'")
    return ds


def inspect(ds, n: int = 3) -> None:
    """In schema + vài mẫu để xác định field question/answer thật."""
    print("\n=== COLUMNS ===")
    print(ds.column_names)

    print("\n=== FEATURES ===")
    for k, v in ds.features.items():
        print(f"  {k}: {v}")

    print(f"\n=== {n} MẪU ĐẦU ===")
    for i in range(min(n, len(ds))):
        for k, v in ds[i].items():
            text = str(v).replace("\n", " ")
            preview = text[:200] + ("..." if len(text) > 200 else "")
            print(f"  [{k}] ({len(str(v))} chars) {preview}")

    print("\n=== NULL / RỖNG (toàn tập) ===")
    for col in ds.column_names:
        empties = sum(
            1 for x in ds[col]
            if x is None or (isinstance(x, str) and not x.strip())
        )
        print(f"  {col}: {empties:,} rỗng / {len(ds):,}")


def normalize_and_write(ds, field_map: dict, out_path: str) -> int:
    """Đổi tên field nguồn -> {question, answer}; bỏ mẫu rỗng; ghi JSONL."""
    q_key = field_map["question"]
    a_key = field_map["answer"]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out.open("w", encoding="utf-8") as f:
        for rec in ds:
            question = (rec.get(q_key) or "").strip()
            answer = (rec.get(a_key) or "").strip()
            if not question or not answer:
                continue
            f.write(json.dumps(
                {"question": question, "answer": answer}, ensure_ascii=False
            ) + "\n")
            n += 1

    print(f"[ingest] {n} mẫu -> {out}")
    return n


def run(config_path: str, corpus_id: str | None, out_override: str | None,
        do_inspect: bool, limit: int | None, token: str | None) -> None:
    cfg = load_config(config_path)
    corpora = cfg.get("corpus") or []
    if not corpora:
        raise SystemExit(f"Không thấy khối `corpus` trong {config_path}.")

    # Chọn corpus: theo --corpus id, hoặc phần tử đầu tiên.
    if corpus_id:
        entry = next((c for c in corpora if c.get("id") == corpus_id), None)
        if entry is None:
            raise SystemExit(f"Không thấy corpus id='{corpus_id}' trong {config_path}.")
    else:
        entry = corpora[0]

    token = authenticate(token)
    ds = load_hf_dataset(entry["source"], entry.get("split", "train"), token)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    if do_inspect:
        inspect(ds)

    field_map = entry.get("field_map")
    if not field_map:
        print("\n[note] field_map=null trong config. Xem schema ở trên, điền "
              "field_map (vd { question: question, answer: answer }) vào "
              f"corpus '{entry.get('id')}' trong {config_path}, rồi chạy lại "
              "(bỏ --inspect) để ghi JSONL.")
        return

    out_path = out_override or entry["out"]
    normalize_and_write(ds, field_map, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest corpus RAG (gated HF) -> JSONL")
    ap.add_argument("--config", default="configs/rag.yaml")
    ap.add_argument("--corpus", default=None, help="corpus id trong rag.yaml (mặc định: phần tử đầu)")
    ap.add_argument("--out", default=None, help="ghi đè đường dẫn output")
    ap.add_argument("--inspect", action="store_true", help="in schema + mẫu")
    ap.add_argument("--limit", type=int, default=None, help="chỉ lấy N mẫu đầu (probe)")
    args = ap.parse_args()

    run(
        config_path=args.config,
        corpus_id=args.corpus,
        out_override=args.out,
        do_inspect=args.inspect,
        limit=args.limit,
        token=os.environ.get("HF_TOKEN"),
    )


if __name__ == "__main__":
    main()
