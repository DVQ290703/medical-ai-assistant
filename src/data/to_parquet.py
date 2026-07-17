"""Gộp data đã validate/regate -> messages format -> Parquet. Tách MCQ sang eval."""
import json, argparse
import pandas as pd
from .mcq import is_mcq

SYSTEM = ("Bạn là trợ lý y khoa. Hãy suy luận từng bước trước khi kết luận, "
          "và trả lời bằng tiếng Việt.")

def to_messages(r):
    cot  = (r.get("cot") or "").strip()
    resp = (r.get("response") or "").strip()
    assistant = f"<think>{cot}</think>{resp}" if cot else resp
    return [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": (r.get("question") or "").strip()},
        {"role": "assistant", "content": assistant},
    ]

def build(in_jsonl, out_parquet, mcq_out="evaluation_sets/eval_mcq.jsonl"):
    from pathlib import Path
    Path(mcq_out).parent.mkdir(parents=True, exist_ok=True)

    rows, mcq_rows, skipped = [], [], 0
    for line in open(in_jsonl, encoding="utf-8"):
        r = json.loads(line)
        if not (r.get("question") or "").strip() or not (r.get("response") or "").strip():
            skipped += 1; continue
        if is_mcq(r):                      # <-- TÁCH MCQ, không cho vào train
            mcq_rows.append(r)
            continue
        rows.append({"messages": to_messages(r)})

    df = pd.DataFrame(rows)
    df.to_parquet(out_parquet, index=False)

    with open(mcq_out, "w", encoding="utf-8") as f:
        for r in mcq_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"train rows: {len(df)} | MCQ tách ra: {len(mcq_rows)} | skipped: {skipped}")
    return len(df), len(mcq_rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_jsonl", default="seed_vi.regated.jsonl")  # <-- đổi default
    ap.add_argument("--out", default="train_vi.parquet")
    ap.add_argument("--mcq-out", default="evaluation_sets/eval_mcq.jsonl")
    a = ap.parse_args()
    build(a.in_jsonl, a.out, a.mcq_out)

if __name__ == "__main__":
    main()