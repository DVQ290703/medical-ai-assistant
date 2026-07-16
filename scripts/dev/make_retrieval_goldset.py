"""Dev: sinh retrieval golden set từ vinmec.jsonl (self-retrieval sanity-check).

Mỗi câu Q&A -> {query: question, gold_doc_ids: [doc_id]}. doc_id khớp payload trong
collection vinmec_qa (embedding.py gán doc_id = line index). Kiểm: hỏi lại câu hỏi có
tìm về đúng cặp Q&A gốc không (Recall@k, MRR).

LƯU Ý: đây là sanity-check (query = chính câu trong corpus -> dễ) — KHÔNG phải benchmark
khó. Benchmark thật cần golden set human-labeled (VM14K). Ghi rõ trong report.

Usage: python scripts/dev/make_retrieval_goldset.py [--n 100]
"""
import sys, io, json, argparse, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CORPUS = "data/raw/vinmec.jsonl"
OUT = "evaluation_sets/retrieval/v1/goldset.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="số query lấy mẫu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    with open(CORPUS, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            q = (o.get("question") or "").strip()
            if q:
                rows.append((i, q))     # doc_id = line index (khớp embedding.py)

    random.seed(args.seed)
    sample = random.sample(rows, min(args.n, len(rows)))

    from pathlib import Path
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for doc_id, q in sample:
            f.write(json.dumps({"query": q, "gold_doc_ids": [doc_id],
                                "collection": "vinmec_qa"}, ensure_ascii=False) + "\n")
    print(f"[goldset] {len(sample)} query -> {OUT}")


if __name__ == "__main__":
    main()
