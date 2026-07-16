"""Phase 3.3 — Retrieval eval: Recall@k, MRR trên golden set tiếng Việt.

Golden set: evaluation_sets/retrieval/v1/goldset.jsonl {query, gold_doc_ids}.
Sinh bằng scripts/dev/make_retrieval_goldset.py (self-retrieval sanity-check).

Với mỗi query: Retriever.retrieve -> so hit.chunk_id với gold_doc_ids:
  - Recall@k: gold có trong top-k không (1/0), trung bình toàn set.
  - MRR: 1/(thứ hạng gold đầu tiên); 0 nếu không thấy.

CẦN Qdrant + model server (remote encode/rerank). Usage: python -m src.evaluation.retrieval
"""
from __future__ import annotations

import json
from pathlib import Path

GOLDSET = "evaluation_sets/retrieval/v1/goldset.jsonl"


def _load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def _hit_ids(hits) -> list[str]:
    """id mỗi hit để so gold (chunk_id, với Q&A = str(doc_id))."""
    return [str(getattr(h, "chunk_id", "")) for h in hits]


def evaluate(goldset_path: str = GOLDSET, top_k: int = 8) -> dict:
    gold = _load(goldset_path)
    if not gold:
        return {"error": f"goldset rỗng ({goldset_path}). Chạy make_retrieval_goldset.py trước."}

    from src.knowledge.retriever import Retriever
    r = Retriever()

    recall_hits = 0
    rr_sum = 0.0
    n = 0
    for case in gold:
        gold_ids = {str(x) for x in case["gold_doc_ids"]}
        ids = _hit_ids(r.retrieve(case["query"]))[:top_k]
        if gold_ids & set(ids):
            recall_hits += 1
        for rank, hid in enumerate(ids, 1):
            if hid in gold_ids:
                rr_sum += 1.0 / rank
                break
        n += 1
        if n % 20 == 0:
            print(f"  ...{n}/{len(gold)}")

    return {"n": n, "top_k": top_k,
            "recall_at_k": recall_hits / n if n else None,
            "mrr": rr_sum / n if n else None}


def run() -> dict:
    res = evaluate()
    print("=== RETRIEVAL EVAL ===")
    if "error" in res:
        print("  ", res["error"])
        return res
    print(f"  n={res['n']} | Recall@{res['top_k']}={res['recall_at_k']:.1%} | MRR={res['mrr']:.3f}")
    print("  (self-retrieval sanity-check — không phải benchmark human-labeled.)")
    return res


if __name__ == "__main__":
    run()
