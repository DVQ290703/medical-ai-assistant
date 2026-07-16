"""Phase 3 — Harness: chạy các eval BẬT được -> report reports/eval_report.md.

Chạy safety (luôn được, không cần Qdrant/LLM) + retrieval (nếu có goldset + Qdrant + model
server). Eval thiếu điều kiện -> ghi "skipped" (không crash). Không bịa số cho eval chưa
có data (VM14K/MedQA -> để roadmap).

Usage:
  python -m src.evaluation.harness              # chạy tất cả bật được
  python -m src.evaluation.harness --safety-only # chỉ safety (nhanh, offline)
"""
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

REPORT = "reports/eval_report.md"


def run(safety_only: bool = False, timestamp: str = "") -> str:
    lines = ["# Báo cáo Evaluation — Medical RAG (VN)", ""]
    if timestamp:
        lines.append(f"_Sinh lúc: {timestamp}_\n")
    lines.append("> Số liệu MVP. Set nhỏ, tự sinh -> ĐỊNH HƯỚNG, không phải benchmark chuẩn.\n")

    # --- Safety (luôn chạy được) ---
    from src.evaluation.safety import run as run_safety
    print("\n>>> Safety eval...")
    s = run_safety()
    e, p, o = s["emergency"], s["pii"], s["out_of_scope"]
    lines += [
        "## An toàn (Safety)", "",
        f"- **Emergency routing** (n={e['n_emergency']} cấp cứu + {e['n_normal']} thường): "
        f"recall **{e['recall']:.0%}**, false-positive **{e['false_positive_rate']:.0%}**.",
        f"- **PII redaction** (n={p['n_pii']}): che **{p['redaction_rate']:.0%}**, "
        f"che nhầm nội dung y khoa: **{p['false_redaction']}**.",
        f"- **Out-of-scope refuse**: đúng {o['refuse_correct']}/{o['n_out']}, "
        f"nhầm câu y tế: {o['refuse_medical_wrong']}.", "",
    ]

    # --- Retrieval (cần goldset + Qdrant + model server) ---
    lines.append("## Retrieval")
    if safety_only:
        lines.append("- _Bỏ qua (--safety-only)._")
    else:
        print("\n>>> Retrieval eval...")
        try:
            from src.evaluation.retrieval import evaluate
            r = evaluate()
            if "error" in r:
                lines.append(f"- _Skipped: {r['error']}_")
            else:
                lines.append(f"- n={r['n']} | **Recall@{r['top_k']} = {r['recall_at_k']:.1%}** | "
                             f"**MRR = {r['mrr']:.3f}** (self-retrieval sanity-check).")
        except Exception as ex:
            lines.append(f"- _Skipped (lỗi hạ tầng: {ex})._")
    lines.append("")

    # --- Chưa làm (trung thực) ---
    lines += [
        "## Chưa đo (roadmap)",
        "- Accuracy MCQ (VM14K/MedQA) — cần golden set human-labeled.",
        "- Quality RAGAS (faithfulness/answer-relevance) — cần LLM-judge.",
        "- Drift monitoring theo thời gian.", "",
    ]

    report = "\n".join(lines)
    out = Path(REPORT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\n[harness] report -> {REPORT}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Chạy evaluation harness -> report md")
    ap.add_argument("--safety-only", action="store_true")
    args = ap.parse_args()
    # timestamp truyền vào (Date trong sandbox bị chặn -> để None khi chạy thật CLI dùng now)
    import datetime
    run(args.safety_only, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))


if __name__ == "__main__":
    main()
