"""Generation — API cấp cao + CLI. Logic điều phối nằm ở src/serving/orchestrator.py.

Giữ lại để tương thích (import answer/Answer từ đây vẫn chạy) + CLI test nhanh:
  python -m src.generation.inference "câu hỏi"
"""
from __future__ import annotations

from src.serving.orchestrator import answer, Answer, NO_INFO_MSG  # noqa: F401 (re-export)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="RAG y tế — hỏi 1 câu, in câu trả lời + nguồn")
    ap.add_argument("query")
    args = ap.parse_args()
    a = answer(args.query)
    print(f"\n=== [{a.kind}] ===\n{a.text}")


if __name__ == "__main__":
    main()
