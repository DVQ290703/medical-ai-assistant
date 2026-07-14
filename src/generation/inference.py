"""Generation — Hàm answer() cấp cao: nối RAG end-to-end.

Luồng:
  query
   -> emergency_check (LUẬT CỨNG: red flag -> cảnh báo 115, KHÔNG gọi LLM)
   -> Retriever.retrieve() -> hits
   -> rỗng (dưới threshold) -> "không đủ thông tin, hãy đi khám" (KHÔNG bịa)
   -> build_user_prompt (context + [số]) + system prompt
   -> engine.generate() (Groq/local)
   -> build_sources ([số] LLM trích -> nguồn) + append danh sách nguồn

CLI: python -m src.generation.inference "câu hỏi"
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.serving.guards.input_guard import emergency_check
from src.prompting.template import load_system_prompt
from src.prompting.builder import build_user_prompt
from src.serving.citation import build_sources, format_sources
from src.generation.engine import engine_from_config, gen_config_from_yaml


NO_INFO_MSG = (
    "Thông tin hiện có chưa đủ để trả lời câu hỏi này một cách đáng tin cậy. "
    "Bạn nên đến khám bác sĩ để được tư vấn chính xác."
)


@dataclass
class Answer:
    text: str
    sources: list = field(default_factory=list)
    kind: str = "normal"        # normal | emergency | no_info


# giữ retriever/engine tái dùng giữa các lần gọi (đỡ load lại model)
_retriever = None
_engine = None
_gen_cfg = None


def _lazy_init():
    global _retriever, _engine, _gen_cfg
    if _retriever is None:
        from src.knowledge.retriever import Retriever
        _retriever = Retriever()
    if _engine is None:
        _gen_cfg = gen_config_from_yaml()
        _engine = engine_from_config(_gen_cfg)


def answer(query: str, citation_required: bool = True) -> Answer:
    # 1. CẤP CỨU: chặn cứng trước mọi thứ
    emg = emergency_check(query)
    if emg:
        return Answer(text=emg, kind="emergency")

    _lazy_init()

    # 2. Retrieve
    hits = _retriever.retrieve(query)
    if not hits:
        return Answer(text=NO_INFO_MSG, kind="no_info")

    # 3. Build prompt + generate
    system = load_system_prompt(_gen_cfg.system_prompt_path)
    user = build_user_prompt(query, hits)
    text = _engine.generate(system, user)

    # 4. Citation
    sources = build_sources(text, hits)
    if citation_required and not sources:
        # LLM không trích [số] nào -> vẫn đính nguồn đã dùng (hits) để minh bạch
        sources = [{"n": i + 1, "title": getattr(h, "title", ""),
                    "url": getattr(h, "url", ""), "source": getattr(h, "source", "")}
                   for i, h in enumerate(hits)]
        text += "\n\n(Lưu ý: câu trả lời tổng hợp từ các nguồn dưới đây.)"
    return Answer(text=text + format_sources(sources), sources=sources)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="RAG y tế — hỏi 1 câu, in câu trả lời + nguồn")
    ap.add_argument("query")
    args = ap.parse_args()
    a = answer(args.query)
    print(f"\n=== [{a.kind}] ===\n{a.text}")


if __name__ == "__main__":
    main()
