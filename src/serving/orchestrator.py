"""Serving — Điều phối request RAG (lớp điều phối chính).

Luồng: input_guard (emergency) -> retrieve -> [rerank trong retriever] -> prompt ->
        generate -> citation.

Đây là nơi ghép các tầng. inference.py + app.py gọi answer() ở đây. Các tầng chưa dùng
(output_guard hallucination-check, policy) sẽ cắm vào đây khi làm.

  query
   -> emergency_check (LUẬT CỨNG: red flag -> cảnh báo 115, KHÔNG gọi LLM)
   -> Retriever.retrieve() -> hits  (rỗng -> "không đủ thông tin", KHÔNG bịa)
   -> build_user_prompt (context + [số]) + system prompt
   -> engine.generate() (Groq/local)
   -> build_sources ([số] LLM trích -> nguồn) + append danh sách nguồn
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.serving.guards.output_guard import check_output
from src.serving.policy.engine import decide as policy_decide
from src.serving.policy.disclaimer import disclaimer, has_disclaimer
from src.prompting.template import load_system_prompt
from src.prompting.builder import build_user_prompt
from src.serving.citation import build_sources, format_sources
from src.generation.engine import engine_from_config, gen_config_from_yaml
from src.monitoring import observability as obs


NO_INFO_MSG = (
    "Thông tin hiện có chưa đủ để trả lời câu hỏi này một cách đáng tin cậy. "
    "Bạn nên đến khám bác sĩ để được tư vấn chính xác."
)
DEGRADED_MSG = (
    "Hệ thống đang bận, vui lòng thử lại sau ít phút. "
    "Nếu đây là trường hợp khẩn cấp, hãy gọi 115 hoặc đến cơ sở y tế gần nhất."
)


@dataclass
class Answer:
    text: str
    sources: list = field(default_factory=list)
    kind: str = "normal"        # normal | emergency | no_info | refuse | degraded
    warnings: list = field(default_factory=list)   # cờ từ output_guard (no_citation, PII...)
    trace_id: str = ""          # id trace Langfuse (để gắn feedback 👍/👎 sau); "" nếu tắt


# giữ retriever/engine tái dùng giữa các lần gọi (đỡ load lại)
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


def warm_up():
    """Khởi tạo trước retriever + engine (dùng lúc app startup cho nhanh)."""
    _lazy_init()


def answer(query: str, citation_required: bool = True) -> Answer:
    with obs.trace("rag-query", input=query) as tr:
        result = _answer(query, citation_required)
        tr.update(output=result.text, metadata={"kind": result.kind,
                                                 "n_sources": len(result.sources)})
        result.trace_id = getattr(tr, "trace_id", None) or ""   # để gắn feedback sau
    obs.flush()
    return result


def _answer(query: str, citation_required: bool) -> Answer:
    # 1. POLICY: quyết định hành động TRƯỚC retrieve/LLM
    #    escalate (cấp cứu -> 115) | refuse (ngoài phạm vi) | answer
    decision = policy_decide(query)
    if decision.action == "escalate":
        return Answer(text=decision.message, kind="emergency")
    if decision.action == "refuse":
        return Answer(text=decision.message, kind="refuse")

    _lazy_init()

    # 2. Retrieve (đã gồm hybrid + rerank + threshold + source priority)
    #    Model server (Colab) chết -> graceful degrade (KHÔNG traceback, KHÔNG sập demo).
    from src.knowledge.remote_client import RemoteUnavailable
    try:
        with obs.span("retrieve") as sp:
            hits = _retriever.retrieve(query)
            sp.update(output=f"{len(hits)} hits",
                      metadata={"sources": [getattr(h, "source", "") for h in hits]})
    except RemoteUnavailable as e:
        print(f"[degraded] model server không phản hồi: {e}")
        return Answer(text=DEGRADED_MSG, kind="degraded")
    if not hits:
        return Answer(text=NO_INFO_MSG, kind="no_info")

    # 3. Build prompt + generate
    system = load_system_prompt(_gen_cfg.system_prompt_path)
    user = build_user_prompt(query, hits)
    with obs.span("generate", as_type="generation", model=_gen_cfg.model) as sp:
        text = _engine.generate(system, user)
        sp.update(output=text)

    # 4. Citation
    sources = build_sources(text, hits)
    if citation_required and not sources:
        # LLM không trích [số] nào -> vẫn đính nguồn đã dùng để minh bạch
        sources = [{"n": i + 1, "title": getattr(h, "title", ""),
                    "url": getattr(h, "url", ""), "source": getattr(h, "source", "")}
                   for i, h in enumerate(hits)]
        text += "\n\n(Lưu ý: câu trả lời tổng hợp từ các nguồn dưới đây.)"

    # 5. Output guard: che PII + kiểm citation/grounding (trên text LLM, trước khi ghép nguồn)
    guard = check_output(text, kind="normal", citation_required=citation_required,
                         has_sources=bool(sources))
    text = guard.text
    if guard.pii_redacted:
        print(f"[guard] đã che {guard.pii_redacted} PII trong câu trả lời.")
    if guard.warnings:
        print(f"[guard] cảnh báo: {guard.warnings}")

    # 6. Policy: ÉP disclaimer y tế (nếu LLM chưa tự nhắc) — đảm bảo 100% có ở tầng app
    if not has_disclaimer(text):
        text += disclaimer(need_doctor=decision.need_doctor)

    return Answer(text=text + format_sources(sources), sources=sources,
                  warnings=guard.warnings)
