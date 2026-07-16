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

import re
from dataclasses import dataclass, field

from src.serving.guards.output_guard import check_output
from src.serving.policy.engine import decide as policy_decide
from src.serving.policy.disclaimer import disclaimer, has_disclaimer
from src.prompting.template import load_system_prompt, load_fewshot
from src.prompting.builder import build_turns
from src.serving.citation import build_sources, strip_llm_sources
from src.generation.engine import engine_from_config, gen_config_from_yaml
from src.monitoring import observability as obs


# Lưới an toàn: bot được HỎI LẠI tối đa ngần này lần rồi buộc trả lời (tránh hỏi vô tận).
MAX_CLARIFY = 3

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
    kind: str = "normal"        # normal | emergency | no_info | refuse | degraded | clarify
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


def answer(query: str, history: list | None = None,
           citation_required: bool = True) -> Answer:
    """history: list {role: 'user'|'assistant', content} các lượt TRƯỚC (không gồm query)."""
    with obs.trace("rag-query", input=query) as tr:
        result = _answer(query, history or [], citation_required)
        tr.update(output=result.text, metadata={"kind": result.kind,
                                                 "n_sources": len(result.sources),
                                                 "n_history": len(history or [])})
        result.trace_id = getattr(tr, "trace_id", None) or ""   # để gắn feedback sau
    obs.flush()
    return result


def _answer(query: str, history: list, citation_required: bool) -> Answer:
    # 1. POLICY: quyết định hành động TRƯỚC retrieve/LLM
    #    escalate (cấp cứu -> 115) | refuse (ngoài phạm vi) | answer
    decision = policy_decide(query, in_conversation=bool(history))
    if decision.action == "escalate":
        return Answer(text=decision.message, kind="emergency")
    if decision.action == "refuse":
        return Answer(text=decision.message, kind="refuse")

    _lazy_init()

    # 2. Retrieve (đã gồm hybrid + rerank + threshold + source priority)
    #    Ghép triệu chứng đã kể ở lượt trước vào query -> tìm đúng khi câu hiện tại ngắn/
    #    tham chiếu ngầm ("cách chữa thì sao?"). Chỉ lấy lượt USER (bỏ câu bot).
    #    Model server (Colab) chết -> graceful degrade (KHÔNG traceback, KHÔNG sập demo).
    from src.knowledge.remote_client import RemoteUnavailable
    ctx = " ".join(h["content"] for h in history[-6:]
                   if h.get("role") == "user" and h.get("content"))
    try:
        with obs.span("retrieve") as sp:
            hits = _retriever.retrieve(query, context=ctx)
            sp.update(output=f"{len(hits)} hits",
                      metadata={"sources": [getattr(h, "source", "") for h in hits]})
    except RemoteUnavailable as e:
        print(f"[degraded] model server không phản hồi: {e}")
        return Answer(text=DEGRADED_MSG, kind="degraded")
    if not hits:
        return Answer(text=NO_INFO_MSG, kind="no_info")

    # 2b. INTENT GATE (ràng buộc CỨNG): nếu người dùng mô tả triệu chứng để tìm bệnh mà
    #     chưa đủ chi tiết -> buộc HỎI LẠI do CODE kiểm soát (model không có cơ hội phỏng
    #     đoán bệnh). Chỉ gác khi CHƯA hỏi lại lần nào (gate tự kiểm tra history).
    from src.prompting.intent_gate import clarification_question
    with obs.span("intent-gate") as sp:
        gate_q = clarification_question(query, history, _engine)
        sp.update(output=gate_q or "(cho qua)")
    if gate_q:
        return Answer(text=gate_q, kind="clarify")

    # 3. Build prompt (gồm lịch sử hội thoại) + generate
    #    Lưới an toàn: nếu bot đã HỎI LẠI >= MAX_CLARIFY lần -> buộc trả lời (không hỏi vô tận).
    system = load_system_prompt(_gen_cfg.system_prompt_path)
    n_clarify = sum(1 for h in history
                    if h.get("role") == "assistant" and "[HỎI LẠI]" in (h.get("content") or ""))
    turns = build_turns(query, hits, history, force_answer=n_clarify >= MAX_CLARIFY,
                        fewshot=load_fewshot())
    with obs.span("generate", as_type="generation", model=_gen_cfg.model) as sp:
        text = _engine.generate_messages(system, turns)
        sp.update(output=text)

    # 3a. LLM đôi khi TỰ viết khối "Nguồn tham khảo:" (vì thấy [1][2] trong context) ->
    #     cắt đi để không in 2 lần (nguồn thật đính ở bước cuối bằng format_sources).
    text = strip_llm_sources(text)

    # 3b. CLARIFY: LLM hỏi lại để làm rõ (câu hỏi mơ hồ) -> trả thẳng câu hỏi ngược,
    #     KHÔNG ép citation/disclaimer/sources (đây là câu hỏi, chưa phải câu trả lời).
    #     Nhận diện marker dù nằm ở đầu hay giữa; rồi XOÁ marker + mọi [số] trích dẫn
    #     (câu hỏi lại không có nguồn -> [1] lọt vào trông vô nghĩa).
    if "[HỎI LẠI]" in text:
        clarify = text.replace("[HỎI LẠI]", " ")
        clarify = re.sub(r"\[\d+\]", "", clarify)      # bỏ [1] [2]... LLM lỡ chèn
        clarify = re.sub(r"[ \t]+", " ", clarify).strip()
        return Answer(text=clarify, kind="clarify")

    # 4. Citation
    sources = build_sources(text, hits)
    if citation_required and not sources:
        # LLM không trích [số] nào -> vẫn đính nguồn đã dùng để minh bạch
        sources = [{"n": i + 1, "title": getattr(h, "title", ""),
                    "url": getattr(h, "url", ""), "source": getattr(h, "source", "")}
                   for i, h in enumerate(hits)]
        text += "\n\n(Lưu ý: câu trả lời tổng hợp từ các nguồn tham khảo kèm theo.)"

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

    # Nguồn KHÔNG đính vào text -> client (web) render link bấm được từ .sources.
    # (format_sources vẫn dùng được cho client text-only/CLI nếu cần.)
    return Answer(text=text, sources=sources, warnings=guard.warnings)
