"""Serving — FastAPI app: /chat (RAG), /health, serve web tĩnh. Rate-limited.

Luồng: trình duyệt (web/) -> POST /chat -> inference.answer() -> {answer, sources, kind}.
Rate-limit in-memory theo IP (per_minute/per_day từ configs/serving.yaml) — đủ demo.

Chạy: python -m src.serving.app  -> http://localhost:8000
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.serving.rate_limiter import RateLimiter

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _load_serving_cfg(path: str = "configs/serving.yaml") -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


CFG = _load_serving_cfg()
_RL = CFG.get("rate_limit", {}) or {}
PER_MIN = _RL.get("per_minute", 30)
PER_DAY = _RL.get("per_day", 500)

app = FastAPI(title="Medical RAG (VN)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_limiter = RateLimiter(PER_MIN, PER_DAY)


def _rate_limit(ip: str):
    ok, reason = _limiter.check(ip)
    if not ok:
        raise HTTPException(429, reason)


class Turn(BaseModel):
    role: str            # "user" | "assistant"
    content: str


class ChatReq(BaseModel):
    message: str
    history: list[Turn] = []    # các lượt TRƯỚC (client giữ) — server stateless


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
def chat(req: ChatReq, request: Request):
    _rate_limit(request.client.host if request.client else "unknown")
    q = (req.message or "").strip()
    if not q:
        raise HTTPException(400, "message rỗng.")
    from src.serving.orchestrator import answer
    history = [{"role": t.role, "content": t.content} for t in req.history]
    try:
        a = answer(q, history=history)
    except SystemExit as e:      # lỗi hạ tầng (Qdrant/Colab/Groq) -> 503 rõ ràng
        raise HTTPException(503, f"Dịch vụ chưa sẵn sàng: {e}")
    except Exception as e:
        raise HTTPException(500, f"Lỗi xử lý: {e}")
    # trace_id -> client giữ để gửi kèm feedback 👍/👎. Không lộ metrics kỹ thuật cho user.
    return {"answer": a.text, "sources": a.sources, "kind": a.kind,
            "warnings": a.warnings, "trace_id": a.trace_id,
            "query": q}    # query để lưu kèm feedback (backup queue)


class FeedbackReq(BaseModel):
    trace_id: str = ""
    query: str = ""
    rating: str          # "up" | "down"
    comment: str = ""


@app.post("/feedback")
def feedback(req: FeedbackReq, request: Request):
    _rate_limit(request.client.host if request.client else "unknown")
    if req.rating not in ("up", "down"):
        raise HTTPException(400, "rating phải là 'up' hoặc 'down'.")
    import datetime
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    # 1. lưu local queue (chờ human review — chống feedback poisoning)
    from src.monitoring.feedback_queue import add_feedback
    add_feedback(req.trace_id, req.query, req.rating, req.comment, ts)
    # 2. gửi score lên Langfuse (nếu bật) — gắn vào trace
    from src.monitoring import observability as obs
    obs.score(req.trace_id, name="user_feedback",
              value=1.0 if req.rating == "up" else 0.0, comment=req.comment)
    return {"ok": True}


# serve web tĩnh tại "/" (đặt SAU các route API)
_web = Path("web")
if _web.is_dir():
    app.mount("/", StaticFiles(directory=str(_web), html=True), name="web")


def main() -> None:
    import uvicorn
    host = CFG.get("host", "0.0.0.0")
    port = CFG.get("port", 8000)
    print(f"[serve] http://localhost:{port}  (rate {PER_MIN}/phút, {PER_DAY}/ngày)")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
