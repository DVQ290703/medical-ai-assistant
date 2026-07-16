"""Dev: kiểm CẢ CHUỖI RAG trước khi demo. In [OK]/[FAIL] từng thành phần.

Chạy TRƯỚC buổi pitch — xanh hết mới yên tâm.
  python scripts/dev/healthcheck.py
"""
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OK, FAIL = "[  OK  ]", "[ FAIL ]"
results = []


def check(name, fn):
    try:
        detail = fn()
        results.append((True, name, detail))
    except Exception as e:
        results.append((False, name, str(e)[:120]))


# 1. Qdrant
def _qdrant():
    from src.knowledge.retriever import config_from_yaml
    from qdrant_client import QdrantClient
    c = config_from_yaml()
    cl = QdrantClient(host=c.qdrant_host, port=c.qdrant_port)
    cols = cl.get_collections().collections
    parts = []
    for col in cols:
        n = cl.get_collection(col.name).points_count
        parts.append(f"{col.name}={n:,}")
    return f"{c.qdrant_host}:{c.qdrant_port} | " + ", ".join(parts)


# 2. Model server (chính + backup) /health
def _model_server():
    import requests
    from src.knowledge.retriever import config_from_yaml
    c = config_from_yaml()
    urls = [u for u in [c.remote_url, c.remote_url_backup] if u]
    if not urls:
        raise RuntimeError("chưa set RAG_REMOTE_URL")
    alive = []
    for u in urls:
        try:
            r = requests.get(u.rstrip("/") + "/health", timeout=10)
            alive.append(f"{u}={'up' if r.ok else r.status_code}")
        except Exception:
            alive.append(f"{u}=down")
    if not any("=up" in a for a in alive):
        raise RuntimeError("không endpoint nào up: " + ", ".join(alive))
    return ", ".join(alive)


# 3. Groq key + gọi thử
def _groq():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("thiếu GROQ_API_KEY")
    import requests
    from src.generation.engine import gen_config_from_yaml
    cfg = gen_config_from_yaml()
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                      json={"model": cfg.model, "messages": [{"role": "user", "content": "ping"}],
                            "max_tokens": 1},
                      headers={"Authorization": f"Bearer {key}"}, timeout=20)
    if r.status_code == 429:
        return f"key OK nhưng RATE-LIMITED (model={cfg.model})"
    r.raise_for_status()
    return f"key OK, model={cfg.model}"


print("=== HEALTH CHECK RAG (chạy trước demo) ===\n")
check("Qdrant (vector DB)", _qdrant)
check("Model server (encode/rerank)", _model_server)
check("Groq (generation)", _groq)

all_ok = True
for ok, name, detail in results:
    mark = OK if ok else FAIL
    if not ok:
        all_ok = False
    print(f"{mark} {name}\n         {detail}")

print("\n" + ("✅ SẴN SÀNG DEMO" if all_ok else "❌ CÓ THÀNH PHẦN LỖI — sửa trước khi pitch"))
sys.exit(0 if all_ok else 1)
