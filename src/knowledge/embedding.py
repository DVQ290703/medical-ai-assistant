"""Knowledge plane — Embed corpus vinmec bằng BGE-M3 (hybrid dense+sparse) -> Qdrant.

Tạo HAI collection để so recall:
  - vinmec_q     : embed chỉ QUESTION  (khớp phân bố query bệnh nhân)
  - vinmec_qa    : embed QUESTION + ANSWER  (nhiều ngữ cảnh hơn)

Cả hai collection payload GIỐNG NHAU (question, answer, doc_id) — chỉ khác cái
được embed thành vector. So recall công bằng: chỉ biến thiên embedding key.

Hybrid: BGE-M3 sinh dense (1024-d) + sparse (lexical) trong 1 forward. Qdrant lưu
cả hai trong cùng point, search bằng RRF fusion (xử ở retriever.py). Sparse quan
trọng cho tên thuốc/thuật ngữ y khoa VN mà dense hay bỏ sót.

Config đọc từ configs/rag.yaml (một nguồn sự thật, khớp phần còn lại của repo).

Yêu cầu:
  pip install FlagEmbedding qdrant-client
  Docker Qdrant chạy sẵn:
    docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant

Cách dùng:
  python -m src.knowledge.embedding                 # cả hai collection, resume tự động
  python -m src.knowledge.embedding --collections qa
  from src.knowledge.embedding import build_index; build_index(collections=["qa"])
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field

import yaml

# LƯU Ý THỨ TỰ IMPORT (Windows): import torch + FlagEmbedding TRƯỚC qdrant_client.
# Nếu qdrant_client (kéo theo native lib của nó) nạp trước FlagEmbedding, tiến trình
# crash cứng không traceback khi sau đó import FlagEmbedding. Đã xác minh: nạp
# torch -> FlagEmbedding trước thì không crash. Giữ đúng thứ tự này.
try:
    import torch  # noqa: F401  (ép nạp torch runtime trước)
    from FlagEmbedding import BGEM3FlagModel  # noqa: F401
    _FLAG_IMPORT_ERR = None
except Exception as _e:  # môi trường chưa cài (vd chỉ chạy test config) -> hoãn báo lỗi
    BGEM3FlagModel = None
    _FLAG_IMPORT_ERR = _e

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, SparseVector, PointStruct,
)


# ============================================================
# CONFIG — đọc từ configs/rag.yaml
# ============================================================

@dataclass
class EmbedConfig:
    corpus_path: str = "data/raw/vinmec.jsonl"
    state_path: str = "data/raw/embed_state.json"

    model_name: str = "BAAI/bge-m3"
    dense_dim: int = 1024
    use_fp16: bool = True
    device: str = "cuda"           # "cuda" | "cpu"
    batch_size: int = 32
    max_length: int = 2048

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # tên collection -> cách build text để embed (Q&A)
    collections: dict = field(default_factory=lambda: {
        "vinmec_q":  "question",
        "vinmec_qa": "question+answer",
    })

    # tri thức nền (article/PDF đã chunk): collection + list JSONL chunk để nạp
    kb_collection: str = "vinmec_kb"
    kb_chunk_paths: list = field(default_factory=list)


def config_from_yaml(path: str = "configs/rag.yaml") -> EmbedConfig:
    """Nạp EmbedConfig từ rag.yaml; default dataclass là fallback nếu thiếu key."""
    d = EmbedConfig()
    if not os.path.exists(path):
        print(f"[warn] Không thấy {path} -> dùng default hardcode.")
        return d
    with open(path, encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    emb = y.get("embedding", {})
    d.model_name = emb.get("model", d.model_name)
    d.dense_dim = emb.get("dense_dim", d.dense_dim)
    d.use_fp16 = emb.get("use_fp16", d.use_fp16)
    d.device = emb.get("device", d.device)
    d.batch_size = emb.get("batch_size", d.batch_size)
    d.max_length = emb.get("max_length", d.max_length)

    vs = y.get("vectorstore", {})
    if isinstance(vs, dict):
        backend = vs.get("backend")
        if backend and backend != "qdrant":
            print(f"[warn] rag.yaml vectorstore.backend='{backend}' nhưng module "
                  "này chỉ hỗ trợ qdrant.")
        d.qdrant_host = vs.get("host", d.qdrant_host)
        d.qdrant_port = vs.get("port", d.qdrant_port)

    # corpus path: lấy từ entry corpus đầu tiên nếu có
    corpus = y.get("corpus") or []
    if corpus and corpus[0].get("out"):
        d.corpus_path = corpus[0]["out"]

    # tri thức nền: collection + JSONL chunk từ mỗi nguồn knowledge_base có chunk=true.
    # Convention: file chunk = <out>.replace('.jsonl', '_chunks.jsonl').
    d.kb_collection = y.get("kb_collection", d.kb_collection)
    kb_paths = []
    for e in (y.get("knowledge_base") or []):
        if e.get("chunk") and e.get("out"):
            kb_paths.append(e["out"].replace(".jsonl", "_chunks.jsonl"))
    d.kb_chunk_paths = kb_paths
    return d


# ============================================================
# STATE (resume) — số doc ĐÃ INDEX làm offset + fingerprint corpus
# ============================================================

def _corpus_fingerprint(path: str) -> dict:
    st = os.stat(path)
    return {"size": st.st_size, "mtime": int(st.st_mtime)}


def load_state(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"counts": {}, "corpus": None}


def save_state(path: str, state: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ============================================================
# LOAD CORPUS
# ============================================================

def load_corpus(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = (obj.get("question") or "").strip()
            a = (obj.get("answer") or "").strip()
            if not q or not a:
                continue  # bỏ record thiếu (đã lọc ở ingest, chắc chắn lại)
            rows.append({"doc_id": i, "question": q, "answer": a})
    print(f"[corpus] {len(rows):,} record hợp lệ từ {path}")
    return rows


def build_text(row: dict, mode: str) -> str:
    if mode == "question":
        return row["question"]
    if mode == "question+answer":
        return f"{row['question']}\n\n{row['answer']}"  # ngăn cách rõ hai phần
    raise ValueError(f"mode lạ: {mode}")


# ============================================================
# QDRANT SETUP
# ============================================================

def _connect_qdrant(cfg: EmbedConfig) -> QdrantClient:
    """Kết nối Qdrant + kiểm ngay. Qdrant chưa chạy -> báo lỗi rõ (kèm lệnh docker)."""
    client = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)
    try:
        client.get_collections()
    except Exception as e:
        raise SystemExit(
            f"Không kết nối được Qdrant tại {cfg.qdrant_host}:{cfg.qdrant_port} ({e}).\n"
            "Bật Qdrant trước:\n"
            "  docker run -p 6333:6333 -p 6334:6334 "
            "-v qdrant_storage:/qdrant/storage qdrant/qdrant"
        ) from e
    return client


def ensure_collection(client: QdrantClient, name: str, dense_dim: int):
    """Tạo collection hybrid (dense + sparse) nếu chưa có."""
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(size=dense_dim, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    print(f"[qdrant] Tạo collection '{name}' (hybrid dense+sparse)")


def _encode_upsert(model, client, coll, batch_texts, batch_meta, cfg):
    """Embed 1 batch (hybrid) + upsert vào Qdrant. batch_meta: list (point_id, payload)."""
    out = model.encode(
        batch_texts,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,   # multi-vector: chưa dùng, tốn RAM
    )
    dense_vecs = out["dense_vecs"]
    sparse_weights = out["lexical_weights"]

    points = []
    for (pid, payload), dv, sw in zip(batch_meta, dense_vecs, sparse_weights):
        idxs = [int(k) for k in sw.keys()]
        vals = [float(v) for v in sw.values()]
        points.append(PointStruct(
            id=pid,
            vector={"dense": dv.tolist(),
                    "sparse": SparseVector(indices=idxs, values=vals)},
            payload=payload,
        ))
    client.upsert(collection_name=coll, points=points)


# ============================================================
# LOAD CHUNK (tri thức nền)
# ============================================================

def load_chunks(paths: list[str]) -> list[dict]:
    """Nạp JSONL chunk từ nhiều nguồn kb. Point-id = hash chunk_id (ổn định, không trùng)."""
    import hashlib

    rows = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[warn] chưa có file chunk: {path} (chạy chunk.py trước) -> bỏ qua.")
            continue
        n0 = len(rows)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = (obj.get("text") or "").strip()
                if not text:
                    continue
                cid = obj.get("chunk_id") or obj.get("doc_id") or text[:64]
                # id Qdrant: int 64-bit ổn định từ chunk_id (string) qua md5
                pid = int(hashlib.md5(str(cid).encode("utf-8")).hexdigest()[:15], 16)
                rows.append({
                    "point_id": pid,
                    "text": text,
                    "payload": {
                        "chunk_id": obj.get("chunk_id", ""),
                        "doc_id": obj.get("doc_id", ""),
                        "source": obj.get("source", ""),
                        "title": obj.get("title", ""),
                        "url": obj.get("url", ""),
                        "section": obj.get("section", ""),
                        "text": text,
                    },
                })
        print(f"[kb] {len(rows)-n0:,} chunk từ {path}")
    print(f"[kb] tổng {len(rows):,} chunk hợp lệ")
    return rows


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(collections: list[str] | None = None, cfg: EmbedConfig | None = None,
                limit: int | None = None):
    """
    Embed + index. collections: ['q','qa'] hoặc None (cả hai). Resume tự động.
    Nếu corpus đổi (size/mtime khác state) -> cảnh báo & reset count để index lại.
    limit: chỉ embed N record đầu (test nhỏ).
    """
    cfg = cfg or config_from_yaml()

    alias = {"q": "vinmec_q", "qa": "vinmec_qa"}
    targets = (list(cfg.collections.keys()) if collections is None
               else [alias.get(c, c) for c in collections])

    rows = load_corpus(cfg.corpus_path)
    if limit:
        rows = rows[:limit]
        print(f"[limit] chỉ embed {len(rows):,} record đầu.")

    # Kết nối Qdrant TRƯỚC khi load model (4.5GB) -> fail sớm nếu Qdrant chưa chạy.
    client = _connect_qdrant(cfg)

    if BGEM3FlagModel is None:
        raise SystemExit(f"Không import được FlagEmbedding/torch: {_FLAG_IMPORT_ERR}")
    print(f"[model] Load {cfg.model_name} trên {cfg.device} (fp16={cfg.use_fp16})...")
    model = BGEM3FlagModel(cfg.model_name, use_fp16=cfg.use_fp16, devices=cfg.device)

    state = load_state(cfg.state_path)

    # phát hiện corpus đổi -> resume cũ sẽ lệch, reset để index lại
    fp = _corpus_fingerprint(cfg.corpus_path)
    if state.get("corpus") and state["corpus"] != fp:
        print("[warn] corpus đã thay đổi so với lần trước (size/mtime khác) -> "
              "reset resume count, index lại từ đầu.")
        state["counts"] = {}
    state["corpus"] = fp
    counts = state.setdefault("counts", {})

    for coll in targets:
        mode = cfg.collections[coll]
        ensure_collection(client, coll, cfg.dense_dim)

        done = counts.get(coll, 0)
        if done >= len(rows):
            print(f"[skip] '{coll}' đã index đủ {done:,} record.")
            continue
        print(f"[index] '{coll}' (embed={mode}) — resume từ {done:,}/{len(rows):,}")

        pending = rows[done:]
        for start in range(0, len(pending), cfg.batch_size):
            batch = pending[start:start + cfg.batch_size]
            texts = [build_text(r, mode) for r in batch]
            meta = [(r["doc_id"], {"doc_id": r["doc_id"], "question": r["question"],
                                   "answer": r["answer"]}) for r in batch]
            _encode_upsert(model, client, coll, texts, meta, cfg)

            counts[coll] = done + start + len(batch)
            save_state(cfg.state_path, state)  # lưu sau mỗi batch -> resume an toàn

            processed = counts[coll]
            if (start // cfg.batch_size) % 10 == 0 or processed >= len(rows):
                print(f"  [{coll}] {processed:,}/{len(rows):,}")

        print(f"[done] '{coll}' index xong {counts[coll]:,} record.")

    print("\n[all done] Kiểm tra:")
    for coll in targets:
        info = client.get_collection(coll)
        print(f"  {coll}: {info.points_count:,} points")


def build_kb_index(cfg: EmbedConfig | None = None, limit: int | None = None):
    """Embed TRI THỨC NỀN (chunk article/PDF) -> collection kb. Resume theo số chunk đã index.

    Field embed = text (đã gồm heading nhờ chunk.py). Payload mang title/url/section để
    src/serving/citation.py trích dẫn được.
    """
    cfg = cfg or config_from_yaml()
    coll = cfg.kb_collection
    if not cfg.kb_chunk_paths:
        raise SystemExit("Không có kb_chunk_paths. Chạy kb_ingest + chunk trước, hoặc kiểm "
                         "knowledge_base trong rag.yaml.")

    rows = load_chunks(cfg.kb_chunk_paths)
    if not rows:
        raise SystemExit("Không có chunk nào để index.")
    if limit:
        rows = rows[:limit]
        print(f"[limit] chỉ embed {len(rows):,} chunk đầu.")

    # Kết nối Qdrant TRƯỚC khi load model (4.5GB) -> fail sớm nếu Qdrant chưa chạy.
    client = _connect_qdrant(cfg)
    ensure_collection(client, coll, cfg.dense_dim)

    if BGEM3FlagModel is None:
        raise SystemExit(f"Không import được FlagEmbedding/torch: {_FLAG_IMPORT_ERR}")
    print(f"[model] Load {cfg.model_name} trên {cfg.device} (fp16={cfg.use_fp16})...")
    model = BGEM3FlagModel(cfg.model_name, use_fp16=cfg.use_fp16, devices=cfg.device)

    # resume riêng cho kb (state key = "kb:<collection>"); fingerprint theo tổng số chunk
    state = load_state(cfg.state_path)
    counts = state.setdefault("counts", {})
    kb_key = f"kb:{coll}"
    kb_fp = {"n_chunks": len(rows)}
    if state.get(kb_key + ":fp") and state[kb_key + ":fp"] != kb_fp:
        print("[warn] số chunk kb đổi so với lần trước -> reset resume, index lại từ đầu.")
        counts[kb_key] = 0
    state[kb_key + ":fp"] = kb_fp

    done = counts.get(kb_key, 0)
    if done >= len(rows):
        print(f"[skip] '{coll}' đã index đủ {done:,} chunk.")
    else:
        print(f"[index] '{coll}' (kb, embed=text) — resume từ {done:,}/{len(rows):,}")
        pending = rows[done:]
        for start in range(0, len(pending), cfg.batch_size):
            batch = pending[start:start + cfg.batch_size]
            texts = [r["text"] for r in batch]
            meta = [(r["point_id"], r["payload"]) for r in batch]
            _encode_upsert(model, client, coll, texts, meta, cfg)

            counts[kb_key] = done + start + len(batch)
            save_state(cfg.state_path, state)
            processed = counts[kb_key]
            if (start // cfg.batch_size) % 10 == 0 or processed >= len(rows):
                print(f"  [{coll}] {processed:,}/{len(rows):,}")
        print(f"[done] '{coll}' index xong {counts[kb_key]:,} chunk.")

    info = client.get_collection(coll)
    print(f"\n[all done] {coll}: {info.points_count:,} points")


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed vinmec (BGE-M3 hybrid) -> Qdrant")
    ap.add_argument("--config", default="configs/rag.yaml")
    ap.add_argument("--collections", nargs="*", default=None,
                    help="q, qa (Q&A) và/hoặc kb (tri thức nền). Mặc định: cả hai Q&A.")
    ap.add_argument("--limit", type=int, default=None,
                    help="chỉ embed N record đầu (test nhỏ trước khi chạy full)")
    args = ap.parse_args()
    cfg = config_from_yaml(args.config)

    cols = args.collections
    if cols and "kb" in cols:
        build_kb_index(cfg=cfg, limit=args.limit)
        cols = [c for c in cols if c != "kb"]        # phần còn lại (nếu có) là Q&A
    if cols or not args.collections:
        build_index(collections=cols or None, cfg=cfg, limit=args.limit)


if __name__ == "__main__":
    main()
