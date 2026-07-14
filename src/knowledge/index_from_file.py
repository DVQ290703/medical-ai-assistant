"""Knowledge plane — Nạp file vector (từ encode_offline trên Colab) -> Qdrant LOCAL.

Chạy trên máy local, KHÔNG cần torch/GPU/FlagEmbedding — chỉ đọc file .jsonl vector
(dense+sparse đã tính sẵn) rồi upsert vào Qdrant. Cặp với src/knowledge/encode_offline.py.

Luồng:
  Colab:  encode_offline.py  -> kb_vectors.jsonl (tải về máy)
  Local:  index_from_file.py -> đẩy vào Qdrant collection

Usage:
  python -m src.knowledge.index_from_file --in kb_vectors.jsonl --collection vinmec_kb
  python -m src.knowledge.index_from_file --in qa_vectors.jsonl --collection vinmec_q
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, SparseVector, PointStruct,
)


CONFIG_PATH = "configs/rag.yaml"


def load_qdrant_cfg(path: str = CONFIG_PATH):
    host, port, dim = "localhost", 6333, 1024
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        vs = y.get("vectorstore", {}) or {}
        host = vs.get("host", host)
        port = vs.get("port", port)
        dim = (y.get("embedding", {}) or {}).get("dense_dim", dim)
    return host, port, dim


def ensure_collection(client: QdrantClient, name: str, dense_dim: int):
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(size=dense_dim, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    print(f"[qdrant] Tạo collection '{name}' (hybrid dense+sparse)")


def delete_by_source(client: QdrantClient, collection: str, source: str) -> None:
    """Xóa mọi point có payload.source == source (vd 'byt-kcb') khỏi collection.

    KHÔNG đụng file chunk trên đĩa — chỉ xóa vector trong Qdrant. Dựng lại được bằng
    embed + index từ file chunk.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        print(f"[del] collection '{collection}' chưa có -> bỏ qua.")
        return
    flt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
    client.delete(collection_name=collection, points_selector=flt)
    info = client.get_collection(collection)
    print(f"[del] đã xóa point source='{source}' khỏi '{collection}'. Còn: {info.points_count:,}")


def index_file(in_path: str, collection: str, config_path: str = CONFIG_PATH,
               batch_size: int = 256, delete_source: str | None = None) -> int:
    host, port, dim = load_qdrant_cfg(config_path)
    client = QdrantClient(host=host, port=port)
    try:
        client.get_collections()
    except Exception as e:
        raise SystemExit(f"Không kết nối Qdrant {host}:{port} ({e}). Bật docker qdrant trước.")
    ensure_collection(client, collection, dim)
    if delete_source:
        delete_by_source(client, collection, delete_source)

    n = 0
    batch = []

    def flush():
        nonlocal batch
        if batch:
            client.upsert(collection_name=collection, points=batch)
            batch = []

    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sp = r["sparse"]
            batch.append(PointStruct(
                id=r["point_id"],
                vector={"dense": r["dense"],
                        "sparse": SparseVector(indices=sp["indices"], values=sp["values"])},
                payload=r["payload"],
            ))
            n += 1
            if len(batch) >= batch_size:
                flush()
                if (n // batch_size) % 10 == 0:
                    print(f"  [{collection}] {n:,}")
    flush()

    info = client.get_collection(collection)
    print(f"[index] {n:,} vector -> '{collection}' | tổng points: {info.points_count:,}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Nạp file vector -> Qdrant local (không cần GPU)")
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--delete-source", default=None,
                    help="xóa point source=X trong collection TRƯỚC khi index (vd byt-kcb)")
    args = ap.parse_args()
    index_file(args.in_path, args.collection, args.config, args.batch_size,
               args.delete_source)


if __name__ == "__main__":
    main()
