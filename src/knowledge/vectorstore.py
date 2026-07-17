"""Knowledge plane — Wrapper Qdrant: hybrid search (dense+sparse RRF).

Chỉ lo tầng vector DB: nhận dense+sparse vector của query -> query_points với Prefetch
(dense + sparse) + FusionQuery(RRF) -> trả candidate. Không encode, không rerank (việc đó
ở retriever.py / reranker.py).

Qdrant 1.18 Query API: query_points + models.Prefetch + models.FusionQuery(Fusion.RRF).
"""
from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector


@dataclass
class Candidate:
    point_id: int | str
    score: float                    # điểm RRF từ Qdrant (chưa rerank)
    collection: str
    payload: dict                   # {text, title, url, source, section, ...}


def connect(host: str = "localhost", port: int = 6333) -> QdrantClient:
    """Kết nối Qdrant. Ưu tiên Qdrant Cloud nếu có QDRANT_URL + QDRANT_API_KEY (deploy),
    ngược lại dùng host:port (local/docker).
    """
    import os
    url = os.environ.get("QDRANT_URL", "").strip()
    api_key = os.environ.get("QDRANT_API_KEY", "").strip()
    # timeout dài cho Cloud (upload batch lớn qua internet dễ ReadTimeout với mặc định ~5s).
    timeout = int(os.environ.get("QDRANT_TIMEOUT", "120"))
    if url:                       # Qdrant Cloud: https://xxx.qdrant.io + api key
        client = QdrantClient(url=url, api_key=api_key or None, timeout=timeout)
        target = url
    else:                         # local / docker: host:port
        client = QdrantClient(host=host, port=port, timeout=timeout)
        target = f"{host}:{port}"
    try:
        client.get_collections()
    except Exception as e:
        raise SystemExit(
            f"Không kết nối Qdrant {target} ({e}). "
            "Local: bật docker qdrant. Cloud: kiểm tra QDRANT_URL/QDRANT_API_KEY."
        ) from e
    return client


def hybrid_search(client: QdrantClient, collection: str,
                  dense: list[float], sparse_indices: list[int],
                  sparse_values: list[float], top_k: int = 8) -> list[Candidate]:
    """1 collection: hybrid dense+sparse, RRF fuse -> top_k candidate.

    Prefetch dense và sparse riêng (mỗi cái lấy top_k), rồi RRF gộp thứ hạng.
    """
    res = client.query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(query=dense, using="dense", limit=top_k),
            Prefetch(query=SparseVector(indices=sparse_indices, values=sparse_values),
                     using="sparse", limit=top_k),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return [Candidate(point_id=p.id, score=p.score or 0.0,
                      collection=collection, payload=p.payload or {})
            for p in res.points]


def hybrid_search_multi(client: QdrantClient, collections: list[str],
                        dense: list[float], sparse_indices: list[int],
                        sparse_values: list[float], top_k: int = 8) -> list[Candidate]:
    """Search nhiều collection, gộp candidate (rerank sẽ xếp lại nên không cần merge điểm)."""
    out: list[Candidate] = []
    for coll in collections:
        out.extend(hybrid_search(client, coll, dense, sparse_indices,
                                 sparse_values, top_k))
    return out
