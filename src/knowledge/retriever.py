"""Knowledge plane — Retriever RAG: hybrid + RRF + rerank + source priority + threshold.

Luồng (mỗi query):
  query
   -> BGE-M3 encode (dense+sparse), device = retriever.query_device (mặc định cpu)
   -> vectorstore.hybrid_search_multi trên [vinmec_kb, vinmec_qa]  (RRF trong Qdrant)
   -> rerank (cross-encoder) toàn bộ candidate  -> điểm relevance
   -> source priority: bonus NHỎ tie-break (byt-kcb > vinmec-article > qa)
   -> threshold: loại hit < min_score; rỗng -> trả [] ("không tìm thấy", KHÔNG bịa)
   -> top_n Hit (kèm text/title/url/source cho citation)

CHỐNG TRẢ LỜI SAI: rerank lọc nhiễu + threshold (rỗng khi không đủ liên quan) + payload
mang nguồn để tầng generation ép citation. KHÔNG thu hẹp nguồn (đó là giảm recall).

Encode 1 query nhẹ -> CPU chịu được (~1-2s), không cần GPU như lúc embed corpus.

Usage:
  from src.knowledge.retriever import Retriever
  r = Retriever()
  for hit in r.retrieve("đau dạ dày uống thuốc gì?"):
      print(hit.score, hit.source, hit.title, hit.text[:100])
"""
from __future__ import annotations

from dataclasses import dataclass

import yaml

# Nạp .env (RAG_REMOTE_URL, RAG_REMOTE_TOKEN...) khi chạy local.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# THỨ TỰ IMPORT (Windows): nạp torch + FlagEmbedding TRƯỚC qdrant_client (qua vectorstore).
# LƯU Ý: KHÔNG import FlagEmbedding ở top-level. Khi backend=remote (encode/rerank gọi
# HTTP tới model server), máy client KHÔNG cần FlagEmbedding — và import nó có thể CRASH
# NATIVE (segfault, try/except không bắt được) trên máy thiếu GPU/thư viện. Chỉ import
# trong nhánh local của _encode_query (lazy), khi thật sự cần.

from src.knowledge.vectorstore import connect, hybrid_search_multi
from src.knowledge.reranker import Reranker


CONFIG_PATH = "configs/rag.yaml"


@dataclass
class Hit:
    text: str
    score: float                    # điểm rerank (đã cộng source bonus)
    rerank_score: float             # điểm rerank thuần (trước bonus)
    source: str
    title: str
    url: str
    chunk_id: str
    collection: str


@dataclass
class RetrieverConfig:
    model_name: str = "BAAI/bge-m3"
    query_device: str = "cpu"
    max_length: int = 1024
    top_k: int = 8
    top_n: int = 4
    min_score: float = 0.0
    collections: tuple = ("vinmec_kb", "vinmec_qa")
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"
    reranker_fp16: bool = False
    source_priority: dict = None    # {source: bonus}
    # remote model server (Colab qua ngrok) — máy yếu không load nổi model
    encoder_backend: str = "local"  # local | remote
    reranker_backend: str = "local"
    remote_url: str = ""
    remote_url_backup: str = ""
    remote_token: str = ""
    remote_timeout: int = 30
    remote_retries: int = 1

    def __post_init__(self):
        if self.source_priority is None:
            self.source_priority = {}


def config_from_yaml(path: str = CONFIG_PATH) -> RetrieverConfig:
    c = RetrieverConfig()
    import os
    if not os.path.exists(path):
        return c
    with open(path, encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    emb = y.get("embedding", {}) or {}
    c.model_name = emb.get("model", c.model_name)
    c.max_length = emb.get("max_length", c.max_length)

    r = y.get("retriever", {}) or {}
    c.query_device = r.get("query_device", c.query_device)
    c.top_k = r.get("top_k", c.top_k)
    c.top_n = r.get("top_n", c.top_n)
    c.min_score = r.get("min_score", c.min_score)
    c.collections = tuple(r.get("collections", list(c.collections)))
    c.encoder_backend = r.get("encoder_backend", c.encoder_backend)
    c.remote_url = r.get("remote_url", c.remote_url)
    c.remote_url_backup = r.get("remote_url_backup", c.remote_url_backup)
    c.remote_token = r.get("remote_token", c.remote_token)
    c.remote_timeout = r.get("remote_timeout", c.remote_timeout)
    c.remote_retries = r.get("remote_retries", c.remote_retries)

    rr = y.get("reranker", {}) or {}
    c.reranker_model = rr.get("model", c.reranker_model)
    c.reranker_device = rr.get("device", c.reranker_device)
    c.reranker_fp16 = rr.get("use_fp16", c.reranker_fp16)
    c.reranker_backend = rr.get("backend", c.reranker_backend)

    # env override (đổi ngrok URL mỗi session Colab dễ, không sửa yaml)
    c.remote_url = os.environ.get("RAG_REMOTE_URL", c.remote_url)
    c.remote_url_backup = os.environ.get("RAG_REMOTE_URL_BACKUP", c.remote_url_backup)
    c.remote_token = os.environ.get("RAG_REMOTE_TOKEN", c.remote_token)

    vs = y.get("vectorstore", {}) or {}
    c.qdrant_host = vs.get("host", c.qdrant_host)
    c.qdrant_port = vs.get("port", c.qdrant_port)
    # env override: trong container docker-compose, Qdrant là service 'qdrant' (không localhost)
    c.qdrant_host = os.environ.get("QDRANT_HOST", c.qdrant_host)
    c.qdrant_port = int(os.environ.get("QDRANT_PORT", c.qdrant_port))

    c.source_priority = y.get("source_priority", {}) or {}
    return c


def _near_dup_key(text: str, n: int = 80) -> str:
    """Chữ ký thô để nhận near-duplicate: n ký tự đầu đã chuẩn hoá khoảng trắng."""
    return " ".join((text or "").split())[:n].lower()


def rank_candidates(cands, rerank_scores, source_priority: dict,
                    min_score: float, top_n: int) -> list[Hit]:
    """Logic THUẦN (không cần model/Qdrant, test được): threshold + bonus + dedup + sort.

    - threshold trên điểm rerank THUẦN (rs < min_score -> loại).
    - source bonus NHỎ (chia 100) chỉ tie-break, không lấn át relevance.
    - khử near-duplicate: cùng url HOẶC ~80 ký tự đầu trùng -> giữ chunk điểm cao hơn.
    """
    hits = []
    for c, rs in zip(cands, rerank_scores):
        if rs < min_score:
            continue
        source = c.payload.get("source", "")
        final = rs + source_priority.get(source, 0) / 100.0
        hits.append(Hit(
            text=c.payload.get("text", ""),
            score=final, rerank_score=rs,
            source=source,
            title=c.payload.get("title", ""),
            url=c.payload.get("url", ""),
            chunk_id=str(c.payload.get("chunk_id", getattr(c, "point_id", ""))),
            collection=getattr(c, "collection", ""),
        ))
    hits.sort(key=lambda h: h.score, reverse=True)

    # khử near-duplicate SAU sort (giữ cái điểm cao nhất của mỗi nhóm trùng)
    seen_url, seen_text, deduped = set(), set(), []
    for h in hits:
        tkey = _near_dup_key(h.text)
        # url rỗng (vd PDF phác đồ) -> không dedup theo url, chỉ theo text
        if (h.url and h.url in seen_url) or tkey in seen_text:
            continue
        if h.url:
            seen_url.add(h.url)
        seen_text.add(tkey)
        deduped.append(h)
    return deduped[:top_n]


class Retriever:
    def __init__(self, cfg: RetrieverConfig | None = None):
        self.cfg = cfg or config_from_yaml()
        self.client = connect(self.cfg.qdrant_host, self.cfg.qdrant_port)
        self.reranker = Reranker(self.cfg.reranker_model, self.cfg.reranker_device,
                                 self.cfg.reranker_fp16,
                                 backend=self.cfg.reranker_backend, cfg=self.cfg)
        self._encoder = None

    def _encode_remote(self, query: str):
        """Gọi model server /encode (có fallback + retry) -> (dense, sparse_idx, sparse_val).

        Cả 2 endpoint chết -> RemoteUnavailable (orchestrator bắt -> graceful degrade).
        """
        from src.knowledge.remote_client import post_with_fallback
        d = post_with_fallback("/encode",
                               {"query": query, "max_length": self.cfg.max_length},
                               self.cfg)
        return d["dense"], d["sparse"]["indices"], d["sparse"]["values"]

    def _encode_query(self, query: str):
        """Encode query -> (dense, sparse_idx, sparse_val). local BGE-M3 | remote Colab."""
        if self.cfg.encoder_backend == "remote":
            return self._encode_remote(query)
        if self._encoder is None:
            import torch  # noqa: F401  (thứ tự import an toàn)
            from FlagEmbedding import BGEM3FlagModel
            print(f"[retriever] Load {self.cfg.model_name} (query) trên {self.cfg.query_device}...")
            self._encoder = BGEM3FlagModel(self.cfg.model_name, use_fp16=False,
                                           devices=self.cfg.query_device)
        out = self._encoder.encode([query], max_length=self.cfg.max_length,
                                   return_dense=True, return_sparse=True,
                                   return_colbert_vecs=False)
        dense = [float(x) for x in out["dense_vecs"][0]]
        sw = out["lexical_weights"][0]
        return dense, [int(k) for k in sw.keys()], [float(v) for v in sw.values()]

    def retrieve(self, query: str) -> list[Hit]:
        dense, sp_idx, sp_val = self._encode_query(query)
        cands = hybrid_search_multi(self.client, list(self.cfg.collections),
                                    dense, sp_idx, sp_val, self.cfg.top_k)
        if not cands:
            return []
        rerank_scores = self.reranker.score(query, [c.payload.get("text", "") for c in cands])
        return rank_candidates(cands, rerank_scores, self.cfg.source_priority,
                               self.cfg.min_score, self.cfg.top_n)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Test retriever RAG")
    ap.add_argument("query")
    ap.add_argument("--config", default=CONFIG_PATH)
    args = ap.parse_args()
    r = Retriever(config_from_yaml(args.config))
    hits = r.retrieve(args.query)
    if not hits:
        print("[retriever] Không tìm thấy thông tin đủ liên quan.")
        return
    for i, h in enumerate(hits, 1):
        print(f"\n#{i} [{h.source}] score={h.score:.3f} (rerank={h.rerank_score:.3f})")
        print(f"   {h.title[:80]}")
        print(f"   {h.url}")
        print(f"   {h.text[:200].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()
