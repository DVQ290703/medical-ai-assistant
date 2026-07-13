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

    rr = y.get("reranker", {}) or {}
    c.reranker_model = rr.get("model", c.reranker_model)
    c.reranker_device = rr.get("device", c.reranker_device)
    c.reranker_fp16 = rr.get("use_fp16", c.reranker_fp16)

    vs = y.get("vectorstore", {}) or {}
    c.qdrant_host = vs.get("host", c.qdrant_host)
    c.qdrant_port = vs.get("port", c.qdrant_port)

    c.source_priority = y.get("source_priority", {}) or {}
    return c


def rank_candidates(cands, rerank_scores, source_priority: dict,
                    min_score: float, top_n: int) -> list[Hit]:
    """Logic THUẦN (không cần model/Qdrant, test được): threshold + source bonus + sort.

    - threshold trên điểm rerank THUẦN (rs < min_score -> loại).
    - source bonus NHỎ (chia 100) chỉ tie-break, không lấn át relevance.
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
    return hits[:top_n]


class Retriever:
    def __init__(self, cfg: RetrieverConfig | None = None):
        self.cfg = cfg or config_from_yaml()
        self.client = connect(self.cfg.qdrant_host, self.cfg.qdrant_port)
        self.reranker = Reranker(self.cfg.reranker_model, self.cfg.reranker_device,
                                 self.cfg.reranker_fp16)
        self._encoder = None

    def _encode_query(self, query: str):
        """BGE-M3 encode query -> (dense list, sparse indices, sparse values)."""
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
