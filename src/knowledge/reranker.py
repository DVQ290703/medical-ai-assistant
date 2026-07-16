"""Knowledge plane — Cross-encoder reranker (bge-reranker-v2-m3), lọc context nhiễu.

Bi-encoder (BGE-M3) retrieve nhanh nhưng thô; cross-encoder đọc CẢ query lẫn chunk cùng
lúc -> điểm relevance chính xác hơn. Xếp lại + lọc candidate trước khi vào prompt.

Dùng sentence-transformers CrossEncoder (KHÔNG dùng FlagReranker): FlagReranker gọi
tokenizer API cũ (prepare_for_model) đã bị bỏ ở transformers 5.x -> lỗi. CrossEncoder
tương thích transformers mới. Cùng model bge-reranker-v2-m3.

Reranker chỉ chấm ~top_k*n_collection candidate (ít) -> CPU đủ. Lazy-load 1 lần.
"""
from __future__ import annotations


class Reranker:
    """score(query, texts) -> điểm relevance. backend: local (CrossEncoder) | remote (Colab)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3",
                 device: str = "cpu", use_fp16: bool = False,
                 backend: str = "local", cfg=None):
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16          # giữ cho khớp config; CrossEncoder tự lo dtype
        self.backend = backend
        self.cfg = cfg                    # RetrieverConfig — dùng cho remote fallback
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print(f"[reranker] Load {self.model_name} trên {self.device}...")
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def _score_remote(self, query: str, texts: list[str]) -> list[float]:
        """Gọi model server /rerank (có fallback + retry). Cả 2 chết -> RemoteUnavailable."""
        from src.knowledge.remote_client import post_with_fallback
        d = post_with_fallback("/rerank", {"query": query, "texts": texts}, self.cfg)
        return [float(s) for s in d["scores"]]

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Điểm relevance cho từng (query, text). texts rỗng -> []. Cao = liên quan hơn.

        bge-reranker xuất logit; normalize/sigmoid -> ~[0,1].
        """
        if not texts:
            return []
        if self.backend == "remote":
            return self._score_remote(query, texts)
        model = self._load()
        scores = model.predict([[query, t] for t in texts])
        return [float(s) for s in scores]
