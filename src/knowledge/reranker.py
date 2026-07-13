"""Knowledge plane — Cross-encoder reranker (bge-reranker-v2-m3), lọc context nhiễu.

Bi-encoder (BGE-M3) retrieve nhanh nhưng thô; cross-encoder đọc CẢ query lẫn chunk cùng
lúc -> điểm relevance chính xác hơn nhiều. Dùng để xếp lại + lọc candidate trước khi đưa
vào prompt.

Reranker chỉ chấm ~top_k*n_collection candidate (ít) -> chạy CPU đủ (khác BGE-M3 embed
152k chunk cần GPU). Lazy-load 1 lần.
"""
from __future__ import annotations


class Reranker:
    """Wrap FlagReranker. score(query, texts) -> list điểm relevance (cao = liên quan hơn)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3",
                 device: str = "cpu", use_fp16: bool = False):
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self._model = None

    def _load(self):
        if self._model is None:
            from FlagEmbedding import FlagReranker
            print(f"[reranker] Load {self.model_name} trên {self.device}...")
            self._model = FlagReranker(self.model_name, use_fp16=self.use_fp16,
                                       devices=self.device)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Điểm relevance cho từng (query, text). texts rỗng -> []."""
        if not texts:
            return []
        model = self._load()
        pairs = [[query, t] for t in texts]
        scores = model.compute_score(pairs, normalize=True)   # normalize -> [0,1] (sigmoid)
        # FlagReranker trả float đơn nếu 1 cặp -> ép list
        return [float(s) for s in (scores if isinstance(scores, list) else [scores])]
