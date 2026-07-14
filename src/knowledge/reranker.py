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
    """Wrap sentence-transformers CrossEncoder. score(query, texts) -> điểm relevance."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3",
                 device: str = "cpu", use_fp16: bool = False):
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16          # giữ cho khớp config; CrossEncoder tự lo dtype
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print(f"[reranker] Load {self.model_name} trên {self.device}...")
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Điểm relevance cho từng (query, text). texts rỗng -> []. Điểm cao = liên quan hơn.

        bge-reranker xuất logit; CrossEncoder.predict mặc định áp sigmoid -> ~[0,1].
        """
        if not texts:
            return []
        model = self._load()
        scores = model.predict([[query, t] for t in texts])
        return [float(s) for s in scores]
