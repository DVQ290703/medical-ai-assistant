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
                 backend: str = "local", remote_url: str = "", remote_token: str = ""):
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16          # giữ cho khớp config; CrossEncoder tự lo dtype
        self.backend = backend
        self.remote_url = remote_url
        self.remote_token = remote_token
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print(f"[reranker] Load {self.model_name} trên {self.device}...")
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def _score_remote(self, query: str, texts: list[str]) -> list[float]:
        """Gọi model server (Colab) /rerank."""
        import requests
        if not self.remote_url:
            raise SystemExit("reranker backend=remote nhưng thiếu remote_url "
                             "(set RAG_REMOTE_URL trong .env).")
        try:
            r = requests.post(self.remote_url.rstrip("/") + "/rerank",
                              json={"query": query, "texts": texts},
                              headers={"X-Token": self.remote_token}, timeout=60)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise SystemExit(f"[remote] rerank server không phản hồi ({e}). "
                             "Kiểm tra notebook Colab còn chạy?")
        return [float(s) for s in r.json()["scores"]]

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
