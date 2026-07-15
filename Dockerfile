# FastAPI serving app cho RAG y tế. NHẸ: encode/rerank chạy remote (Colab), không load model.
FROM python:3.11-slim

WORKDIR /app

# deps hệ thống tối thiểu
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# cài deps trước (cache layer) — chỉ requirements serving nhẹ
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# copy code + config + web (KHÔNG copy data/ nặng — xem .dockerignore)
COPY src/ ./src/
COPY configs/ ./configs/
COPY prompts/ ./prompts/
COPY web/ ./web/

EXPOSE 8000

# healthcheck: /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -fs http://localhost:8000/health || exit 1

CMD ["python", "-m", "src.serving.app"]
