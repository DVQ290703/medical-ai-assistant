#!/usr/bin/env bash
# Phase 4: FastAPI + RAG chat UI. Cần: Qdrant chạy + .env (GROQ_API_KEY, RAG_REMOTE_*).
# Mở http://localhost:8000 sau khi chạy.
set -euo pipefail
python -m src.serving.app
