#!/usr/bin/env bash
# Phase 4: FastAPI + RAG (demo offline)
set -euo pipefail
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
