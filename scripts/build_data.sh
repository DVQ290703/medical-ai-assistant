#!/usr/bin/env bash
# Phase 1 (VN): ingest -> translate -> validate -> clean/dedup -> pii -> decontaminate -> split
set -euo pipefail
python -m src.data.ingest        --config configs/data.yaml
python -m src.data.translate     --config configs/data.yaml   # dịch seed EN -> VI + quality gate
python -m src.data.validate      --config configs/data.yaml
python -m src.data.clean_dedup   --config configs/data.yaml
python -m src.data.pii_scrub     --config configs/data.yaml
python -m src.data.decontaminate --config configs/data.yaml   # BẮT BUỘC: loại rò rỉ train/eval
python -m src.data.split         --config configs/data.yaml
echo "[build_data] done -> data/processed"
