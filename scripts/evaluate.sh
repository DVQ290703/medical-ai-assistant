#!/usr/bin/env bash
# Phase 3: chạy eval harness theo manifest. --smoke = subset nhanh cho CI.
set -euo pipefail
python -m src.evaluation.harness --manifest configs/evaluation_manifest.yaml "$@"
echo "[evaluate] done -> reports/"
