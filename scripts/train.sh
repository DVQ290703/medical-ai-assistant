#!/usr/bin/env bash
# Phase 2: QLoRA fine-tune (Unsloth)
set -euo pipefail
python -m src.training.finetune --config configs/training/qlora_r16.yaml
echo "[train] done -> artifacts/adapters"
