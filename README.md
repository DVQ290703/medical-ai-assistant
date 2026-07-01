# Medical AI Assistant (Unsloth + QLoRA + RAG)

Trợ lý Q&A **y học hiện đại tiếng Việt**. Fine-tune (Llama 3.1 8B / QLoRA-Unsloth) để dạy
*reasoning + phong cách + citation*; facts đến từ **RAG nguồn y khoa VN** (không nhớ trong model).

**Tiếng Việt = low-resource:** không có sẵn dataset reasoning y khoa VN -> data theo hướng
hybrid (dịch medical-o1 làm seed + augment Vietnamese-native). Base model & embedding phải
benchmark cho tiếng Việt (xem ADR-0003, ADR-0005).

> ⚠️ Research/education only. KHÔNG chẩn đoán, kê đơn, hay quyết định y tế thật.
> Xem `governance/model_card.md` và `docs/medical-ai-assistant-plan.md`.

## Designed vs Built
- **Built (chạy thật, kể cả trên Kaggle)**: `src/data`, `src/knowledge`, `src/prompting`,
  `src/training`, `src/evaluation`, RAG demo offline.
- **Designed + stub (interface + mock, deploy thật cần cloud)**: phần cloud của `src/serving`
  (Redis, autoscale) và `src/monitoring` (live dashboard, alerting).

## Architecture (3 planes)
- **Knowledge plane** (offline): sources → chunk/embed → vector DB. → `src/knowledge/`
- **Serving plane** (online): input guard → retrieval → prompt → generation → output guard →
  policy → citation. → `src/serving/`, `src/prompting/`, `src/generation/`
- **Observability plane**: monitoring → human review queue → feedback. → `src/monitoring/`

Chi tiết: `docs/architecture.md`.

## Ba loại "prompt" (đừng nhầm)
- `prompts/`        → TEXT template thật (versioned, là asset)
- `configs/prompt.yaml` → tham số (temperature, max_tokens, citation_required)
- `src/prompting/`  → CODE build prompt

## Quickstart
```bash
pip install -e .
make data       # Phase 1: ingest -> validate -> clean -> pii -> split
make train      # Phase 2: QLoRA fine-tune (Unsloth)
make eval       # Phase 3: benchmark + quality + retrieval + safety (đọc evaluation_manifest)
make serve      # Phase 4: FastAPI + RAG (demo offline)
```

## Layout
Xem `docs/architecture.md`. `data/` và `artifacts/` KHÔNG commit (DVC-tracked).
