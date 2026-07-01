# Medical AI Assistant — Production Plan (Unsloth + QLoRA)

> Fine-tune Llama 3.1 8B cho medical reasoning bằng QLoRA + Unsloth, thiết kế theo chuẩn
> enterprise production. Build & train trên Kaggle, thiết kế + document toàn bộ vòng đời như
> một sản phẩm AI y tế thật.

---

## Vòng đời hệ thống (đây là AI System, không phải notebook train model)

```
Problem Definition -> Risk Assessment -> Data Pipeline -> Golden Evaluation Sets
    -> Prompt Engineering -> Fine-tuning -> Evaluation -> Failure Attribution
    -> Safety Validation -> Serving -> Monitoring -> Feedback Loop
```

Triết lý lặp: `Prompt -> RAG -> Fine-tune -> Evaluation -> Failure Attribution -> Engineering Fix`

---

## Nguyên tắc nền tảng (đọc trước khi làm bất cứ gì)

**Kaggle không phải nơi chạy production.** Free ~30h GPU/tuần (P100 16GB hoặc 2×T4), session
tối đa 12h — đủ để build data pipeline + fine-tune + evaluate, nhưng **không host được API
live 24/7**. Chiến lược: build & train trên Kaggle, thiết kế + document toàn bộ theo chuẩn
enterprise, phần serving stub lại và chỉ rõ deploy ở đâu.

**Fine-tune KHÔNG dùng để nhớ facts.** Facts (liều thuốc, guideline) lấy từ **RAG có trích
dẫn**. Fine-tune chỉ dạy *phong cách, reasoning, format, persona*. Hệ thống thật = fine-tune
+ RAG, và **RAG nằm TRƯỚC model** trong luồng runtime.

**Prompt trước, fine-tune sau.** Optimize prompt + RAG trước; fine-tune đẩy nốt phần còn lại.

**Designed vs Built.** Trong report đánh dấu rõ phần *build thật* (data, EDA, fine-tune, eval
core, RAG demo) vs phần *thiết kế + stub* (serving cloud, online eval, CI/CD, dashboard).

**Ranh giới công sức:** fine-tune (Unsloth lo ~80%, đừng dồn công) — evaluation (~40% giá trị).

---

## Goals / Non-goals

**Goals — project HƯỚNG TỚI:**
```
✓ Trả lời câu hỏi y khoa có trích nguồn (education · drug info · guideline)
✓ Reasoning minh bạch + citation bắt buộc
✓ An toàn: từ chối / escalate đúng lúc
✓ Reproducible + evaluated nghiêm túc (benchmark + safety + retrieval)
```

**Non-goals — project KHÔNG hướng tới:**
```
✗ Build a diagnostic system
✗ Replace physicians
✗ Provide prescriptions
✗ Interpret medical images
✗ Serve as a regulated medical device
```

---

## Kiến trúc runtime (RAG trước model)

```
                    +------------------------+
                    | Medical Knowledge      |
                    | Guidelines · Drug DB   |
                    | WHO · CDC · PubMed     |
                    +-----------+------------+
                                |
                          Chunk & Embed
                                |
                          Vector Database
                                |
User                            |
 |                              |
 v                              |
Input Guard                     |
 |                              v
Retriever  <--------------------+
 |
 v
Re-ranking
 |
 v
Prompt Builder
 |
 v
Fine-tuned Llama 3.1 (QLoRA)
 |
 v
Output Guard
 |
 v
Policy Engine   (need doctor? disclaimer? escalation? refuse?)
 |
 v
Citation Injection
 |
 v
Response
 |
 v
Monitoring  --->  Feedback Loop (đã ẩn PII) ---> Data pipeline (vòng sau)
```

Kaggle: Data pipeline, Prompt eng, Fine-tune, Eval. Cloud (stub): Registry, Serving, RAG
runtime, Guardrails, Monitoring.

---

## Phase 0 — Scoping, Risk & Compliance

```
Problem Definition -> Risk Assessment -> Compliance -> Scope -> Success Metrics
```

### 0.1 Problem Definition
"Trợ lý trả lời câu hỏi y khoa có trích nguồn, phục vụ tra cứu & giáo dục, KHÔNG chẩn đoán
cá nhân."

### 0.2 Risk Assessment
| Rủi ro | Mức độ | Giảm thiểu |
|---|---|---|
| Bịa liều thuốc / facts sai | Nghiêm trọng | RAG + citation + drug-level eval |
| Chẩn đoán khi không được phép | Nghiêm trọng | Policy Engine + out-of-scope detection |
| Bỏ sót cấp cứu | Nghiêm trọng | Emergency routing (recall > 95%) |
| Rò rỉ PII/PHI | Cao | PII scrub in/out + audit log ẩn PII |
| Jailbreak / prompt injection | Cao | Input Guard + red-team suite |
| Feedback poisoning | Cao | Human review queue trước khi vào train |
| Bị xếp là thiết bị y tế | Cao (pháp lý) | Scope chặt + disclaimer + human-in-the-loop |

### 0.3 Compliance
- [ ] HIPAA / GDPR — không log PHI thô
- [ ] License base model — Llama 3.1 (thương mại có điều kiện)
- [ ] License dataset — `medical-o1-reasoning-SFT`: check commercial terms
- [ ] Disclaimer bắt buộc mọi output
- [ ] Human-in-the-loop rõ ràng

### 0.4 Scope (chi tiết capability)
| ✅ In scope | ❌ Out of scope |
|---|---|
| Medical education | Diagnosis (chẩn đoán cá nhân) |
| Drug information (tra cứu) | Prescription (kê đơn) |
| Guideline explanation | Emergency advice (thay cấp cứu) |
| Q&A có trích nguồn | Image interpretation |

### 0.5 Success Metrics (kèm phương pháp đo)
| Metric | Target | Đo bằng |
|---|---|---|
| MedQA accuracy | > base +5% (CI không chồng 0) | Accuracy trên held-out MedQA, ≥3 seeds |
| General hallucination | < 10% | Human review + LLM judge trên red-team set |
| Drug dosage hallucination | < 5% | Manual annotation trên drug safety set |
| Emergency-advice hallucination | < 2% | Manual annotation trên emergency set |
| Emergency routing recall | > 95% | test set câu hỏi cấp cứu gán nhãn |
| Citation coverage | > 95% | % câu trả lời có ≥1 citation hợp lệ |
| Groundedness (Faithfulness) | > 0.90 | RAGAS Faithfulness (proxy, xem 3.2) |
| Latency (p95) | < 2s | P95 inference latency (load test) |
| GPU memory | < 12GB | Peak VRAM inference (cần quantized: AWQ/GPTQ; fp16 8B ≈ 16GB) |

### 0.6 Metrics theo tầng (Model / Retrieval / System)
| Category | Metric |
|---|---|
| Model | Accuracy · Hallucination · Faithfulness |
| Retrieval | Recall@5 · MRR · Context Precision · Context Recall |
| System | Latency (p95) · Cost/query · GPU memory |

---

## Phase 1 — Data Pipeline & Golden Evaluation Sets

```
Ingest -> Validate -> Clean/Dedup -> PII Scrub -> Split (no leakage) -> Data Quality Report -> Version
```

- **Versioning**: DVC / HF Datasets.
- **Cleaning**: dedup near-duplicate (MinHash/embedding), lọc rác, chuẩn hóa format.
- **PII/PHI scrub**: Presidio hoặc regex + NER.
- **Split chống leakage**: tách train/val/test *trước* mọi thứ.

### Data Validation (kiểu Great Expectations)
```
Schema validation · Missing field · Empty answer · Long sequence
Broken UTF-8 · Language mismatch
```
Fail sớm ở đây rẻ hơn nhiều so với phát hiện lúc train/eval.

### Data Quality Report + `EDA.ipynb`
```
Number of samples · Average tokens (in/out) · Medical specialties
Language distribution · Conversation length · Reasoning length
```

### Golden Evaluation Sets (deliverable giá trị nhất — bắt đầu từ đây)
```
datasets/
    train/  val/  test/
evaluation/
    medqa/  emergency/  drug_safety/  retrieval/  pii/  jailbreak/
```
- **Versioned, không overwrite**: `drug_safety_v1 -> drug_safety_v2 -> ...`
- **Retrieval ground truth**: mỗi query gắn **gold document(s)**. Không có nó thì Recall@5 / MRR
  KHÔNG tồn tại — nhiều project tưởng benchmark retrieval nhưng thực ra chỉ benchmark answer.
- Kích thước tối thiểu cho safety sets: xem 3.4 Statistical Validity.

- **Data card**: nguồn, license, giới hạn, bias.

**Deliverable:** dataset versioned + validation report + `EDA.ipynb` + Data Quality Report + golden eval sets (versioned) + data card.

---

## Phase X — Prompt Engineering (TRƯỚC Fine-tune)

```
Prompt Template -> System Prompt -> Few-shot -> Reasoning format -> Answer schema
```
- System prompt: persona medical expert, ràng buộc scope, bắt buộc trích nguồn.
- Few-shot + chain-of-thought có cấu trúc + answer schema nhất quán.
- **Prompt-only baseline chạy cùng eval harness với fine-tuned** → biết fine-tune thêm bao nhiêu.
- **Prompt Versioning**: `prompt_v1 -> prompt_v2 -> ...` để khớp với `evaluation_manifest.yaml`.

**Deliverable:** prompt template (versioned) + prompt-only baseline scores.

---

## Phase 2 — Fine-tuning (Unsloth + QLoRA)

- **Base**: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` (4-bit).
- **Tracking bắt buộc**: W&B / MLflow.
- **Sweep nhẹ**: r, alpha, target modules, lr.
- **Benchmark tốc độ**: Unsloth vs baseline.

### Reproducibility bundle (mỗi experiment)
```
Experiment: dataset version · adapter · training config · random seed · commit hash · metrics
```

### Khái niệm
- **QLoRA**: quantize model gốc 4-bit (NF4) + double quant + paged optimizers, LoRA train
  16-bit. Tính toán 16-bit, chỉ lưu trữ 4-bit → gần như không mất chất lượng.
- **Unsloth**: wrapper đơn giản hóa `transformers`+`peft`+`bitsandbytes` + kernel Triton tự
  viết → nhanh ~2x, nhẹ VRAM.

### VRAM model 8B
| Cách | VRAM | Chạy trên |
|---|---|---|
| Full fine-tune | ~60–80GB | Nhiều A100 |
| LoRA (16-bit) | ~18–20GB | 1× A100/3090 |
| QLoRA (4-bit) | ~6–8GB | 1× T4 / Kaggle |

**Deliverable:** adapter + reproducibility bundle + logs + báo cáo tốc độ/VRAM.

---

## Phase 3 — Evaluation (dồn công vào đây)

Tất cả là **offline** (pre-deployment). Online eval → Phase 5.

### 3.1 Benchmark
```
MedQA · PubMedQA · MedMCQA
Run ≥3 seeds (42, 3407, 1234) -> mean ± std -> Confidence Interval (bootstrap)
```
So sánh **base vs prompt-only vs fine-tuned**. `80.7 ±0.2` vs `79.4 ±0.1` mới đáng tin.

### 3.2 Quality + Metric Validation
```
Automatic:  BERTScore · RAGAS (Faithfulness, Answer Relevance, Context Precision, Context Recall) · LLM Judge
Human Review:  Helpfulness · Completeness · Clinical correctness
```
**Metric Validation (RAGAS là proxy, không phải ground truth):**
- Coi RAGAS là **proxy metric** (nó chấm bằng LLM-judge nội bộ → có variance).
- Gán nhãn thủ công một **stratified sample (~200 QA)**.
- Báo cáo tương quan RAGAS vs human (**Spearman/Pearson**).
- **Human annotation là source of truth** cho mọi đánh giá safety-critical.

### 3.3 Retrieval Evaluation
```
Recall@k · MRR · Context Precision · Context Recall
```
Cần **retrieval relevance set** (query → gold passages) từ Phase 1.

**Embedding benchmark (để justify ADR):**
```
BGE-large  vs  E5-large  vs  NV-Embed   trên retrieval_golden_set
-> chọn model có Recall@5 cao nhất -> ghi vào ADR
```

### 3.4 Safety + Statistical Validity
```
General hallucination (<10%) · Drug safety (<5%)
Emergency-advice hallucination (<2%) · Emergency routing recall (>95%)
PII leakage · Prompt injection · Jailbreak · Toxicity
```
**Statistical Validity — sample size tối thiểu:**

| Test | Minimum labeled cases |
|---|---|
| Drug dosage | ≥ 300 |
| Emergency triage | ≥ 300 |
| General hallucination | ≥ 500 |
| Prompt injection | ≥ 200 |
| Jailbreak | ≥ 200 |

> **Rule of three:** 0 lỗi trong n mẫu → cận trên 95% của tỷ lệ thật ≈ 3/n. Claim "< 2%" cần
> n ≥ 150; claim "< 1%" cần n ≥ 300. Đó là gốc của các ngưỡng trên.
>
> *Safety claims are reported with 95% confidence intervals and should not be interpreted from
> small evaluation sets.*

### 3.5 Regression + Forgetting
```
Fixed test suite · General capability · Catastrophic forgetting
```

### 3.6 Failure Attribution (Metric → Diagnosis → Engineering Action)
| Quan sát | Nguyên nhân khả dĩ | Hành động |
|---|---|---|
| Recall@5 ↓ | Retriever kém | đổi embedding, chunking, reranker |
| Recall@5 ↑, Faithfulness ↓ | LLM bỏ qua context | sửa prompt, fine-tune, context formatting |
| Recall↑ Faithfulness↑ Accuracy↓ | Answer synthesis lỗi | output schema, CoT, extraction |
| Accuracy↑ Hallucination↑ | Model suy diễn quá mức | guardrails, stricter prompting |
| Faithfulness↑ Citation↓ | Prompt không ép cite | sửa template |

### 3.7 Error Taxonomy
| Error Type | Example |
|---|---|
| Retrieval Failure | không tìm thấy guideline |
| Context Ignored | retrieve đúng nhưng model không dùng |
| Hallucination | tự bịa thông tin |
| Citation Error | cite sai nguồn |
| Dosage Error | sai liều thuốc |
| Temporal Error | dùng guideline cũ |
| Formatting Error | JSON sai schema |
| Safety Policy Error | không escalate emergency |

Debug workflow: `Accuracy -> Error Breakdown -> Top Failure Types -> Prioritize Fix`

### Evaluation Manifest (reproduce toàn bộ eval bằng 1 file)
```yaml
evaluation_version: 1.2.0
dataset: [medqa_v2, drug_v1, retrieval_v3]
retriever: bge-large
generator: llama3.1-8b-lora-r12
prompt: prompt_v8
seed: 42
metrics: [accuracy, faithfulness, recall@5]
```

**Deliverable:** báo cáo eval + golden eval sets versioned + safety scorecard + `evaluation_manifest.yaml`.

---

## Phase 4 — Serving, RAG & Guardrails (thiết kế đủ, stub phần chạy)

### 4.1 RAG pipeline
```
Medical Guidelines · Drug Database · WHO · CDC · PubMed
  -> Chunking -> Embedding -> Vector Database -> Retriever -> Re-ranking -> Prompt Builder -> LLM
```
Nguồn **versioned** (guideline đổi theo thời gian → tránh Temporal Error). Re-ranking lọc
context nhiễu. Latency budget: retrieval < 200ms, rerank < 100ms, generation < ~1.5s → p95 < 2s.

### 4.2 Guardrail flow + Policy Engine
```
User -> Input Guard -> Retriever -> LLM -> Output Guard -> Policy Engine -> Answer
```
- **Input Guard**: jailbreak, out-of-scope, emergency detection, PII.
- **Output Guard**: hallucination check, PII filter, ép citation.
- **Policy Engine**: `Need doctor? / Need disclaimer? / Need escalation? / Need refuse?`

### 4.3 Serving stack
```
Inference Server -> Rate Limiter -> FastAPI -> vLLM -> Redis Cache -> Logging -> Monitoring
```
**Rate Limiter** bắt buộc: medical API rất dễ bị API abuse / DoS / flood. Kèm auth + per-user quota.
(Redis cache, logging mock trên Kaggle; production để cloud VM / HF Endpoints / k8s.)

**Deliverable:** RAG pipeline + guardrail + Policy Engine + rate-limited API (demo offline).

---

## Phase 5 — Monitoring, Online Eval & MLOps

### Online Evaluation (traffic thật)
```
Live Feedback (thumb up/down) · Latency · Escalation Rate · Citation Rate
Abandonment Rate · Cost/token · Quality (A/B)
```

### System & drift monitoring
```
Latency · GPU utilization · Prompt length · Completion length
Citation rate · Hallucination trend · Safety violations · User feedback
Data drift · Retrieval drift (Recall@5 giảm theo thời gian)
```
> **Retrieval drift** đo bằng **canary retrieval set** chạy định kỳ (không thể relabel traffic
> live), hoặc drift phân bố embedding của query làm proxy.

### Feedback Loop (có Human Review Queue — chặn feedback poisoning)
```
Production -> Feedback -> Human Review Queue -> Approved -> Training Data
```
KHÔNG cho `feedback -> train luôn`: kẻ xấu có thể 👍 câu sai / nhồi nội dung độc để đầu độc
data. Review queue vừa lọc chất lượng vừa là hàng rào an ninh.

### Model Registry (mọi artifact đều version)
```
Model · Adapter · Prompt · Retriever · Embedding · Evaluation Manifest
```
- **CI/CD** — đổi model → tự chạy eval suite trước khi lên (eval gate).

**Deliverable:** thiết kế MLOps + eval-gate CI + dashboard monitoring (mock).

---

## Phase 6 — Governance & Responsible AI

```
Decision Log (ADR) · Known Limitations · Known Failure Modes
Model Risks · Dataset Bias · Responsible AI Statement
```

### Architecture Decision Records (ADR) — trả lời "tại sao", không chỉ "làm gì"
| Quyết định | Chọn | Vì sao |
|---|---|---|
| Facts source | RAG thay vì fine-tune facts | facts đổi theo thời gian, cần trích nguồn, tránh hallucinate liều |
| Chunking | Recursive 512 tokens + overlap 64 | trade-off recall vs context efficiency; (so với semantic chunking: đo trên golden set) |
| Embedding | BGE (điền sau khi benchmark) | Recall@5 cao nhất trên retrieval_golden_set (xem 3.3) |
| Serving | vLLM thay vì TGI | (điền: throughput / paged-attention / license lý do cụ thể) |
| PEFT | QLoRA thay vì full FT | GPU đơn, chi phí, chất lượng gần bằng |

### Known Failure Modes (kiểu Model Card hãng lớn)
```
Rare diseases · Multilingual queries · Very long documents · Conflicting guidelines
```

- Model card + audit logging mọi request.
- Minh bạch Known Limitations / Model Risks / Dataset Bias / Out-of-scope.
- Runbook xử lý sự cố.

**Deliverable:** ADR log, model card (kèm known failure modes), governance docs, runbook.

---

## Lộ trình Milestone (làm kỹ, không gấp)

| Milestone | Thời gian | Nội dung |
|---|---|---|
| **M1** | Tuần 1 | Phase 0 + Phase 1 (data + validation + EDA + khởi tạo golden eval sets) |
| **M2** | Tuần 2 | **Evaluation Harness + Golden Set v1** + Phase X (prompt + baseline scores) |
| **M3** | Tuần 3 | Phase 2 (fine-tune + reproducibility bundle) |
| **M4** | Tuần 4–5 | Phase 3 đầy đủ (benchmark/quality/retrieval/safety/attribution/taxonomy) |
| **M5** | Tuần 6+ | Phase 4 (RAG + guardrails + serving) + Phase 5/6 + ADR |

> Eval Harness là **dependency của mọi thứ sau** → phải xong ở M2 (cùng Golden Set v1) thì
> Phase X mới có baseline scores và Phase 3 mới chạy được.

---

## Stack tổng hợp

| Mảng | Công cụ |
|---|---|
| Fine-tune | Unsloth, PEFT, bitsandbytes, TRL |
| Base model | Llama 3.1 8B Instruct (4-bit) |
| Data | DVC / HF Datasets, Presidio, Great Expectations |
| Tracking | Weights & Biases hoặc MLflow |
| Eval | MedQA, PubMedQA, MedMCQA, RAGAS, BERTScore, LLM-judge |
| Retrieval | Vector DB (FAISS/Chroma), re-ranker, embedding (BGE) |
| Serving | vLLM, FastAPI, Redis (mock), rate limiter |
| Registry/CI | MLflow, GitHub Actions |

---

## Disclaimer

Project nghiên cứu/giáo dục. Model KHÔNG dùng cho chẩn đoán, kê đơn, hoặc quyết định y tế
thật. Mọi output cần bác sĩ có chuyên môn review.
