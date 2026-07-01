# Model Card — Medical AI Assistant

## Intended use
Tra cứu & giáo dục y khoa có trích nguồn. KHÔNG chẩn đoán/kê đơn/thay bác sĩ.

## Out of scope
Diagnosis · prescription · emergency advice · image interpretation · regulated medical device.

## Training
Llama 3.1 8B + QLoRA (Unsloth). Data: xem data_card.md. Facts qua RAG, không qua weights.

## Evaluation
Benchmark (MedQA/PubMedQA/MedMCQA), quality (RAGAS+human), safety scorecard, retrieval report.

## Known failure modes
- Rare diseases
- Thuật ngữ y khoa dịch sai (do MT của seed data)
- Bệnh/thuốc đặc thù VN thiếu trong nguồn RAG
- Very long documents
- Conflicting guidelines (phác đồ VN vs quốc tế)

## Risks & bias
TODO — dataset bias, specialty skew, hallucination tiers.
