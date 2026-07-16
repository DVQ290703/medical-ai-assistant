# ADR-0004: Serving inference engine = vLLM

**Status:** Accepted

**Context:**
Tầng serving cần host model Llama 3.1 8B (base + LoRA adapter fine-tune) cho inference online,
với ràng buộc:
- **Đơn GPU, VRAM hạn chế** (Kaggle/Colab T4-16GB hoặc 1×A10/A100): phải chạy được model 8B +
  KV-cache cho nhiều request đồng thời mà không OOM.
- **Nhiều request song song**: Q&A y khoa có prompt dài (context RAG nhiều chunk) → KV-cache
  phình nhanh, cấp phát KV-cache ngây thơ lãng phí VRAM và giới hạn số request đồng thời.
- **Nạp được LoRA adapter** từ `artifacts/adapters` mà không phải merge cứng vào base mỗi lần.
- **OpenAI-compatible API**: để phần orchestrator/generation gọi qua HTTP đồng nhất, đổi backend
  dễ (xem `configs/serving.yaml`).

Hai ứng viên chính: **vLLM** và **TGI** (Text Generation Inference của HuggingFace).

**Decision:**
Dùng **vLLM** làm inference engine cho deploy có GPU (`engine: vllm` trong `configs/serving.yaml`).
Có **fallback hai lớp** cho môi trường không đủ điều kiện:
- Không có GPU khỏe / model fine-tune chưa xong → gọi **Groq API** (`generation.backend: groq`).
- Chạy nhanh trên Kaggle → fallback `transformers.generate` (`local` backend).

**Why:**
- **PagedAttention** — vLLM quản lý KV-cache theo "trang" như virtual memory, gần như không phân
  mảnh; nhồi được nhiều request đồng thời hơn trên cùng VRAM. Đây là lợi thế quyết định với đơn
  GPU hạn chế.
- **Continuous batching** — ghép/tách request theo từng bước decode thay vì batch tĩnh → throughput
  cao hơn rõ khi tải không đều (đúng với pattern hỏi–đáp lẻ tẻ).
- **LoRA runtime** — vLLM nạp adapter lúc chạy (`--enable-lora`), không phải merge vào base; hợp
  với vòng lặp fine-tune → thử adapter nhanh.
- **OpenAI-compatible server** sẵn có → orchestrator gọi đồng nhất, đổi sang Groq/local chỉ là đổi
  config, không sửa code.
- **License Apache-2.0**, cộng đồng lớn, tài liệu đầy đủ.
- So với TGI: TGI mạnh và production-ready, nhưng license (BFHL, đã đổi vài lần) và việc gắn với
  hệ sinh thái HF làm nó kém linh hoạt hơn cho một dự án research đơn GPU; PagedAttention của vLLM
  hợp với ràng buộc VRAM ở đây hơn.

**Consequences:**
- **Phụ thuộc GPU + CUDA của vLLM**: không chạy được thuần CPU. Vì vậy mới cần fallback Groq/
  transformers ở trên — trade-off là code serving phải hỗ trợ nhiều backend (thêm độ phức tạp,
  bù lại chạy được ở mọi môi trường).
- **Chưa benchmark định lượng** vLLM vs TGI trên chính corpus này (throughput/latency thực tế) —
  quyết định dựa trên đặc tính kiến trúc, không phải số đo tại chỗ.
- **Chưa dùng tính năng nâng cao** (speculative decoding, prefix caching cho system prompt chung) —
  ghi nhận là hướng tối ưu về sau nếu cần giảm latency.
- TODO: đo latency p50/p95 và throughput khi có GPU deploy thật + adapter fine-tune hoàn chỉnh,
  để xác nhận lựa chọn bằng số liệu.
