# ADR-0005: Base model cho y hiện đại tiếng Việt

**Status:** Proposed (chờ benchmark)
**Context:**
Domain = y học HIỆN ĐẠI, ngôn ngữ = tiếng Việt. Trực giác chọn model Việt-centric
(Vistral/PhoGPT/VinaLlama), nhưng với domain chuyên sâu, kiến thức y khoa quan trọng hơn độ trôi chảy tiếng Việt. Lưu ý: lợi thế của Qwen (priors Trung, Hán-Việt) chủ yếu ở Y học CỔ
TRUYỀN; với y HIỆN ĐẠI (thuật ngữ gốc Latin/Anh) lợi thế này yếu đi.

**Decision (đề xuất):**
Mặc định **Llama 3.1 8B Instruct** (đa ngôn ngữ + kiến thức y khoa mạnh từ corpus EN).
Benchmark đối chứng: Qwen2.5-7B, Vistral-7B trên golden set VN (VM14K) trước khi chốt.

**Why:**
- Modern medicine ít Hán-Việt -> Llama không thua thiệt như ở cổ truyền.
- Llama 3.1 hỗ trợ tiếng Việt + kiến thức y khoa từ data EN dồi dào.
- Facts đằng nào cũng đến từ RAG (ADR-0001) -> base model không cần "nhớ" facts VN.

**Consequences:**
- Cần chạy benchmark base model (thêm chi phí M2/M3).
- Nếu tiếng Việt của Llama chưa đủ mượt -> cân nhắc Qwen2.5 hoặc continual-pretrain nhẹ.
- TODO: điền số liệu VM14K cho 3 ứng viên -> chốt.
