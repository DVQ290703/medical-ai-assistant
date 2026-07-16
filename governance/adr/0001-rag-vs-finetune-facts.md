# ADR-0001: RAG cho facts, không fine-tune facts

**Status:** Accepted

**Context:**
Trợ lý Q&A y học phải trả lời đúng các **fact có rủi ro cao**: liều thuốc, chống chỉ định, phác
đồ điều trị, ngưỡng chẩn đoán. Có hai cách để model "biết" những fact này:
- **Nhồi fact vào trọng số qua fine-tune** — model học thuộc từ dữ liệu train.
- **Để fact ngoài model, truy xuất lúc chạy qua RAG** — model chỉ đọc và tổng hợp từ nguồn.

Ràng buộc của bài toán khiến lựa chọn không trung lập:
- **Fact y khoa đổi theo thời gian**: phác đồ Bộ Y tế cập nhật, liều thuốc điều chỉnh. Fact đã
  nhồi vào trọng số thì "đóng băng" tại thời điểm train → sai lệch âm thầm (Temporal Error).
- **Yêu cầu trích dẫn kiểm chứng được**: câu trả lời y tế phải truy về tài liệu gốc. Fact sinh
  từ trọng số không có nguồn để dẫn.
- **Hallucinate liều thuốc = rủi ro pháp lý và an toàn**: model ngôn ngữ bịa số liều nghe rất
  "trơn tru". Không thể chấp nhận trong miền y khoa.
- **Tiếng Việt low-resource**: datang train y khoa VN ít và không đủ phủ; ép fine-tune học fact
  từ tập nhỏ càng dễ nhớ sai/nhớ lệch.

**Decision:**
Tách bạch hai nguồn tri thức:
1. **Facts (liều, chống chỉ định, guideline, ngưỡng...) → lấy từ RAG có trích dẫn.** Model chỉ
   được trả lời fact dựa trên context truy xuất; không đủ nguồn → nói "không tìm thấy" chứ không
   bịa (khớp threshold ở retriever, `citation_required: true` ở serving).
2. **Fine-tune (QLoRA/Unsloth) chỉ dạy *reasoning, văn phong, cách trích dẫn*** — tức là *cách*
   suy luận và trình bày trên context, không phải *nội dung* fact.

**Why:**
- Fact ở ngoài model → **cập nhật corpus là đủ**, không phải train lại → luôn theo kịp guideline mới.
- Mọi fact **truy vết được về nguồn** → đáp ứng yêu cầu trích dẫn, kiểm chứng được.
- **Giảm mạnh rủi ro hallucinate liều**: model không được "nhớ" liều, chỉ được đọc từ tài liệu
  đã kiểm duyệt; thiếu nguồn thì từ chối.
- Phân vai đúng thế mạnh: LLM giỏi *lập luận/diễn đạt*, hệ retrieval giỏi *tra cứu chính xác* →
  fine-tune tập trung vào cái LLM thực sự cần học, tránh lãng phí dữ liệu VN ít ỏi để học thuộc fact.

**Consequences:**
- **Chất lượng câu trả lời phụ thuộc chất lượng retrieval**: retrieval trượt (miss) → thiếu fact
  hoặc trả rỗng. Đẩy trọng tâm kỹ thuật sang RAG (hybrid + rerank + threshold, xem ADR-0002/0003)
  và sang đo retrieval (recall@k trên golden set).
- **Bắt buộc trích dẫn** → thêm tầng citation và output guard; câu trả lời không dẫn được nguồn
  bị coi là lỗi, không phát cho người dùng.
- **Corpus trở thành tài sản cần quản trị**: phải version, cấp phép, cập nhật nguồn (`data/`,
  DVC, `governance/knowledge_sources_license.md`).
- **Không kỳ vọng model trả lời fact khi offline/không có RAG** — đó là thiết kế, không phải khiếm khuyết.
- Đánh đổi độ trễ: mỗi query tốn thêm bước encode + retrieve + rerank so với sinh thẳng từ trọng số.
