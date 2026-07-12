# ADR-0002: Chunking structure-aware 768 / overlap 96

**Status:** Accepted (cập nhật 2026-07 — thay quyết định recursive 512/64 ban đầu)

**Context:**
Corpus RAG có 2 loại đơn vị tri thức rất khác nhau về độ dài & cấu trúc:
- **Q&A ngắn** (vinmec.jsonl, urnus11 medical_qa): mỗi cặp đã là 1 đơn vị ngữ nghĩa trọn
  vẹn, đa số < 800 token. Chunk nhỏ hơn = phá ngữ cảnh Q→A.
- **Tài liệu dài** (bài viết bệnh/thuốc, phác đồ Bộ Y tế PDF): nhiều mục đánh số (1., 1.1,
  I., a)), có bảng liều thuốc. Cắt cứng theo token dễ tách liều khỏi tên thuốc → **sai sót
  y khoa**.

Embedding BGE-M3 (ADR-0003) hỗ trợ tới 8192 token → không bị chặn ở 512.

**Decision:**
1. **Q&A ngắn: KHÔNG chunk** (1 cặp = 1 unit).
2. **Tài liệu dài: chunk STRUCTURE-AWARE**, size **768** token / overlap **96** (12.5%):
   - Tách theo heading (mục đánh số) thành section trước.
   - Section ngắn (≤ size) giữ nguyên; section dài mới recursive (đoạn→câu) + prepend
     heading vào mỗi chunk con.
   - **Bảng liều / danh sách liều: giữ nguyên 1 khối**, không tách dù vượt size.

**Why:**
- 768 > 512 cũ: giữ trọn ngữ cảnh 1 mục lâm sàng, ít cắt vụn; vẫn trong khoảng 500-1000
  thông dụng. Overlap 12.5% chuẩn (10-15%).
- Structure-aware: article_main đã tách sẵn theo mục (đo thực tế max ~800 token → phần lớn
  no-op); giá trị thật của structure-aware là ở PDF phác đồ dài.
- Dosage-guard: ưu tiên an toàn kê đơn hơn là tuân thủ size cứng.

**Consequences:**
- Một số chunk (bảng liều) cố ý vượt 768 token → chấp nhận (đánh dấu khi chunk).
- Đếm token xấp xỉ theo từ × 1.5 (tránh phụ thuộc tokenizer nặng ở bước chunk) → size thực
  tế dao động nhẹ quanh 768.
- TODO: benchmark recall@k trên retrieval golden set với 512 vs 768 vs 1000 khi có
  `evaluation_sets/retrieval/` tiếng Việt.
