"""Phase 1 — Decontamination: loại rò rỉ giữa TRAIN và EVAL.

VÌ SAO SỐNG CÒN: các dataset reasoning y khoa (medical-o1, MedReason, ReasonMed) được xây
TỪ chính câu hỏi của MedQA/MedMCQA/PubMedQA. Nếu train chứa câu trùng eval -> accuracy tăng
GIẢ -> claim "+5% so với base" vô nghĩa.

Cách làm:
  1. Chuẩn hoá câu hỏi (lowercase, bỏ dấu câu, chuẩn hoá khoảng trắng; tiếng Việt: giữ dấu).
  2. Exact match + near-duplicate:
     - n-gram / MinHash Jaccard giữa train question và mọi eval question.
     - (tuỳ chọn) embedding cosine similarity > ngưỡng -> nghi ngờ.
  3. Loại mẫu train trùng với eval (VM14K, MedQA translated...).
  4. Report: bao nhiêu mẫu bị loại, ví dụ cặp trùng.

Chạy TRƯỚC khi train. Đây là điều kiện hợp lệ của toàn bộ Phase 3.
"""

# TODO: implement decontaminate(train, eval_sets, cfg) -> clean_train, report
# TODO: xuất reports/decontamination_report.md
