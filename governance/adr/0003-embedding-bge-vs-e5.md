# ADR-0003: Embedding cho retrieval TIẾNG VIỆT

**Status:** Proposed (chờ benchmark)
**Context:**
Query + tài liệu là TIẾNG VIỆT. Embedding tiếng Anh (bge-large-en) cho tiếng Việt -> Recall@5
rất tệ. Retrieval là xương sống của cả hệ -> đây là quyết định sống còn, không tùy chọn.

**Decision (đề xuất):**
Dùng embedding ĐA NGỮ: **BGE-M3** (mặc định), đối chứng **multilingual-e5-large**.
Reranker cũng phải đa ngữ: **bge-reranker-v2-m3** (không dùng bản -en).

**Why:**
Recall@5 cao nhất trên `evaluation_sets/retrieval/` (tài liệu tiếng Việt) — xem
`reports/retrieval_report.md`. Chọn theo SỐ LIỆU, không theo mặc định.

**Consequences:**
- Phải build retrieval golden set tiếng Việt (query -> gold_docs) để benchmark.
- BGE-M3 nặng hơn -> cân nhắc latency budget (retrieval < 200ms).
