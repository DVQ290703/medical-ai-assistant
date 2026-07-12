# Plan: Pipeline tri thức nền y khoa VN cho RAG

## Context (vì sao làm)

Hệ RAG y khoa tiếng Việt hiện chỉ có **1 corpus**: `data/raw/vinmec.jsonl` (16k cặp Q&A,
đã ingest + có `embedding.py` đẩy vào Qdrant). Nhưng kiến thức y khoa **tiếng Việt khan
hiếm trong trọng số model** → RAG là thiết yếu, và tri thức nền (bài viết bệnh/thuốc,
phác đồ) mới là cái model thiếu nhất, không chỉ Q&A.

`configs/rag.yaml` liệt kê 5 nguồn mong muốn (Bộ Y tế, Dược thư, phác đồ BV, sách Y, MSD)
nhưng chỉ là **list string chưa cấu trúc**; `chunk.py / vectorstore.py / retriever.py /
reranker.py / citation.py` đều là stub `# TODO`.

**Quyết định đã chốt với user:**
- Mục đích: **học tập/nghiên cứu** (phi thương mại) → dùng data crawl Vinmec rủi ro thấp.
- Nguồn khởi đầu: **`urnus11/Vietnamese-Healthcare`** (HF, tải 1 phát) + **kcb.vn** (PDF
  phác đồ Bộ Y tế — hợp pháp nhất theo nghiên cứu: văn bản hành chính không bị bảo hộ
  bản quyền, Điều 15 Luật SHTT).
- **Phạm vi article: CHỈ split `vinmec_article_main` (138k)** để bắt đầu (nhẹ, cô đọng).
- **Làm luôn kcb.vn trong đợt này** (kb_fetch + loader PDF), không để sau.
- **Q&A: giữ `phuocsang/vinmec-medical-qa` (16k) + GỘP THÊM `urnus11` split `medical_qa`
  (10k)**. Đã đo thực tế: trùng chéo giữa 2 nguồn ≈ 0% (1 dòng), urnus11 tự trùng nội bộ
  3.4% (345 dòng). urnus11 map `title`→question, `content`→answer, giữ `url` để citation.
  → cần **bước dedup bằng answer-fingerprint** (NFC+lowercase, 200 ký tự đầu) khi gộp.
- **KHÔNG** tự crawl: Dược thư QG, MSD Manual, sách giáo khoa Y, phác đồ BV → đều có bản
  quyền / cần xin phép. Chỉ thêm sau nếu có license.

**Kết quả nhắm tới:** khung pipeline nguồn-dài chạy được end-to-end, có metadata để trích
dẫn, khớp pattern repo; nạp được `urnus11/Vietnamese-Healthcare` + phác đồ kcb.vn vào Qdrant.

---

## Hiện trạng (từ khảo sát repo)

- Pattern nên theo: `src/knowledge/embedding.py` — `@dataclass Config` + `config_from_yaml()`
  đọc `configs/rag.yaml`, lazy-import lib nặng, resume-state JSON, ghi/đọc JSONL
  `ensure_ascii=False`.
- Schema RAG hiện tại: `{question, answer}` + `field_map` từ config (`src/knowledge/ingest.py`).
- `chunk/vectorstore/retriever/reranker/citation` = stub. `utils/io.py` = stub.
- Chưa có lib PDF/HTML trong bất kỳ requirements nào.
- Không có DVC stage cho knowledge (chạy tay `python -m src.knowledge.X`).

## Dataset đã kiểm chứng: `urnus11/Vietnamese-Healthcare`
- Gated (cần Agree + HF_TOKEN — đã có sẵn), 1 config `default`, schema `{url, title, content}`.
- 5 split: `medical_qa` (10k Q&A, **NGẮN → không chunk**), `vinmec_article_main` (138k),
  `vinmec_article_content` (32k), `vinmec_article_subtitle` (163k), `full` (173k gộp).
  → article là **văn bản DÀI → CẦN chunk** (khác vinmec.jsonl Q&A ngắn).

---

## Thiết kế

Nguyên tắc: **tách 2 loại đơn vị tri thức**, không ép chung một schema.
- **Q&A ngắn** (vinmec.jsonl, split medical_qa): 1 cặp = 1 unit, **không chunk**.
- **Article/phác đồ dài** (vinmec_article_*, PDF kcb.vn): **chunk theo cấu trúc**.

Schema chuẩn hoá nội bộ cho tri thức nền (JSONL ở `data/raw/kb/`):
```
{doc_id, source, title, url, section, text, meta}
```
`source` (vd "vinmec-article", "byt-kcb"), `url` để **trích dẫn** → phục vụ `citation.py`.

### Các file sẽ tạo/sửa

**1. `configs/rag.yaml`** — cấu trúc hoá `knowledge_sources` thành list có schema (giống
khối `corpus` đã có), mỗi nguồn: `id, type (hf|pdf|html), source, splits/paths, chunk (bool),
out`. Giữ list string mô tả cũ thành comment. Thêm nhánh `chunk` đã có sẵn (size/overlap).

**2. `src/knowledge/kb_ingest.py`** (mới) — nạp nguồn tri thức nền → JSONL chuẩn hoá.
   - Đọc `configs/rag.yaml` theo pattern `config_from_yaml` của embedding.py.
   - Loader theo `type`:
     - `hf`: dùng `datasets.load_dataset` (như ingest.py) — nạp split `vinmec_article_main`
       của `urnus11/Vietnamese-Healthcare`, map `{url,title,content}` → schema chuẩn.
     - `pdf`: lazy-import `pymupdf` (fitz), trích text theo trang + giữ heading. Phát hiện
       scan (text/trang < ngưỡng) → cảnh báo, bỏ qua.
     - `html`: lazy-import `trafilatura`/`bs4` cho nguồn web (để dành, chưa bật).
   - CLI `python -m src.knowledge.kb_ingest --source <id> [--inspect] [--limit N]`.
   - Reuse `authenticate()` + `.env` loader từ `ingest.py` (HF_TOKEN).

**2b. `src/knowledge/qa_ingest.py`** (mới, hoặc mở rộng `ingest.py`) — gộp Q&A đa nguồn +
   dedup. Nạp `urnus11medical_qa` (title→question, content→answer, +url), rồi **dedup bằng
   answer-fingerprint** (NFC+lowercase, 200 ký tự đầu) so với `data/raw/vinmec.jsonl` đã có
   và nội bộ chính nó. Ghi bổ sung vào corpus Q&A (giữ nguyên phuocsang, thêm phần mới).
   Reuse pattern `_norm_q()` trong `src/data/validate.py`.

**3. `src/knowledge/chunk.py`** (thay stub) — chunk cho văn bản DÀI.
   - `strategy: recursive` size 512 / overlap 64 (theo `rag.yaml` + ADR-0002) làm mặc định.
   - Chunk **theo cấu trúc trước, cắt token sau**: tách theo heading/đoạn, chỉ cắt cứng khi
     một đoạn vượt size. Giữ `title`+`section` vào mỗi chunk (ngữ cảnh + citation).
   - Nhận JSONL chuẩn hoá từ kb_ingest → xuất JSONL chunk `{doc_id, chunk_id, source, title,
     url, section, text}`. Hàm gọi trực tiếp được (notebook) + CLI.
   - Lazy-import `langchain_text_splitters.RecursiveCharacterTextSplitter` (hoặc tự viết
     recursive splitter nhẹ nếu muốn tránh dep — quyết định lúc code, ưu tiên tránh dep nặng).

**4. `src/knowledge/embedding.py`** (sửa nhẹ) — hiện hardcode 2 collection Q&A. Thêm khả năng
   embed **collection tri thức nền** từ JSONL chunk: field embed = `text` (thay vì question/
   answer), payload mang `title/url/section` để citation. Giữ nguyên hybrid BGE-M3 + resume.
   → thêm collection `vinmec_kb` (article) và cho phép nguồn kcb.vn dùng chung đường ống.

**5. `governance/adr/0002-chunking-512-overlap-64.md`** (cập nhật) — bổ sung Decision: chunk
   **theo cấu trúc** cho tài liệu dài, Q&A ngắn **không chunk**. Điền phần TODO còn trống.

**6. `governance/` — bảng license nguồn** (mới, `governance/knowledge_sources_license.md`):
   ghi rõ mỗi nguồn: license/tình trạng bản quyền + được index hay không (từ nghiên cứu).
   Chặn nhầm lẫn crawl nguồn có bản quyền (Dược thư/MSD/sách Y).

**7. `requirements.txt`** (thêm) — `qdrant-client`, `FlagEmbedding` (đang dùng nhưng chưa
   khai báo), `pymupdf` (parse PDF kcb.vn), `requests` đã có. `langchain-text-splitters`
   nếu chọn dùng. **User tự cài vào venv** (đã thống nhất — tôi không tự pip install).

**8. `src/knowledge/kb_fetch.py`** (mới, tùy chọn — crawl kcb.vn có văn hoá) — tải PDF phác
   đồ Bộ Y tế từ kcb.vn: User-Agent rõ ràng, rate-limit (robots chưa xác nhận → mặc định
   lịch sự, delay), lưu về `data/raw/kb/pdf/`. Chỉ nguồn kcb.vn (hợp pháp). **Không** crawl
   MSD/Dược thư.

### Thứ tự triển khai (từng bước chạy được)
1. `requirements.txt` + `governance` license table + cấu trúc `rag.yaml` (nền).
2. `kb_ingest.py` cho nguồn `hf` → nạp **CHỈ split `vinmec_article_main` (138k)** của
   `urnus11/Vietnamese-Healthcare` → `data/raw/kb/vn_healthcare.jsonl`.
3. `chunk.py` → chunk article JSONL.
4. `embedding.py` mở rộng → embed chunk vào Qdrant collection `vinmec_kb`.
5. `kb_fetch.py` (tải PDF kcb.vn có văn hoá) + loader `pdf` trong `kb_ingest.py` → nạp
   phác đồ Bộ Y tế → cùng schema → chunk → embed (chung collection `vinmec_kb`, phân biệt
   bằng `source`).

---

## Verification (test end-to-end)

- **kb_ingest (hf):** `python -m src.knowledge.kb_ingest --source vietnamese-healthcare
  --inspect --limit 20` → in schema `{url,title,content}`, xác nhận map đúng. Sau đó chạy
  thật ghi `data/raw/kb/vn_healthcare.jsonl`, kiểm số dòng khớp split.
- **chunk:** chạy trên ~50 article, in phân bố độ dài chunk (median/p90/max token) để xác
  nhận không có chunk vượt size, và bảng liều/heading không bị cắt giữa chừng (spot-check tay).
- **embedding:** cần Docker Qdrant chạy + GPU. `python -m src.knowledge.embedding
  --collections kb` (thêm alias) → embed vài batch, `client.get_collection('vinmec_kb')`
  báo points_count > 0. Xác minh payload có `title/url` để citation.
- **kb_fetch (kcb.vn):** tải 1 PDF mẫu đã biết (HDĐT ung thư vú) → mở bằng pymupdf, in
  1000 ký tự đầu để xác nhận trích text được (không phải scan).
- Chạy thử nhỏ bằng `--limit` trước khi chạy full 334k article (embed nặng, cần GPU).

## Rủi ro / lưu ý
- Chỉ nạp `vinmec_article_main` (138k) — chunk ra ước tính vài trăm nghìn vector, đủ nặng;
  chạy `--limit` thử trước rồi mới full. GPU gần như bắt buộc cho bước embed.
- kcb.vn: robots.txt của chính kcb.vn **chưa xác nhận** (agent mới thấy footer + tải được
  PDF). `kb_fetch.py` mặc định lịch sự: User-Agent rõ, delay giữa request, đọc robots trước
  khi tải hàng loạt. Chỉ tải kcb.vn, không đụng nguồn có bản quyền.
- License: chỉ index nguồn đã xác nhận (Vietnamese-Healthcare cho học tập, kcb.vn). Bảng
  license trong governance là nguồn sự thật để tránh index nhầm nguồn có bản quyền.
- `device: cuda` trong rag.yaml — nếu máy user không có GPU phải đổi `cpu` (embed sẽ chậm).
- Một số PDF kcb.vn có thể là scan (không trích được text) → kb_fetch/loader phải phát hiện
  và cảnh báo (spot-check: nếu trích được < N ký tự/trang thì đánh dấu "cần OCR", bỏ qua).
```
