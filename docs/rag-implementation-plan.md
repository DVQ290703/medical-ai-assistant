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

---

# PHẦN 2: RETRIEVER (hybrid + RRF + rerank + source priority)

## Context (vì sao)
Pipeline đã embed 3 collection vào Qdrant (`vinmec_kb`, `vinmec_q`, `vinmec_qa`). Nhưng
`vectorstore.py / retriever.py / reranker.py` còn stub → chưa truy vấn được. Đây là mảnh
làm RAG thật sự trả lời được. User lo "phủ rộng → trả lời sai" — giải quyết bằng rerank +
score threshold + source priority + citation, KHÔNG bằng thu hẹp nguồn (giảm recall).

## Quyết định đã chốt với user
- **Retrieve từ:** `vinmec_kb` + `vinmec_qa`. **BỎ `vinmec_q`** (trùng data qa, chỉ khác
  embed key → gây kết quả lặp). vinmec_q giữ lại chỉ để benchmark sau, không dùng runtime.
- **Ưu tiên nguồn uy tín khi trùng chủ đề:** `byt-kcb` (phác đồ Bộ Y tế) > `vinmec-article`
  > Q&A vinmec. Field `source` đã có sẵn trong payload mỗi point.
- **Chống trả lời sai:** rerank cross-encoder + score threshold (dưới ngưỡng → trả rỗng,
  "không tìm thấy thông tin") + citation bắt buộc.

## API đã xác minh (Qdrant 1.18 + FlagEmbedding trong venv)
- `client.query_points(...)` + `models.Prefetch` + `models.FusionQuery(fusion=Fusion.RRF)`
  → hybrid dense+sparse + RRF trong 1 lần gọi. ✅
- `FlagEmbedding.FlagReranker("BAAI/bge-reranker-v2-m3")` — cross-encoder rerank. ✅

## Thiết kế

Luồng: `query → BGE-M3 hybrid encode → Qdrant query_points (Prefetch dense + sparse,
RRF fuse) → top_k candidate → rerank → source-priority tie-break → threshold → top_n + citation`

### Các file (thay stub)
**1. `src/knowledge/vectorstore.py`** — wrapper Qdrant:
   - `connect()` (reuse pattern `_connect_qdrant` từ embedding.py), `hybrid_search(dense,
     sparse, collections, top_k)`.
   - Dùng `query_points` với `Prefetch(query=dense, using="dense", limit=k)` +
     `Prefetch(query=SparseVector, using="sparse", limit=k)` + `FusionQuery(RRF)`.
   - Search nhiều collection (kb + qa) → gộp candidate.

**2. `src/knowledge/reranker.py`** — cross-encoder:
   - Lazy-load `FlagReranker(cfg.reranker.model, use_fp16)`. Rerank cặp (query, chunk.text).
   - Device: reranker nhỏ hơn BGE-M3, **chạy CPU được** (chỉ rerank ~16 candidate) → không
     vướng GPU 4GB. Config `reranker.device` (mặc định cpu cho máy user).

**3. `src/knowledge/retriever.py`** — orchestrate:
   - `Retriever(cfg)`: nạp BGE-M3 (encode query) + vectorstore + reranker.
   - `retrieve(query) -> list[Hit]`:
     1. encode query (dense+sparse).
     2. `hybrid_search` trên [vinmec_kb, vinmec_qa] → top_k (rag.yaml, mặc định 8) mỗi coll.
     3. rerank toàn bộ candidate → điểm relevance.
     4. **source priority**: cộng bonus nhỏ theo `source` (byt-kcb > vinmec-article > qa)
        để tie-break khi điểm rerank sát nhau (không lấn át relevance).
     5. **threshold**: loại hit dưới `min_score`; nếu rỗng → trả [] (caller báo "không có").
     6. trả top_n (mặc định 4) kèm payload (text, title, url, source) cho citation.
   - `Hit` dataclass: {text, score, source, title, url, chunk_id}.
   - Encode query cần BGE-M3 → cũng vướng GPU 4GB. Giải: config `query_device`; hoặc
     tài liệu hoá rằng retriever chạy nơi có GPU. (Với 1 query, CPU encode cũng chịu được.)

### Config thêm vào rag.yaml (khối retriever/reranker đã có, bổ sung)
```
retriever: { top_k: 8, top_n: 4, min_score: <ngưỡng>, collections: [vinmec_kb, vinmec_qa] }
reranker:  { model: BAAI/bge-reranker-v2-m3, device: cpu, use_fp16: false }
source_priority: { byt-kcb: 2, vinmec-article: 1, "": 0 }   # bonus tie-break
```
`min_score` cần calibrate bằng vài query thật (chưa biết trước → để mặc định thấp, tinh
sau khi test).

## Verification
- Cần Qdrant đang chạy + đã index (ít nhất vài trăm point để test thật).
- `test_retriever.py` (pytest, mark skip nếu Qdrant/model không có):
  - encode query "đau dạ dày uống thuốc gì" → có hit, hit[0].score hợp lý.
  - query vô nghĩa "xyz qwerty 123" → sau threshold trả [] (không bịa).
  - query về chủ đề có trong phác đồ → byt-kcb được ưu tiên lên đầu khi điểm sát.
- Script tay `scripts/dev/` (hoặc reuse check_qdrant.py) chạy 3-5 query thật, in top_n +
  source + score để mắt thường đánh giá + calibrate min_score.
- Đơn vị logic thuần (source priority sort, threshold filter) test được KHÔNG cần Qdrant.

## Rủi ro / lưu ý
- GPU 4GB: encode query bằng BGE-M3 vẫn vướng như embed. Reranker thì CPU OK. Nếu máy user
  không encode nổi query → retriever cũng phải chạy nơi có GPU (hoặc CPU chậm cho 1 query).
- `min_score` threshold: đặt cao quá → hay trả rỗng (bỏ sót); thấp quá → lọt nhiễu. Phải
  calibrate bằng query thật, không hardcode mù. Bắt đầu thấp, siết dần.
- source priority là bonus NHỎ (tie-break), không được lấn át điểm rerank — nếu không phác
  đồ sẽ luôn lên đầu kể cả khi không liên quan bằng.

---

# PHẦN 3: GENERATION (nối retriever -> prompt -> LLM trả lời + citation)

## Context (vì sao)
Retrieval xong (Qdrant có 152k article + 25k Q&A; phác đồ tạm gác vì lỗi version torch Colab).
`src/generation/*`, `src/prompting/*`, `src/serving/citation.py`, `guards/*` đều stub. Cần
nối: query -> retrieve chunk -> build prompt (context + citation) -> LLM sinh câu trả lời.

**Quyết định đã chốt:**
- Model fine-tune CHƯA xong + GPU 4GB không chạy Llama 8B -> **engine LINH HOẠT**: backend
  API (chạy ngay) HOẶC local transformers (khi có GPU/adapter). Chọn qua config.
- API tạm: **Groq** (free, nhanh, Llama 3.x). Gọi qua `requests` (không bắt buộc SDK groq).
- Observability (Langfuse): GÁC LẠI — chỉ để hook log sẵn, gắn sau.

## Thiết kế (theo pattern config-driven của embedding.py/retriever.py)

Luồng: `query -> Retriever.retrieve() -> hits -> build_prompt(context+citation) -> engine.generate() -> answer + nguồn`

### Các file (thay stub)
**1. `configs/serving.yaml`** (bổ sung) — khối `generation`:
```
generation:
  backend: groq            # groq | local
  model: llama-3.3-70b-versatile   # model Groq
  temperature: 0.2         # y khoa -> thấp, bám context
  max_tokens: 1024
  api_key_env: GROQ_API_KEY   # đọc từ .env
  # local (khi có GPU + adapter fine-tune):
  local_model: unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
  adapter_path: artifacts/adapters   # LoRA nếu có
citation_required: true
```

**2. `src/generation/engine.py`** — abstraction backend:
   - `class Engine` (base) + `GroqEngine` (gọi API qua requests) + `LocalEngine` (lazy
     transformers/unsloth, chỉ load khi backend=local).
   - `engine_from_config(cfg)` factory đọc serving.yaml -> chọn backend.
   - `generate(system, user) -> str`. GroqEngine: POST api.groq.com/openai/v1/chat/completions.
   - Log structured mỗi call (model, latency, token) — chỗ sau gắn Langfuse.

**3. `src/prompting/template.py`** — load system prompt từ `prompts/` (versioned). Tạo
   `prompts/system_prompt_v1.txt`: vai trò trợ lý y tế VN, CHỈ trả lời dựa trên context,
   không bịa, khuyên đi khám khi nghiêm trọng, luôn kèm nguồn.

**4. `src/prompting/builder.py`** — `build_prompt(query, hits, cfg)`:
   - Format context từ hits: đánh số [1][2]... kèm title/url/source.
   - Ghép system + context + query. Chèn chỉ dẫn "trích [số] khi dùng thông tin".

**5. `src/serving/citation.py`** — `attach_citations(answer, hits)`: map [số] trong answer
   -> nguồn (title/url). Nếu `citation_required` mà answer không có [số] nào -> cảnh báo/append
   danh sách nguồn đã dùng.

**6. `src/generation/inference.py`** — `answer(query) -> {answer, sources}`: hàm cấp cao nối
   Retriever + builder + engine + citation. Đây là API chính cho serving/CLI.
   - Nếu retrieve trả [] (dưới threshold) -> trả "không đủ thông tin, hãy đi khám" (không gọi LLM bịa).

**7. Guards (tối thiểu, an toàn y khoa)** — `src/serving/guards/input_guard.py`:
   - `emergency_check(query)`: regex red-flag ("đau ngực dữ dội", "khó thở", "co giật",
     "ngất", "chảy máu không cầm"...) -> trả cảnh báo GỌI CẤP CỨU 115 NGAY, không cần chờ LLM.
   - Đây là LUẬT CỨNG ở tầng app (không dựa RAG) — đúng như user nêu "biết khi nào đi viện gấp".

### CLI test
`src/generation/inference.py` có `main()`: `python -m src.generation.inference "câu hỏi"`
-> in answer + nguồn. Đây là cách test end-to-end nhanh (chưa cần FastAPI).

## Verification
- Cần Qdrant chạy + GROQ_API_KEY trong .env.
- `python -m src.generation.inference "đau dạ dày uống thuốc gì"` -> câu trả lời tiếng Việt
  bám context + có [số] trích nguồn + list nguồn (url).
- Query vô nghĩa -> "không đủ thông tin" (không bịa).
- Query cấp cứu ("đau ngực dữ dội lan tay trái") -> cảnh báo 115 NGAY (emergency_check chặn trước).
- Test pytest logic thuần: build_prompt format đúng, citation map đúng, emergency regex khớp
  (không cần API/Qdrant).

## Rủi ro / lưu ý
- GROQ free tier có rate limit -> engine bắt lỗi 429, báo rõ.
- Groq đôi khi đổi tên model -> để model trong config, dễ đổi.
- LocalEngine (transformers/unsloth) KHÔNG chạy nổi trên GPU 4GB máy user -> chỉ dùng khi
  deploy nơi có GPU; mặc định backend=groq.
- emergency_check là AN TOÀN TỐI THIỂU, không thay thế phán đoán y tế — ghi rõ disclaimer.

---

# PHẦN 4: MODEL SERVER trên Colab (encode + rerank qua ngrok)

## Context (vì sao)
Máy user GPU 4GB / RAM eo hẹp: load 2 model (BGE-M3 encode + bge-reranker-v2-m3) cùng lúc
-> `OSError paging file too small` (hết RAM). Retriever không chạy nổi local.

Giải pháp user chọn: **Colab (GPU T4 16GB) làm MODEL SERVER** cho 2 model nặng, expose qua
ngrok. Máy user giữ Qdrant + retriever orchestration + generation (đều nhẹ), gọi HTTP tới
Colab cho 2 bước cần GPU.

Nguyên tắc: giữ nguyên interface retriever (`_encode_query` trả dense/sparse; reranker.score
trả list điểm) — chỉ đổi NGUỒN từ local model sang HTTP call. Retriever không cần biết model
ở đâu.

## Thiết kế

```
Máy user: query
  -> POST colab/encode {query}  -> {dense, sparse}   (Colab: BGE-M3)   ← qua ngrok
  -> Qdrant hybrid search (máy user, nhẹ)
  -> POST colab/rerank {query, texts} -> {scores}    (Colab: reranker) ← qua ngrok
  -> rank + threshold + source priority (máy user)
  -> Groq generate (máy user)
```

### Các file

**1. `notebooks/model_server_colab.ipynb`** (mới) — self-contained, chạy trên Colab:
   - Cài fastapi, uvicorn, pyngrok, FlagEmbedding.
   - Load BGE-M3 + FlagReranker 1 lần (GPU). LƯU Ý thứ tự import + bỏ HF_TOKEN lỗi (đã gặp).
   - FastAPI 2 endpoint:
     - `POST /encode {query} -> {dense: [float], sparse: {indices, values}}`
     - `POST /rerank {query, texts: [str]} -> {scores: [float]}`
     - `GET /health`.
   - Bảo vệ nhẹ: 1 secret token header (tránh ai cũng gọi được ngrok URL công khai).
   - ngrok expose -> in ra public URL để dán vào máy user.

**2. `configs/rag.yaml`** (sửa khối retriever/reranker) — thêm chế độ remote:
```
retriever:
  ...
  encoder_backend: remote     # local | remote
  remote_url: ""              # ngrok URL, vd https://xxx.ngrok-free.app (để .env override)
  remote_token: ""            # khớp secret Colab
reranker:
  backend: remote             # local | remote
```
remote_url/token nên đọc từ .env (RAG_REMOTE_URL, RAG_REMOTE_TOKEN) để không hardcode +
đổi mỗi session Colab dễ.

**3. `src/knowledge/retriever.py`** (sửa `_encode_query`) — nếu encoder_backend=remote:
   POST {remote_url}/encode, trả dense/sparse từ JSON. Local giữ nguyên nhánh cũ.

**4. `src/knowledge/reranker.py`** (thêm nhánh remote) — Reranker.score: nếu backend=remote
   POST {remote_url}/rerank, trả scores. Local giữ CrossEncoder cũ.

### Đọc config remote
Thêm vào RetrieverConfig: encoder_backend, reranker_backend, remote_url, remote_token
(đọc rag.yaml + override bằng env RAG_REMOTE_URL / RAG_REMOTE_TOKEN).

## Verification
- Chạy notebook Colab -> lấy ngrok URL + token. Set vào .env máy user:
  `RAG_REMOTE_URL=https://xxx.ngrok-free.app` , `RAG_REMOTE_TOKEN=...`
- `curl {url}/health` -> ok.
- Máy user (Qdrant chạy): `python -m src.knowledge.retriever "đau dạ dày uống thuốc gì"`
  -> encode + rerank chạy trên Colab, trả hits đúng. RAM máy user KHÔNG tăng vọt (không load model).
- `python -m src.generation.inference "..."` -> full RAG: remote encode/rerank + Groq gen.
- Test local vẫn phải chạy được khi encoder_backend=local (không phá nhánh cũ).

## Rủi ro / lưu ý
- ngrok URL ĐỔI mỗi lần restart Colab -> cập nhật .env. (ngrok free: 1 tunnel, URL ngẫu nhiên.)
- Colab session 12h + hay disconnect -> model server chết -> máy user gọi lỗi. Cần bắt lỗi
  kết nối, báo rõ "model server (Colab) không phản hồi — kiểm tra notebook còn chạy?".
- Bảo mật: ngrok URL công khai -> BẮT BUỘC secret token header, không để lộ.
- Latency: mỗi query 2 round-trip tới Colab -> chậm hơn local ~vài trăm ms. Chấp nhận được.
- Đây là giải pháp TẠM cho máy yếu / demo. Production thì host model server ổn định (không Colab).

---

# PHẦN 5: FastAPI /chat + giao diện web chat

## Context (vì sao)
RAG chạy end-to-end qua CLI (`inference.answer()` trả `Answer{text, sources, kind}`). Nhưng
giao diện web không gọi CLI được -> cần HTTP API. `src/serving/app.py` còn stub. User chọn:
**FastAPI /chat + web tĩnh HTML/JS** gọi API.

## Thiết kế

```
Trình duyệt (web/index.html + chat.js)
   └── POST /chat {message}  ──> FastAPI (app.py) ──> inference.answer() ──> RAG
                              <── {answer, sources, kind}
```

### Các file

**1. `src/serving/app.py`** (thay stub) — FastAPI:
   - `POST /chat {message: str} -> {answer, sources: [{n,title,url,source}], kind}`.
     Gọi `inference.answer(message)`. kind = normal|emergency|no_info.
   - `GET /health -> {ok}`.
   - Serve web tĩnh: mount `web/` (StaticFiles) tại `/` -> mở trình duyệt là thấy chat.
   - CORS mở (demo). Rate-limit đơn giản theo IP dùng `rate_limit` trong serving.yaml
     (per_minute/per_day) — dict đếm in-memory (đủ cho demo; production dùng redis như config ghi).
   - Đọc host/port từ serving.yaml. `main()` chạy uvicorn.
   - Warm-up: khởi tạo Retriever + engine 1 lần lúc startup (tránh load lần đầu chậm).

**2. `web/index.html` + `web/chat.js` + `web/style.css`** (mới) — chat UI tĩnh:
   - Khung chat đơn giản: ô nhập + nút gửi + khung hội thoại.
   - JS fetch POST /chat, render câu trả lời + nguồn (link url) + disclaimer.
   - kind=emergency -> hiện băng đỏ cảnh báo 115 nổi bật.
   - kind=no_info -> hiện nhẹ nhàng "chưa đủ thông tin".
   - Tiếng Việt, gọn, không framework (chỉ HTML/CSS/JS thuần).

**3. `scripts/serve.sh`** (đã có, sửa) — chạy `python -m src.serving.app`.

### CLI/chạy
`python -m src.serving.app` -> uvicorn tại localhost:8000. Mở http://localhost:8000 -> chat UI.

## Verification
- Cần: Qdrant chạy + Colab model server chạy (.env RAG_REMOTE_URL) + GROQ_API_KEY.
- `python -m src.serving.app` -> log "Uvicorn running on :8000".
- `curl -X POST localhost:8000/chat -H "Content-Type: application/json" -d '{"message":"đau dạ dày uống thuốc gì"}'`
  -> JSON {answer, sources, kind:normal}.
- Mở http://localhost:8000 -> gõ câu hỏi -> thấy trả lời + nguồn.
- Câu cấp cứu -> băng đỏ 115. Câu vô nghĩa -> "chưa đủ thông tin".
- Rate-limit: gọi quá per_minute -> HTTP 429.

## Rủi ro / lưu ý
- app.py phụ thuộc toàn chuỗi (Qdrant + Colab remote + Groq) -> nếu 1 mắt xích chết, /chat
  trả lỗi rõ ràng (không treo). Bọc try/except, trả 503 + thông báo.
- Rate-limit in-memory reset khi restart app — đủ cho demo, không phải chống DoS thật.
- CORS mở * chỉ hợp demo local; siết lại nếu deploy công khai.
- Warm-up lúc startup: nếu Colab/Qdrant chưa sẵn sàng, startup vẫn chạy nhưng /chat đầu tiên
  sẽ lỗi -> lazy init trong answer() đã xử; warm-up chỉ để nhanh hơn, bọc try/except.

---

# PHẦN 6: RELIABILITY cho MVP demo (fallback + graceful degrade + health-check)

## Context (vì sao)
Bối cảnh THẬT: MVP để demo/gọi vốn. Model server hiện = Colab+ngrok -> DỄ SẬP giữa buổi
pitch (session 12h, URL đổi, disconnect). Hiện code `raise SystemExit` khi remote lỗi ->
sập cả request, hiện traceback. Cần: fallback tự động sang backup + báo lịch sự khi cả 2
chết + script pre-check trước demo. Đây là best-practice reliability (multi-endpoint,
retry, circuit-break nhẹ, graceful degradation) — chi phí ~0, gây ấn tượng "production-ready".

## Thiết kế

### 1. Multi-endpoint + retry (helper chung)
`src/knowledge/remote_client.py` (mới) — `post_with_fallback(path, payload, cfg)`:
   - Danh sách endpoint: [remote_url, remote_url_backup] (lọc rỗng).
   - Mỗi endpoint: retry 1 lần nếu timeout/5xx, timeout ngắn (vd 30s).
   - Endpoint đầu fail -> sang backup. Cả 2 fail -> raise RemoteUnavailable (exception riêng).
   - Nhớ endpoint đang "sống" (circuit nhẹ): lần sau thử cái sống trước (tránh chờ cái chết).

### 2. retriever.py + reranker.py dùng helper
   - `_encode_remote` / `_score_remote` gọi `post_with_fallback` thay requests trực tiếp.
   - Bỏ `raise SystemExit` -> để RemoteUnavailable nổi lên orchestrator.

### 3. Graceful degrade (orchestrator)
   - answer() bọc retrieve trong try/except RemoteUnavailable:
     -> trả Answer(kind="degraded", text="Hệ thống đang bận, vui lòng thử lại sau ít phút.")
     KHÔNG traceback, KHÔNG sập. Log lỗi thật ra structured log (để dev thấy).

### 4. config (rag.yaml) — thêm endpoint backup
```
retriever:
  remote_url: ""            # chính (env RAG_REMOTE_URL)
  remote_url_backup: ""     # dự phòng (env RAG_REMOTE_URL_BACKUP) — Colab thứ 2 / VPS
  remote_timeout: 30
  remote_retries: 1
```

### 5. Health pre-check trước demo
`scripts/dev/healthcheck.py` (mới) — kiểm CẢ CHUỖI, in xanh/đỏ từng mắt:
   - Qdrant: get_collections + đếm point mỗi collection.
   - Model server chính + backup: GET /health.
   - Groq: gọi 1 request nhỏ (hoặc chỉ kiểm key tồn tại).
   - In bảng: [OK]/[FAIL] từng thành phần -> chạy TRƯỚC khi pitch, xanh hết mới yên tâm.

## Verification
- Tắt Colab chính -> retriever tự chuyển backup (nếu set) -> vẫn trả lời.
- Tắt CẢ 2 -> /chat trả "hệ thống đang bận" (kind=degraded), KHÔNG traceback, HTTP 200 or 503 rõ.
- `python scripts/dev/healthcheck.py` -> bảng trạng thái đúng (đỏ khi tắt server).
- Test logic thuần: post_with_fallback chọn endpoint đúng (mock 2 url, url1 fail -> gọi url2).
- Regression: local backend + full test suite vẫn pass.

## Rủi ro / lưu ý
- MVP demo: KHÔNG over-engineer. Không làm HA/K8s/service-mesh — chỉ fallback + degrade đủ
  để buổi pitch không sập. Roadmap production thật (HA, auth, PHI compliance) để SLIDE.
- circuit-break chỉ "nhớ endpoint sống" trong process — reset khi restart. Đủ cho demo.
- Vẫn nên chuẩn bị: Colab chính + 1 Colab backup (2 tài khoản) chạy song song lúc pitch.
- Groq cũng có thể rate-limit giữa demo -> health-check cảnh báo trước; cân nhắc key dự phòng.
