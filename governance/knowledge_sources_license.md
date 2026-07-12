# Knowledge Sources — Tình trạng License / Bản quyền

Nguồn sự thật để quyết định nguồn nào ĐƯỢC index vào RAG. Trước khi thêm bất kỳ nguồn mới
vào `configs/rag.yaml`, kiểm tra ở đây. Mục đích dự án hiện tại: **học tập / nghiên cứu
(phi thương mại)**.

> Cảnh báo: bảng này là tham khảo kỹ thuật, KHÔNG phải tư vấn pháp lý. Trước khi dùng
> thương mại hoặc thu thập quy mô lớn, rà soát lại với chuyên gia pháp lý.

| Nguồn | Định dạng | License / Bản quyền | Index? | Ghi chú |
|---|---|---|---|---|
| **urnus11/Vietnamese-Healthcare** (HF) | dataset parquet | Không công bố; nội dung crawl từ Vinmec/VnExpress | ✅ (học tập) | Bản quyền gốc thuộc Vinmec. Rủi ro thấp cho nghiên cứu phi thương mại. **Không dùng thương mại** nếu chưa xin phép. |
| **phuocsang/vinmec-medical-qa** (HF) | dataset | Không công bố; crawl Vinmec | ✅ (học tập) | Đã ingest → `data/raw/vinmec.jsonl`. Cùng lưu ý như trên. |
| **Phác đồ / HDĐT Bộ Y tế** (kcb.vn) | PDF | **Văn bản hành chính nhà nước — KHÔNG bị bảo hộ** (Điều 15 Luật SHTT) | ✅ | Nguồn hợp pháp nhất. Tải từ kcb.vn (kênh gốc). Crawl có văn hoá: User-Agent rõ, rate-limit, tôn trọng robots. Phần "tài liệu chuyên môn" đính kèm hiếm khi chứa hình/bảng bản quyền bên thứ ba — lưu ý khi tái xuất bản nguyên văn. |
| **Dược thư Quốc gia VN** | ebook thương mại | **Có bản quyền** © Viện Kiểm nghiệm thuốc TW; bán qua ebook365.vn | ❌ | Nội dung 743 chuyên luận có bản quyền. Con đường hợp pháp = mua/xin license (NIDQC). Không dùng bản PDF lậu. |
| **MSD Manual tiếng Việt** (msdmanuals.com/vi) | HTML | **Có bản quyền** © Merck/MSD; ToS chỉ cho dùng cá nhân phi thương mại | ❌ | robots.txt không chặn kỹ thuật (Crawl-delay 5) nhưng ToS yêu cầu **xin phép văn bản** để tái dùng. Nạp vào RAG phục vụ người khác = "mục đích công khai" → cần email `msdmanualspermissions@msd.com`. |
| **Phác đồ BV lớn** (Chợ Rẫy, Bạch Mai) | sách in | **Có bản quyền** (BV + NXB) | ❌ | Không có kênh tải chính thức. PDF trôi nổi là bản lậu. Cần xin phép BV/NXB. |
| **Sách giáo khoa trường Y** | sách in / ebook | **Có bản quyền đầy đủ** (NXB Y học) | ❌ | Rủi ro cao nhất. Chỉ dùng nếu có license/hợp tác. |

## Nguyên tắc
1. Chỉ index nguồn cột "Index? = ✅".
2. Nguồn ❌ chỉ được thêm SAU KHI có license/thư đồng ý — cập nhật lại bảng này trước.
3. Mọi document index vào Qdrant phải mang metadata `source` + `url` để **trích dẫn được**
   (phục vụ `src/serving/citation.py`) và truy vết nguồn.

## Nguồn nghiên cứu pháp lý (2026-07)
- Luật SHTT Điều 15 — văn bản hành chính không thuộc phạm vi bảo hộ quyền tác giả.
- kcb.vn — Cục Quản lý Khám chữa bệnh, Bộ Y tế (kênh gốc phác đồ).
- NIDQC — thông báo phát hành ebook Dược thư QG lần 3 (bán thương mại).
- MSD Manual — robots.txt (Crawl-delay 5) + trang Permissions (yêu cầu xin phép).
