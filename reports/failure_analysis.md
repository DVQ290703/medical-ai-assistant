# Failure Analysis — Error Taxonomy

Phân loại lỗi để **quy trách nhiệm về đúng tầng** (attribution) rồi ưu tiên fix, thay vì chỉ
biết "sai bao nhiêu %". Sinh từ [`src/evaluation/failure_attribution.py`](../src/evaluation/failure_attribution.py):
chạy model qua eval set → mỗi câu sai gán vào một loại lỗi dưới đây.

> ⚠️ **Trạng thái: CHƯA có số liệu.** Cần chạy generation end-to-end (Groq/local) qua eval set
> đã label rồi phân loại thủ công/bán tự động. Bảng dưới là **taxonomy + hướng fix**; cột Count/%
> điền sau khi có bản chạy thật.

| Loại lỗi | Tầng chịu trách nhiệm | Count | % | Hướng fix chính |
|---|---|---|---|---|
| **Retrieval Failure** — không lấy được tài liệu chứa đáp án | Knowledge (retriever) | TODO | | Cải thiện chunking/embedding, tăng top_k, giảm threshold, kiểm tra coverage corpus |
| **Context Ignored** — có tài liệu đúng trong context nhưng model bỏ qua/trả sai | Generation | TODO | | Prompt bám context chặt hơn, giảm temperature, ép trích dẫn từ context |
| **Hallucination** — bịa fact không có trong bất kỳ nguồn nào | Generation | TODO | | Threshold retrieval (rỗng → từ chối), output guard bắt câu không có citation |
| **Dosage Error** — sai liều thuốc (loại nguy hiểm nhất) | Generation + Knowledge | TODO | | Dosage-guard khi chunk (giữ nguyên bảng liều — ADR-0002), rerank ưu tiên nguồn BYT |
| **Safety Policy Error** — thiếu disclaimer, không định tuyến cấp cứu, không từ chối câu ngoài phạm vi | Serving (policy/guard) | TODO | | Bổ sung rule ở `src/serving/policy/`, mở rộng eval set cấp cứu/PII |

## Cách đọc bảng này

- **Retrieval Failure vs Context Ignored** tách nhau là điểm mấu chốt: cùng là "trả lời sai" nhưng
  fix ở hai tầng khác nhau. Attribution sai → fix nhầm chỗ.
- **Dosage Error** đếm riêng dù hiếm, vì đây là loại lỗi rủi ro cao nhất về an toàn/pháp lý
  (xem ADR-0001 — lý do facts phải đến từ RAG có trích dẫn).

## Việc cần làm để điền số

1. Chuẩn bị eval set có đáp án tham chiếu (VM14K/MedQA đã label, hoặc golden set tự dựng).
2. Chạy pipeline serving end-to-end trên từng câu, lưu (query, context, answer).
3. Phân loại mỗi câu sai theo taxonomy trên → điền Count/%.
4. Sắp theo % giảm dần → chọn "Top fix" cho 1-2 loại lỗi lớn nhất.
