"""Knowledge plane — Ingest TRI THỨC NỀN (article dài / phác đồ PDF) -> JSONL chuẩn hoá.

Khác src/knowledge/ingest.py (đó là Q&A NGẮN, 1 cặp = 1 unit, không chunk). Đây là
tài liệu DÀI -> chunk.py sẽ cắt sau.

Schema chuẩn hoá nội bộ (mỗi dòng JSONL):
    {doc_id, source, title, url, section, text, meta}
- source: nhãn nguồn (vd "vinmec-article", "byt-kcb") -> phân biệt trong cùng collection.
- url/title: để TRÍCH DẪN (phục vụ src/serving/citation.py).
- section: rỗng ở bước ingest; chunk.py điền heading khi cắt.

Nguồn đọc từ configs/rag.yaml khối `knowledge_base`. Loader theo `type`:
  - hf : datasets.load_dataset (vd urnus11/Vietnamese-Healthcare split vinmec_article_main)
  - pdf: pymupdf (fitz) trích text từ PDF trong input_dir (vd phác đồ kcb.vn)

Chỉ index nguồn có license OK — xem governance/knowledge_sources_license.md.

Usage:
  python -m src.knowledge.kb_ingest --source vn-healthcare-article --inspect --limit 20
  python -m src.knowledge.kb_ingest --source vn-healthcare-article
  python -m src.knowledge.kb_ingest --source byt-kcb
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import yaml

# Nạp .env nếu chạy local (HF_TOKEN...). Trên Kaggle không có .env -> bỏ qua.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# reuse xác thực HF (xử lý stale HF_TOKEN) từ ingest.py
from src.knowledge.ingest import authenticate


CONFIG_PATH = "configs/rag.yaml"


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_source_entry(cfg: dict, source_id: str) -> dict:
    entries = cfg.get("knowledge_base") or []
    entry = next((e for e in entries if e.get("id") == source_id), None)
    if entry is None:
        ids = [e.get("id") for e in entries]
        raise SystemExit(f"Không thấy source id='{source_id}' trong knowledge_base. Có: {ids}")
    return entry


# ============================================================
# LOADER: HF dataset (article dài)
# ============================================================

def load_hf_rows(entry: dict, token: str | None, limit: int | None):
    """Nạp split article từ HF -> list dict schema chuẩn. field_map: {title, text, url}."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Thiếu `datasets`. pip install datasets.") from e

    source = entry["source"]
    split = entry.get("split", "train")
    fm = entry["field_map"]                      # {title:..., text:..., url:...}
    source_tag = entry.get("source_tag", entry["id"])

    try:
        ds = load_dataset(source, split=split, token=token)
    except Exception as e:
        msg = str(e).lower()
        if "401" in msg or "403" in msg or "gated" in msg:
            raise PermissionError(
                f"Chưa có quyền truy cập {source}. Vào trang dataset HF bấm "
                "'Agree and access repository', rồi thử lại."
            ) from e
        raise
    print(f"[ok] Loaded {len(ds):,} rows từ {source}, split='{split}'")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    rows = []
    for i, rec in enumerate(ds):
        title = (rec.get(fm["title"]) or "").strip()
        text = (rec.get(fm["text"]) or "").strip()
        url = (rec.get(fm.get("url", "")) or "").strip() if fm.get("url") else ""
        if not text:
            continue  # bỏ bài rỗng
        rows.append({
            "doc_id": f"{source_tag}:{i}",
            "source": source_tag,
            "title": title,
            "url": url,
            "section": "",
            "text": text,
            "meta": {},
        })
    return rows


# ============================================================
# LÀM SẠCH TEXT PDF (font TCVN3 cũ + mục lục)
# ============================================================

# Ký tự sai do font cũ khi trích qua pymupdf -> Unicode chuẩn.
# Đã xác minh trên PDF kcb.vn (scripts/dev/diag_encoding.py): chữ "Ư/ư có sừng"
# bị map nhầm sang "OI" (U+01A2/U+01A3) vì gần nhau trong bảng mã.
# Dùng code-point tường minh (\u...) để tránh nhầm lẫn hiển thị ký tự.
_TCVN_FIX = {
    "Ƣ": "Ư",   # Ƣ (LATIN CAPITAL OI) -> Ư (U WITH HORN)
    "ƣ": "ư",   # ƣ (LATIN SMALL OI)   -> ư
}
# dòng mục lục: kết thúc bằng chuỗi dấu chấm dẫn + số trang, vd "1. ĐẠI CƯƠNG ...... 4"
_TOC_LINE_RE = re.compile(r"\.{4,}\s*\d+\s*$")


def _fix_tcvn(text: str) -> str:
    for bad, good in _TCVN_FIX.items():
        if bad in text:
            text = text.replace(bad, good)
    return text


def _strip_toc(text: str) -> str:
    """Bỏ các dòng mục lục (dấu chấm dẫn + số trang) — rác cho retrieval."""
    kept = [ln for ln in text.splitlines() if not _TOC_LINE_RE.search(ln)]
    return "\n".join(kept)


def clean_pdf_text(text: str) -> str:
    return _strip_toc(_fix_tcvn(text)).strip()


# ============================================================
# LOADER: PDF (phác đồ kcb.vn)
# ============================================================

def load_pdf_rows(entry: dict, cfg: dict, limit: int | None):
    """Trích text từ mọi PDF trong input_dir. Phát hiện scan -> cảnh báo, bỏ qua."""
    try:
        import fitz  # pymupdf
    except ImportError as e:
        raise SystemExit("Thiếu `pymupdf`. pip install pymupdf.") from e

    input_dir = Path(entry["input_dir"])
    if not input_dir.exists():
        raise SystemExit(
            f"Không thấy thư mục PDF: {input_dir}. Chạy kb_fetch.py để tải phác đồ kcb.vn trước."
        )
    source_tag = entry.get("source_tag", entry["id"])
    min_chars = (cfg.get("pdf") or {}).get("min_chars_per_page", 100)

    pdfs = sorted(input_dir.glob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]
    print(f"[pdf] {len(pdfs)} file trong {input_dir}")

    rows, skipped = [], 0
    for pi, pdf_path in enumerate(pdfs):
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"  [warn] mở lỗi {pdf_path.name}: {e}")
            continue
        pages_text = [page.get_text("text") for page in doc]
        n_pages = len(pages_text)
        total_chars = sum(len(t) for t in pages_text)
        doc.close()

        # scan check: quá ít text/trang -> nhiều khả năng PDF scan (cần OCR)
        if n_pages and total_chars / n_pages < min_chars:
            print(f"  [skip-scan] {pdf_path.name}: {total_chars/max(1,n_pages):.0f} ký tự/trang "
                  f"< {min_chars} -> có thể là scan, cần OCR.")
            skipped += 1
            continue

        text = clean_pdf_text("\n".join(pages_text))   # fix font TCVN3 + bỏ mục lục
        if not text:
            skipped += 1
            continue
        rows.append({
            "doc_id": f"{source_tag}:{pi}",
            "source": source_tag,
            "title": pdf_path.stem,          # tên file làm title (thường có tên phác đồ)
            "url": "",                        # kb_fetch.py có thể ghi map file->url sau
            "section": "",
            "text": text,
            "meta": {"file": pdf_path.name, "n_pages": n_pages},
        })
    if skipped:
        print(f"[pdf] bỏ qua {skipped} file (scan/rỗng).")
    return rows


# ============================================================
# INSPECT + WRITE
# ============================================================

def inspect(rows: list[dict], n: int = 3):
    print(f"\n=== {len(rows):,} document, {min(n,len(rows))} mẫu đầu ===")
    for r in rows[:n]:
        print(f"\n--- {r['doc_id']} (source={r['source']}) ---")
        print(f"  title: {r['title'][:120]}")
        print(f"  url  : {r['url']}")
        t = r["text"].replace("\n", " ")
        print(f"  text ({len(r['text'])} chars): {t[:300]}...")
    if rows:
        lens = sorted(len(r["text"]) for r in rows)
        n_ = len(lens)
        print(f"\n=== ĐỘ DÀI text (chars) ===")
        print(f"  median={lens[n_//2]:,}  p90={lens[int(n_*0.9)]:,}  max={max(lens):,}")


def write_jsonl(rows: list[dict], out_path: str) -> int:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[kb_ingest] {len(rows):,} document -> {out}")
    return len(rows)


def ingest_source(source_id: str, config_path: str = CONFIG_PATH,
                  do_inspect: bool = False, limit: int | None = None) -> None:
    cfg = load_config(config_path)
    entry = get_source_entry(cfg, source_id)
    stype = entry.get("type")

    if stype == "hf":
        token = authenticate(os.environ.get("HF_TOKEN"))
        rows = load_hf_rows(entry, token, limit)
    elif stype == "pdf":
        rows = load_pdf_rows(entry, cfg, limit)
    else:
        raise SystemExit(f"type '{stype}' chưa hỗ trợ (chỉ hf | pdf).")

    if do_inspect:
        inspect(rows)
        if limit and not entry.get("_force_write"):
            print("\n[note] Đang --inspect (có --limit). Bỏ --inspect để ghi JSONL đầy đủ.")
            return
    write_jsonl(rows, entry["out"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest tri thức nền (article/PDF) -> JSONL")
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--source", required=True, help="id trong knowledge_base của rag.yaml")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    ingest_source(args.source, args.config, args.inspect, args.limit)


if __name__ == "__main__":
    main()
