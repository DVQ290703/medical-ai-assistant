"""Knowledge plane — Chunk STRUCTURE-AWARE tài liệu DÀI. Xem ADR-0002.

Nhận JSONL chuẩn hoá từ kb_ingest.py ({doc_id, source, title, url, section, text, meta})
-> xuất JSONL chunk ({doc_id, chunk_id, source, title, url, section, text}).

Chiến lược 3 tầng (cấu trúc trước, cắt cứng sau, an toàn y khoa trên hết):
  1. HEADING-SPLIT: tách text theo heading (mục đánh số "1." / "1.1" / "I." / "a)") thành
     các SECTION. Mỗi section mang heading của nó.
  2. DOSAGE/LIST GUARD: section trông như bảng liều / danh sách liều (nhiều dòng ngắn +
     đơn vị mg/ml/liều...) -> GIỮ NGUYÊN 1 chunk dù vượt size (KHÔNG tách — an toàn kê đơn).
  3. RECURSIVE FALLBACK: section thường mà vẫn > size -> tách recursive (đoạn -> câu),
     gộp tới ~size với overlap. LUÔN prepend heading section vào mỗi chunk con (giữ ngữ cảnh).

Section ngắn (<= size, đa số article_main) -> giữ nguyên 1 chunk (no-op).

Đơn vị size/overlap theo TOKEN xấp xỉ (từ tiếng Việt ~1.5 token) — đếm theo TỪ để tránh
phụ thuộc tokenizer nặng ở bước chunk. Config: configs/rag.yaml khối `chunk`
(mặc định size 768 / overlap 96 = 12.5%).

Usage:
  python -m src.knowledge.chunk --in data/raw/kb/vn_healthcare.jsonl --out data/raw/kb/vn_healthcare_chunks.jsonl --stats
  python -m src.knowledge.chunk --in data/raw/kb/byt_kcb.jsonl --out data/raw/kb/byt_kcb_chunks.jsonl --stats
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml


CONFIG_PATH = "configs/rag.yaml"
TOKENS_PER_WORD = 1.5           # từ tiếng Việt ~ 1.5 token BGE-M3. size/overlap trong yaml là TOKEN.

# --- Heading: đầu dòng là mục đánh số / La Mã / chữ cái ---
#   "1. ", "1.2 ", "2.1.3 ", "I. ", "IV) ", "a) "
_HEADING_RE = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*[.)]?\s+"                 # 1.  / 1.2  / 2.1.3.
    r"|[IVXLC]+[.)]\s+"                        # I.  II)  IV.
    r"|[a-zA-Z][.)]\s+"                        # a)  b.
    r")\S",
    re.UNICODE,
)
_PARA_RE = re.compile(r"\n\s*\n")            # ranh giới đoạn
_SENT_RE = re.compile(r"(?<=[.!?…])\s+")      # ranh giới câu (thô)

# --- Nhận bảng liều / danh sách: đơn vị thuốc ---
_DOSAGE_UNIT_RE = re.compile(
    r"\b\d+([.,]\d+)?\s*(mg|mcg|µg|g|ml|mL|UI|IU|đơn vị|viên|ống|lần/ngày|mg/kg|mg/ngày)\b",
    re.IGNORECASE,
)


def load_chunk_cfg(path: str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    c = y.get("chunk", {}) or {}
    # min_size: ngưỡng gom chunk nhỏ liền kề (mặc định ~size/3). 0 = tắt gom.
    return {"size": c.get("size", 768), "overlap": c.get("overlap", 96),
            "min_size": c.get("min_size", c.get("size", 768) // 3),
            "strategy": c.get("strategy", "structure-aware")}


def _n_tokens(text: str) -> int:
    return int(len(text.split()) * TOKENS_PER_WORD)


def _looks_like_dosage_block(text: str) -> bool:
    """Section trông như bảng liều/danh sách liều -> không được tách."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    dosage_hits = len(_DOSAGE_UNIT_RE.findall(text))
    short_lines = sum(1 for ln in lines if len(ln.split()) <= 12)
    return dosage_hits >= 2 and short_lines >= max(3, len(lines) * 0.5)


def split_by_heading(text: str) -> list[dict]:
    """Tách text thành list section {heading, body}. Heading = dòng khớp _HEADING_RE."""
    sections: list[dict] = []
    cur_heading = ""
    cur_lines: list[str] = []

    def flush():
        body = "\n".join(cur_lines).strip()
        if cur_heading or body:
            sections.append({"heading": cur_heading.strip(), "body": body})

    for ln in text.splitlines():
        stripped = ln.strip()
        if stripped and _HEADING_RE.match(stripped):
            flush()
            cur_heading, cur_lines = stripped, []
        else:
            cur_lines.append(ln)
    flush()

    if not sections:
        sections = [{"heading": "", "body": text.strip()}]
    return sections


def _split_recursive(text: str) -> list[str]:
    """Tách theo ranh giới giảm dần: đoạn -> câu."""
    out = []
    for p in (p.strip() for p in _PARA_RE.split(text) if p.strip()):
        sents = [s.strip() for s in _SENT_RE.split(p) if s.strip()]
        out.extend(sents if sents else [p])
    return out


def _pack(pieces: list[str], size: int, overlap: int, prefix: str = "") -> list[str]:
    """Gộp mảnh -> chunk ~size token, overlap giữa các chunk. prefix (heading) vào mỗi chunk."""
    prefix_tok = _n_tokens(prefix) if prefix else 0
    budget = max(1, size - prefix_tok)         # chừa chỗ cho heading prepend
    chunks, cur, cur_tok = [], [], 0
    for piece in pieces:
        pt = _n_tokens(piece)
        if cur and cur_tok + pt > budget:
            chunks.append(" ".join(cur))
            if overlap > 0:                    # giữ đuôi ~overlap token cho chunk sau
                tail, tail_tok = [], 0
                for s in reversed(cur):
                    st = _n_tokens(s)
                    if tail_tok + st > overlap:
                        break
                    tail.insert(0, s)
                    tail_tok += st
                cur, cur_tok = tail[:], tail_tok
            else:
                cur, cur_tok = [], 0
        cur.append(piece)
        cur_tok += pt
    if cur:
        chunks.append(" ".join(cur))
    if prefix:
        chunks = [f"{prefix}\n{c}" if c else prefix for c in chunks]
    return chunks


def chunk_section(sec: dict, size: int, overlap: int) -> list[str]:
    """1 section -> list chunk. dosage-guard + recursive fallback + prepend heading."""
    heading, body = sec["heading"], sec["body"]
    full = (f"{heading}\n{body}" if heading and body else heading or body).strip()
    if not full:
        return []
    if _n_tokens(full) <= size:                # đã ngắn -> giữ nguyên
        return [full]
    if _looks_like_dosage_block(body):         # bảng liều -> KHÔNG tách (an toàn y khoa)
        return [full]
    return _pack(_split_recursive(body), size, overlap, prefix=heading)


def _merge_small(chunks: list[str], size: int, min_size: int) -> list[str]:
    """Gộp chunk liền kề nhỏ (< min_size token) tới khi đạt min_size hoặc chạm size.

    Giảm chunk vụn (heading trơ, 1-2 dòng) -> retrieval bớt nhiễu. KHÔNG gộp vượt size
    (giữ chunk lớn/bảng liều đứng riêng).
    """
    merged, buf, buf_tok = [], [], 0
    for c in chunks:
        ct = _n_tokens(c)
        if ct >= size:                      # chunk lớn (vd bảng liều): flush buf rồi để riêng
            if buf:
                merged.append("\n".join(buf)); buf, buf_tok = [], 0
            merged.append(c)
            continue
        if buf and buf_tok + ct > size:     # gộp thêm sẽ vượt size -> flush
            merged.append("\n".join(buf)); buf, buf_tok = [], 0
        buf.append(c); buf_tok += ct
        if buf_tok >= min_size:             # đủ lớn -> chốt 1 chunk
            merged.append("\n".join(buf)); buf, buf_tok = [], 0
    if buf:
        merged.append("\n".join(buf))
    return merged


def chunk_text(text: str, size: int, overlap: int, min_size: int = 0) -> list[str]:
    """Structure-aware: heading-split -> chunk từng section -> gộp chunk nhỏ liền kề."""
    text = (text or "").strip()
    if not text:
        return []
    out = []
    for sec in split_by_heading(text):
        out.extend(chunk_section(sec, size, overlap))
    out = [c for c in out if c.strip()]
    if min_size > 0:
        out = _merge_small(out, size, min_size)
    return out


def chunk_record(rec: dict, size: int, overlap: int, min_size: int = 0) -> list[dict]:
    """1 document -> nhiều chunk record. Giữ title/url; section = heading của chunk (nếu có)."""
    pieces = chunk_text(rec.get("text", ""), size, overlap, min_size)
    out = []
    for i, piece in enumerate(pieces):
        first_line = piece.splitlines()[0] if piece else ""
        sec = first_line if _HEADING_RE.match(first_line.strip()) else rec.get("section", "")
        out.append({
            "doc_id": rec["doc_id"],
            "chunk_id": f"{rec['doc_id']}#{i}",
            "source": rec.get("source", ""),
            "title": rec.get("title", ""),
            "url": rec.get("url", ""),
            "section": sec,
            "text": piece,
        })
    return out


def run(in_path: str, out_path: str, config_path: str = CONFIG_PATH,
        show_stats: bool = False, limit: int | None = None) -> int:
    cfg = load_chunk_cfg(config_path)
    size, overlap, min_size = cfg["size"], cfg["overlap"], cfg["min_size"]
    print(f"[chunk] structure-aware | size={size} overlap={overlap} min={min_size} tok "
          f"(~{TOKENS_PER_WORD} tok/từ)")

    inp, out = Path(in_path), Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_doc = n_chunk = n_kept_big = 0
    chunk_tok_lens = []
    with inp.open(encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if limit and n_doc >= limit:
                break
            rec = json.loads(line)
            n_doc += 1
            for ch in chunk_record(rec, size, overlap, min_size):
                fout.write(json.dumps(ch, ensure_ascii=False) + "\n")
                n_chunk += 1
                tk = _n_tokens(ch["text"])
                if tk > size:
                    n_kept_big += 1           # vượt size = block liều giữ nguyên (cố ý)
                if show_stats:
                    chunk_tok_lens.append(tk)

    print(f"[chunk] {n_doc:,} document -> {n_chunk:,} chunk ({n_chunk/max(1,n_doc):.2f}/doc) -> {out}")
    if n_kept_big:
        print(f"[chunk] {n_kept_big} chunk vượt size (cố ý giữ: bảng liều/section không tách).")
    if show_stats and chunk_tok_lens:
        chunk_tok_lens.sort()
        m = len(chunk_tok_lens)
        print(f"[stats] chunk token ~: median={chunk_tok_lens[m//2]}  "
              f"p90={chunk_tok_lens[int(m*0.9)]}  max={chunk_tok_lens[-1]}")
    return n_chunk


def main() -> None:
    ap = argparse.ArgumentParser(description="Chunk structure-aware tài liệu dài -> JSONL chunk")
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--stats", action="store_true", help="in phân bố độ dài chunk")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args.in_path, args.out_path, args.config, args.stats, args.limit)


if __name__ == "__main__":
    main()
