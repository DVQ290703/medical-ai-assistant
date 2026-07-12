"""Knowledge plane — Tải PDF phác đồ Bộ Y tế từ kcb.vn (có văn hoá).

Chỉ kcb.vn (Cục Quản lý Khám chữa bệnh) — văn bản hành chính nhà nước KHÔNG bị bảo hộ
bản quyền (Điều 15 Luật SHTT). KHÔNG dùng cho Dược thư/MSD/sách Y (có bản quyền) —
xem governance/knowledge_sources_license.md.

Nguyên tắc crawl có văn hoá:
  - Đọc robots.txt kcb.vn trước; tôn trọng Crawl-delay + Disallow.
  - User-Agent rõ ràng (không giả mạo trình duyệt).
  - Rate-limit: delay giữa các request (mặc định lịch sự).
  - Không tải lại file đã có (idempotent).

Sau khi tải, chạy:
  python -m src.knowledge.kb_ingest --source byt-kcb        # PDF -> JSONL chuẩn hoá
  python -m src.knowledge.chunk --in data/raw/kb/byt_kcb.jsonl --out data/raw/kb/byt_kcb_chunks.jsonl

Cách dùng:
  # (a) tải từ danh sách URL trong 1 file text (mỗi dòng 1 URL PDF):
  python -m src.knowledge.kb_fetch --url-file data/raw/kb/kcb_urls.txt
  # (b) tải trực tiếp vài URL:
  python -m src.knowledge.kb_fetch --urls https://kcb.vn/upload/.../abc.pdf
"""
from __future__ import annotations

import argparse
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

import requests


ALLOWED_HOST = "kcb.vn"                 # CHỈ nguồn này (hợp pháp)
# HTTP header PHẢI là ASCII (mã hoá latin-1) -> không bỏ dấu tiếng Việt vào đây.
USER_AGENT = "medical-ai-assistant/0.1 (research/educational; contact via repo)"
OUT_DIR = "data/raw/kb/pdf"
DEFAULT_DELAY = 3.0                     # giây giữa các request (lịch sự; robots có thể yêu cầu hơn)
TIMEOUT = 60


def _check_host(url: str) -> None:
    host = urlparse(url).netloc.lower()
    if not (host == ALLOWED_HOST or host.endswith("." + ALLOWED_HOST)):
        raise SystemExit(
            f"Từ chối tải {host}: kb_fetch CHỈ cho phép {ALLOWED_HOST} (nguồn hợp pháp). "
            "Nguồn khác có bản quyền — xem governance/knowledge_sources_license.md."
        )


def load_robots(host: str) -> tuple[urllib.robotparser.RobotFileParser, float]:
    """Đọc robots.txt. Trả (parser, crawl_delay). Lỗi mạng -> mặc định lịch sự."""
    rp = urllib.robotparser.RobotFileParser()
    robots_url = f"https://{host}/robots.txt"
    delay = DEFAULT_DELAY
    try:
        resp = requests.get(robots_url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
            cd = rp.crawl_delay(USER_AGENT)
            if cd:
                delay = max(delay, float(cd))
            print(f"[robots] đọc {robots_url} OK. crawl-delay dùng: {delay}s")
        else:
            print(f"[robots] {robots_url} -> HTTP {resp.status_code}; dùng delay mặc định {delay}s.")
            rp = None
    except Exception as e:
        print(f"[robots] lỗi đọc robots ({e}); dùng delay mặc định {delay}s.")
        rp = None
    return rp, delay


def _filename_from_url(url: str) -> str:
    name = Path(urlparse(url).path).name or "download.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


def fetch_urls(urls: list[str], out_dir: str = OUT_DIR, delay: float | None = None) -> int:
    if not urls:
        raise SystemExit("Không có URL nào.")
    for u in urls:
        _check_host(u)                  # chặn nguồn ngoài kcb.vn NGAY

    rp, robots_delay = load_robots(ALLOWED_HOST)
    delay = delay if delay is not None else robots_delay

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})

    ok = skip = fail = 0
    for i, url in enumerate(urls):
        if rp is not None and not rp.can_fetch(USER_AGENT, url):
            print(f"[robots-block] bỏ qua (Disallow): {url}")
            skip += 1
            continue
        dest = out / _filename_from_url(url)
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] đã có: {dest.name}")
            skip += 1
            continue
        try:
            if i > 0:
                time.sleep(delay)       # rate-limit giữa các request
            r = sess.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "")
            if "pdf" not in ctype.lower() and not r.content[:4] == b"%PDF":
                print(f"[warn] {url} không phải PDF (Content-Type={ctype}); vẫn lưu.")
            dest.write_bytes(r.content)
            print(f"[ok] {dest.name} ({len(r.content)/1024:.0f} KB)")
            ok += 1
        except Exception as e:
            print(f"[fail] {url}: {e}")
            fail += 1

    print(f"\n[kb_fetch] tải {ok} | bỏ qua {skip} | lỗi {fail} -> {out}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Tải PDF phác đồ kcb.vn (có văn hoá)")
    ap.add_argument("--urls", nargs="*", default=[], help="URL PDF trực tiếp (kcb.vn)")
    ap.add_argument("--url-file", default=None, help="file text, mỗi dòng 1 URL")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--delay", type=float, default=None, help="ghi đè crawl-delay (giây)")
    args = ap.parse_args()

    urls = list(args.urls)
    if args.url_file:
        with open(args.url_file, encoding="utf-8") as f:
            urls += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    fetch_urls(urls, args.out, args.delay)


if __name__ == "__main__":
    main()
