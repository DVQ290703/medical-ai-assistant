"""Knowledge plane — Client gọi model server remote CÓ FALLBACK + retry.

Model server (Colab qua ngrok) dễ chết giữa phiên. Client này:
  - Thử nhiều endpoint [chính, backup] theo thứ tự.
  - Mỗi endpoint retry 1 lần nếu timeout/5xx.
  - Nhớ endpoint "sống" gần nhất -> lần sau thử nó trước (circuit nhẹ).
  - Cả hai chết -> raise RemoteUnavailable (orchestrator bắt -> graceful degrade).

Dùng chung cho /encode (retriever) và /rerank (reranker).
"""
from __future__ import annotations

import time


class RemoteUnavailable(Exception):
    """Mọi endpoint model server đều không phản hồi."""


# endpoint sống gần nhất (nhớ giữa các call trong 1 process) -> thử trước cho nhanh
_last_ok: str | None = None


def _endpoints(cfg) -> list[str]:
    urls = [getattr(cfg, "remote_url", "") or "",
            getattr(cfg, "remote_url_backup", "") or ""]
    urls = [u.rstrip("/") for u in urls if u]
    # ưu tiên endpoint sống gần nhất
    if _last_ok and _last_ok in urls:
        urls = [_last_ok] + [u for u in urls if u != _last_ok]
    return urls


def post_with_fallback(path: str, payload: dict, cfg) -> dict:
    """POST payload tới {endpoint}{path} qua các endpoint đến khi thành công.

    path: '/encode' | '/rerank'. Trả JSON dict. Cả 2 endpoint fail -> RemoteUnavailable.
    """
    global _last_ok
    import requests

    endpoints = _endpoints(cfg)
    if not endpoints:
        raise RemoteUnavailable(
            "Chưa cấu hình remote_url (set RAG_REMOTE_URL trong .env hoặc rag.yaml)."
        )
    token = getattr(cfg, "remote_token", "") or ""
    timeout = getattr(cfg, "remote_timeout", 30)
    retries = getattr(cfg, "remote_retries", 1)

    errors = []
    for url in endpoints:
        for attempt in range(retries + 1):
            try:
                r = requests.post(url + path, json=payload,
                                  headers={"X-Token": token}, timeout=timeout)
                if r.status_code >= 500:      # server lỗi -> retry rồi mới bỏ
                    raise requests.exceptions.RequestException(f"HTTP {r.status_code}")
                r.raise_for_status()
                _last_ok = url
                return r.json()
            except requests.exceptions.RequestException as e:
                errors.append(f"{url} (lần {attempt+1}): {e}")
                if attempt < retries:
                    time.sleep(0.5)           # backoff ngắn trước khi retry
        # endpoint này hỏng -> sang endpoint sau

    raise RemoteUnavailable("Model server không phản hồi. Chi tiết: " + " | ".join(errors))
