"""Monitoring — Observability qua Langfuse (trace query -> retrieve/generate).

AN TOÀN: nếu chưa cài langfuse HOẶC thiếu key -> mọi thứ thành NO-OP (không phá luồng
chạy thường). Bật bằng cách cài `langfuse` + set LANGFUSE_PUBLIC_KEY / SECRET_KEY / BASE_URL.

Dùng ở orchestrator: bọc 1 query thành trace, các bước (retrieve, generate) thành span.
KHÔNG dùng decorator (giữ code tự viết minh bạch) — dùng context manager thủ công.

SDK v3: langfuse.start_as_current_observation(as_type=..., name=...).
"""
from __future__ import annotations

import os
from contextlib import contextmanager

_client = None
_enabled = False


def _init():
    global _client, _enabled
    if _client is not None or _enabled:
        return
    if not os.environ.get("LANGFUSE_PUBLIC_KEY") or not os.environ.get("LANGFUSE_SECRET_KEY"):
        return  # thiếu key -> no-op
    try:
        from langfuse import get_client
        _client = get_client()
        _enabled = True
    except Exception as e:  # SDK chưa cài / lỗi -> no-op, không phá app
        print(f"[obs] Langfuse tắt ({e}); chạy không trace.")


class _NoopSpan:
    trace_id = None            # để code lấy .trace_id không lỗi khi no-op
    def update(self, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False


@contextmanager
def trace(name: str, **fields):
    """Root trace cho 1 query. Trả object có .update(input=..., output=...)."""
    _init()
    if not _enabled:
        yield _NoopSpan()
        return
    with _client.start_as_current_observation(as_type="span", name=name) as t:
        if fields:
            t.update(**fields)
        yield t


@contextmanager
def span(name: str, as_type: str = "span", **fields):
    """Span con (retrieve/generate...). as_type='generation' cho bước gọi LLM."""
    if not _enabled:
        yield _NoopSpan()
        return
    with _client.start_as_current_observation(as_type=as_type, name=name, **fields) as s:
        yield s


def score(trace_id: str, name: str, value: float, comment: str = "") -> bool:
    """Gắn score (feedback người dùng 👍/👎) vào trace đã có. No-op nếu tắt/thiếu id.

    Trả True nếu gửi được. value: 1.0 (👍) / 0.0 (👎). Gọi từ endpoint /feedback (request
    sau) — không cần trace còn 'sống', chỉ cần trace_id.
    """
    _init()
    if not _enabled or not trace_id:
        return False
    try:
        _client.create_score(name=name, value=value, trace_id=trace_id,
                             data_type="NUMERIC", comment=comment or None)
        _client.flush()
        return True
    except Exception as e:
        print(f"[obs] score lỗi ({e})")
        return False


def flush():
    """Đẩy dữ liệu lên Langfuse (gọi cuối request — app ngắn hạn cần flush)."""
    if _enabled and _client:
        try:
            _client.flush()
        except Exception:
            pass
