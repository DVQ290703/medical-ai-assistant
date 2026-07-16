"""Test post_with_fallback: multi-endpoint + retry (mock requests, không gọi mạng thật)."""
import pytest
from dataclasses import dataclass

import src.knowledge.remote_client as rc
from src.knowledge.remote_client import post_with_fallback, RemoteUnavailable


@dataclass
class Cfg:
    remote_url: str = "http://primary"
    remote_url_backup: str = "http://backup"
    remote_token: str = "t"
    remote_timeout: int = 5
    remote_retries: int = 1


class FakeResp:
    def __init__(self, status=200, data=None):
        self.status_code = status
        self._data = data or {"ok": True}
    def raise_for_status(self):
        import requests
        if self.status_code >= 400:
            raise requests.exceptions.RequestException(f"HTTP {self.status_code}")
    def json(self):
        return self._data


@pytest.fixture(autouse=True)
def reset_last_ok():
    rc._last_ok = None
    yield
    rc._last_ok = None


def test_primary_ok(monkeypatch):
    calls = []
    def fake_post(url, **kw):
        calls.append(url)
        return FakeResp(200, {"from": "primary"})
    monkeypatch.setattr("requests.post", fake_post)
    d = post_with_fallback("/encode", {}, Cfg())
    assert d["from"] == "primary"
    assert calls[0].startswith("http://primary")


def test_primary_chet_chuyen_backup(monkeypatch):
    import requests
    def fake_post(url, **kw):
        if "primary" in url:
            raise requests.exceptions.RequestException("down")
        return FakeResp(200, {"from": "backup"})
    monkeypatch.setattr("requests.post", fake_post)
    d = post_with_fallback("/encode", {}, Cfg())
    assert d["from"] == "backup"


def test_ca_hai_chet_raise(monkeypatch):
    import requests
    def fake_post(url, **kw):
        raise requests.exceptions.RequestException("down")
    monkeypatch.setattr("requests.post", fake_post)
    with pytest.raises(RemoteUnavailable):
        post_with_fallback("/encode", {}, Cfg())


def test_khong_co_endpoint_raise(monkeypatch):
    with pytest.raises(RemoteUnavailable):
        post_with_fallback("/encode", {}, Cfg(remote_url="", remote_url_backup=""))
