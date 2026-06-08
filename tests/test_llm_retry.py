import io
import json
import urllib.error

import pytest

from wireless_taxonomy import llm


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._data = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://api.openai.com/v1/responses",
        code=code,
        msg="err",
        hdrs=None,
        fp=io.BytesIO(b"upstream connect error"),
    )


def test_post_json_retries_transient_then_succeeds(monkeypatch) -> None:
    monkeypatch.setenv("WIRELESS_TAXONOMY_LLM_MAX_RETRIES", "4")
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def fake_urlopen(request, timeout=0):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _http_error(503)
        return _FakeResponse({"ok": True})

    monkeypatch.setattr(llm, "urlopen", fake_urlopen)
    out = llm._post_json("https://x", {"a": 1}, {})
    assert out == {"ok": True}
    assert calls["n"] == 3


def test_post_json_does_not_retry_client_error(monkeypatch) -> None:
    monkeypatch.setenv("WIRELESS_TAXONOMY_LLM_MAX_RETRIES", "4")
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def fake_urlopen(request, timeout=0):
        calls["n"] += 1
        raise _http_error(400)

    monkeypatch.setattr(llm, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="HTTP 400"):
        llm._post_json("https://x", {"a": 1}, {})
    assert calls["n"] == 1
