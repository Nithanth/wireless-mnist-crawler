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


def _http_error(code: int, body: bytes = b"upstream connect error") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://api.openai.com/v1/responses",
        code=code,
        msg="err",
        hdrs=None,
        fp=io.BytesIO(body),
    )


_GEMINI_RPM_BODY = (
    b'{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded for '
    b'quota metric GenerateRequestsPerMinutePerProjectPerModel", '
    b'"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"}]}}'
)
_GEMINI_DAILY_BODY = (
    b'{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded for '
    b'quota metric GenerateRequestsPerDayPerProjectPerModel"}}'
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


def test_429_per_minute_rate_limit_is_not_credit_exhaustion(monkeypatch) -> None:
    """A Gemini per-minute RPM limit retries and, if persistent, raises a plain
    RuntimeError — NEVER CreditExhaustedError (the quota is fine)."""
    monkeypatch.setenv("WIRELESS_TAXONOMY_LLM_MAX_RETRIES", "3")
    sleeps: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(llm, "urlopen", lambda *a, **k: (_ for _ in ()).throw(_http_error(429, _GEMINI_RPM_BODY)))
    with pytest.raises(RuntimeError) as excinfo:
        llm._post_json("https://x", {"a": 1}, {})
    assert not isinstance(excinfo.value, llm.CreditExhaustedError)
    # Honors the server's retryDelay of 7s over the 2s first-attempt backoff.
    assert sleeps and sleeps[0] == 7.0


def test_429_per_minute_recovers_after_retry(monkeypatch) -> None:
    monkeypatch.setenv("WIRELESS_TAXONOMY_LLM_MAX_RETRIES", "3")
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def fake_urlopen(request, timeout=0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(429, _GEMINI_RPM_BODY)
        return _FakeResponse({"ok": True})

    monkeypatch.setattr(llm, "urlopen", fake_urlopen)
    assert llm._post_json("https://x", {"a": 1}, {}) == {"ok": True}


def test_429_daily_quota_raises_credit_exhausted(monkeypatch) -> None:
    monkeypatch.setenv("WIRELESS_TAXONOMY_LLM_MAX_RETRIES", "2")
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)
    monkeypatch.setattr(llm, "urlopen", lambda *a, **k: (_ for _ in ()).throw(_http_error(429, _GEMINI_DAILY_BODY)))
    with pytest.raises(llm.CreditExhaustedError):
        llm._post_json("https://x", {"a": 1}, {})
