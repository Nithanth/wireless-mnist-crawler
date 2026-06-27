from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from wireless_taxonomy.config import LlmSettings, ProviderConfig


class CreditExhaustedError(RuntimeError):
    """Raised when the LLM provider reports exhausted credits / quota.

    Callers should save any cache/checkpoint state before propagating so the
    pipeline can resume where it left off after the user reloads credits.
    """


@dataclass(frozen=True)
class LlmRequest:
    task: str
    prompt: str
    schema_name: str | None = None
    metadata: dict[str, Any] | None = None
    pdf_bytes: bytes | None = None
    use_web_search: bool = False


@dataclass(frozen=True)
class LlmResponse:
    provider: str
    model: str
    content: str
    parsed: dict[str, Any] | list[Any] | None
    confidence: float | None = None


class LlmRouter:
    """Small provider router using raw HTTPS APIs to avoid SDK lock-in."""

    def __init__(self, settings: LlmSettings):
        self.settings = settings

    def configured_providers(self) -> tuple[ProviderConfig, ...]:
        return tuple(provider for provider in self.settings.ordered_providers() if provider.api_key_configured)

    def select_provider(self) -> ProviderConfig:
        configured = self.configured_providers()
        if not configured:
            raise RuntimeError(
                "No LLM provider API key configured. Set one of OPENAI_API_KEY, "
                "ANTHROPIC_API_KEY, or GOOGLE_API_KEY."
            )
        return configured[0]

    def complete(self, request: LlmRequest) -> LlmResponse:
        errors: list[str] = []
        for provider in self.configured_providers():
            try:
                content = self._complete_with_provider(provider, request)
                return LlmResponse(
                    provider=provider.provider,
                    model=provider.model,
                    content=content,
                    parsed=_parse_json_content(content),
                )
            except Exception as exc:  # Providers are fallbacks; preserve concise failure context.
                errors.append(f"{provider.provider}: {exc}")
        if not errors:
            self.select_provider()
        raise RuntimeError("All configured LLM providers failed: " + " | ".join(errors))

    def _complete_with_provider(self, provider: ProviderConfig, request: LlmRequest) -> str:
        if provider.provider == "openai":
            return _openai_complete(provider, request)
        if provider.provider == "anthropic":
            return _anthropic_complete(provider, request)
        if provider.provider == "google":
            return _google_complete(provider, request)
        raise ValueError(f"Unsupported provider: {provider.provider}")


def _pdf_bytes_to_text(pdf_bytes: bytes, max_chars: int = 120_000) -> str:
    """Extract plain text from PDF bytes using pypdf. Truncates to max_chars."""
    try:
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n".join(pages).strip()[:max_chars]
    except Exception:
        return ""


def _openai_complete(provider: ProviderConfig, request: LlmRequest) -> str:
    prompt = request.prompt
    if request.pdf_bytes:
        pdf_text = _pdf_bytes_to_text(request.pdf_bytes)
        if pdf_text:
            prompt = f"[Full paper text extracted from PDF]\n---\n{pdf_text}\n---\n\n{request.prompt}"
    body = {
        "model": provider.model,
        "input": prompt,
        "temperature": 0,
    }
    data = _post_json(
        "https://api.openai.com/v1/responses",
        body,
        {
            "Authorization": f"Bearer {os.environ[provider.api_key_env]}",
            "Content-Type": "application/json",
        },
    )
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    chunks: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def _anthropic_complete(provider: ProviderConfig, request: LlmRequest) -> str:
    import base64

    if request.pdf_bytes:
        content: list[dict[str, Any]] = [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64.standard_b64encode(request.pdf_bytes).decode(),
                },
            },
            {"type": "text", "text": request.prompt},
        ]
    else:
        content = request.prompt  # type: ignore[assignment]
    body = {
        "model": provider.model,
        "max_tokens": int(os.getenv("WIRELESS_TAXONOMY_LLM_MAX_TOKENS", "16000")),
        "temperature": 0,
        "messages": [{"role": "user", "content": content}],
    }
    data = _post_json(
        "https://api.anthropic.com/v1/messages",
        body,
        {
            "x-api-key": os.environ[provider.api_key_env],
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    )
    return "\n".join(part.get("text", "") for part in data.get("content", []) if part.get("type") == "text").strip()


def _google_complete(provider: ProviderConfig, request: LlmRequest) -> str:
    import base64

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")
    parts: list[dict[str, Any]] = []
    if request.pdf_bytes:
        parts.append({
            "inline_data": {
                "mime_type": "application/pdf",
                "data": base64.standard_b64encode(request.pdf_bytes).decode(),
            }
        })
    parts.append({"text": request.prompt})
    body: dict[str, Any] = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }
    if request.use_web_search:
        body["tools"] = [{"google_search": {}}]
        body["generationConfig"].pop("responseMimeType", None)
    data = _post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{quote(provider.model)}:generateContent?key={api_key}",
        body,
        {"Content-Type": "application/json"},
    )
    parts_out = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "\n".join(part.get("text", "") for part in parts_out).strip()


_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_CREDIT_EXHAUSTED_STATUS = {402, 403}


def _post_json(url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    timeout = int(os.getenv("WIRELESS_TAXONOMY_LLM_TIMEOUT_SECONDS", "120"))
    attempts = max(1, int(os.getenv("WIRELESS_TAXONOMY_LLM_MAX_RETRIES", "4")))
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            if exc.code in _CREDIT_EXHAUSTED_STATUS and _looks_like_quota(detail):
                raise CreditExhaustedError(
                    f"Credits/quota exhausted (HTTP {exc.code}). "
                    "Top up your account and re-run; the cache will resume where you left off."
                ) from exc
            last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
            if exc.code not in _RETRYABLE_STATUS:
                raise last_error from exc
        except URLError as exc:
            last_error = RuntimeError(str(exc))
        except ConnectionError as exc:
            last_error = RuntimeError(str(exc))
        if attempt + 1 < attempts:
            time.sleep(min(2.0 * (2**attempt), 30.0))
    assert last_error is not None
    raise last_error


def _looks_like_quota(detail: str) -> bool:
    """Heuristic check whether an error body indicates quota/credit exhaustion."""
    lower = detail.lower()
    return any(term in lower for term in (
        "quota", "billing", "exceeded", "insufficient", "payment",
        "resource_exhausted", "rate_limit", "limit exceeded",
    ))


def _parse_json_content(content: str) -> dict[str, Any] | list[Any] | None:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min((idx for idx in [text.find("{"), text.find("[")] if idx >= 0), default=-1)
        end = max(text.rfind("}"), text.rfind("]"))
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None
