from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode

from wireless_taxonomy.analyze.text_match import title_matches

FetchJson = Callable[[str], dict[str, Any]]

_JATS_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_MIN_ABSTRACT_CHARS = 40


@dataclass(frozen=True)
class AbstractResult:
    abstract: str
    provider: str
    source_url: str


class AbstractEnricher:
    """Fetches paper abstracts from open metadata APIs (no ACM full text).

    Abstracts are bibliographic metadata, so OpenAlex / Crossref / Semantic
    Scholar expose them even for papers whose PDFs are paywalled. Providers are
    tried in order and the first usable abstract wins.
    """

    def __init__(self, fetch_json: FetchJson | None = None, providers: list[str] | None = None) -> None:
        self.fetch_json = fetch_json or _default_fetch_json
        self.providers = providers or ["openalex", "crossref", "semantic_scholar"]
        self._mailto = (os.getenv("WIRELESS_TAXONOMY_CONTACT_EMAIL") or "").strip()

    def fetch(self, title: str | None, doi: str | None) -> AbstractResult | None:
        for provider in self.providers:
            handler = getattr(self, f"_{provider}", None)
            if handler is None:
                continue
            try:
                result = handler(title, doi)
            except Exception:
                result = None
            if result is not None:
                return result
        return None

    def _openalex(self, title: str | None, doi: str | None) -> AbstractResult | None:
        payload: dict[str, Any] = {}
        source_url = ""
        if doi:
            source_url = f"https://api.openalex.org/works/https://doi.org/{quote(doi, safe='')}"
            payload = self.fetch_json(self._with_mailto(source_url))
        if not _openalex_abstract(payload) and title:
            source_url = "https://api.openalex.org/works?" + urlencode({"search": title, "per-page": "1"})
            search = self.fetch_json(self._with_mailto(source_url))
            results = search.get("results") if isinstance(search.get("results"), list) else []
            payload = results[0] if results and isinstance(results[0], dict) else {}
            if payload and not title_matches(title, _str(payload.get("title"))):
                return None
        abstract = _openalex_abstract(payload)
        if abstract:
            return AbstractResult(abstract, "openalex", source_url)
        return None

    def _crossref(self, title: str | None, doi: str | None) -> AbstractResult | None:
        if not doi:
            return None
        source_url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
        payload = self.fetch_json(self._with_mailto(source_url))
        message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
        abstract = _strip_jats(_str(message.get("abstract")))
        if abstract and len(abstract) >= _MIN_ABSTRACT_CHARS:
            return AbstractResult(abstract, "crossref", source_url)
        return None

    def _semantic_scholar(self, title: str | None, doi: str | None) -> AbstractResult | None:
        if doi:
            source_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{quote(doi, safe='')}?fields=title,abstract"
            payload = self.fetch_json(source_url)
            abstract = _str(payload.get("abstract"))
            if abstract and len(abstract) >= _MIN_ABSTRACT_CHARS:
                return AbstractResult(abstract, "semantic_scholar", source_url)
        if title:
            source_url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urlencode(
                {"query": title, "limit": "1", "fields": "title,abstract"}
            )
            payload = self.fetch_json(source_url)
            data = payload.get("data") if isinstance(payload.get("data"), list) else []
            first = data[0] if data and isinstance(data[0], dict) else {}
            if first and not title_matches(title, _str(first.get("title"))):
                return None
            abstract = _str(first.get("abstract"))
            if abstract and len(abstract) >= _MIN_ABSTRACT_CHARS:
                return AbstractResult(abstract, "semantic_scholar", source_url)
        return None

    def _with_mailto(self, url: str) -> str:
        if not self._mailto:
            return url
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}mailto={quote(self._mailto)}"


def _openalex_abstract(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""
    index = payload.get("abstract_inverted_index")
    if not isinstance(index, dict) or not index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, locs in index.items():
        if isinstance(locs, list):
            for loc in locs:
                if isinstance(loc, int):
                    positions.append((loc, word))
    if not positions:
        return ""
    positions.sort(key=lambda item: item[0])
    text = " ".join(word for _, word in positions)
    return _WS_RE.sub(" ", text).strip()


def _strip_jats(value: str) -> str:
    if not value:
        return ""
    text = _JATS_TAG_RE.sub(" ", value)
    return _WS_RE.sub(" ", text).strip()


def _str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _default_fetch_json(url: str) -> dict[str, Any]:
    import json as _json
    import time
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    headers = {"User-Agent": "wireless-taxonomy/0.1"}
    if "api.semanticscholar.org" in url:
        api_key = (os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY") or "").strip()
        if api_key:
            headers["x-api-key"] = api_key
    attempts = max(1, int(os.getenv("WIRELESS_TAXONOMY_FETCH_MAX_RETRIES", "3")))
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlopen(Request(url, headers=headers), timeout=30) as response:
                payload = _json.loads(response.read().decode("utf-8"))
            return payload if isinstance(payload, dict) else {}
        except HTTPError as exc:
            last_error = exc
            if exc.code not in _RETRYABLE_STATUS:
                raise
        except URLError as exc:
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(min(1.5 * (2**attempt), 20.0))
    if last_error is not None:
        raise last_error
    return {}
