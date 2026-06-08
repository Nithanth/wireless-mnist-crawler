from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode

from wireless_taxonomy.analyze.text_match import title_matches

FetchJson = Callable[[str], dict[str, Any]]
FetchText = Callable[[str], str]

_JATS_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_MIN_ABSTRACT_CHARS = 40


@dataclass(frozen=True)
class AbstractResult:
    abstract: str
    provider: str
    source_url: str


@dataclass(frozen=True)
class DoiResult:
    doi: str
    provider: str
    source_url: str


class AbstractEnricher:
    """Fetches paper abstracts from open metadata APIs (no ACM full text).

    Abstracts are bibliographic metadata, so OpenAlex / Crossref / Semantic
    Scholar expose them even for papers whose PDFs are paywalled. Providers are
    tried in order and the first usable abstract wins.
    """

    def __init__(
        self,
        fetch_json: FetchJson | None = None,
        providers: list[str] | None = None,
        fetch_text: FetchText | None = None,
    ) -> None:
        self.fetch_json = fetch_json or _default_fetch_json
        self.fetch_text = fetch_text or _default_fetch_text
        # "usenix" is a page-scrape fallback: it only fires when a USENIX paper
        # URL is supplied, so it costs nothing for venues the JSON APIs cover.
        self.providers = providers or ["openalex", "crossref", "semantic_scholar", "usenix"]
        self._mailto = (os.getenv("WIRELESS_TAXONOMY_CONTACT_EMAIL") or "").strip()

    def fetch(
        self, title: str | None, doi: str | None, url: str | None = None
    ) -> AbstractResult | None:
        for provider in self.providers:
            handler = getattr(self, f"_{provider}", None)
            if handler is None:
                continue
            try:
                result = handler(title, doi, url)
            except Exception:
                result = None
            if result is not None:
                return result
        return None

    def _openalex(
        self, title: str | None, doi: str | None, url: str | None = None
    ) -> AbstractResult | None:
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

    def _crossref(
        self, title: str | None, doi: str | None, url: str | None = None
    ) -> AbstractResult | None:
        if not doi:
            return None
        source_url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
        payload = self.fetch_json(self._with_mailto(source_url))
        message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
        abstract = _strip_jats(_str(message.get("abstract")))
        if abstract and len(abstract) >= _MIN_ABSTRACT_CHARS:
            return AbstractResult(abstract, "crossref", source_url)
        return None

    def _semantic_scholar(
        self, title: str | None, doi: str | None, url: str | None = None
    ) -> AbstractResult | None:
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

    def _usenix(
        self, title: str | None, doi: str | None, url: str | None = None
    ) -> AbstractResult | None:
        """Scrape the abstract from a USENIX paper page.

        USENIX (NSDI/OSDI/ATC/Security) deposits no DOIs in DBLP, so the JSON
        metadata APIs frequently miss these abstracts. DBLP, however, links the
        paper's ``ee`` directly to its USENIX page, whose body holds the full
        abstract. This fallback only fires for those URLs.
        """
        if not url or "usenix.org" not in url:
            return None
        html = self.fetch_text(url)
        if not html:
            return None
        page_title = _html_h1(html)
        if title and page_title and not title_matches(title, page_title):
            return None
        abstract = _usenix_abstract(html)
        if abstract and len(abstract) >= _MIN_ABSTRACT_CHARS:
            return AbstractResult(abstract, "usenix", url)
        return None

    def _with_mailto(self, url: str) -> str:
        if not self._mailto:
            return url
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}mailto={quote(self._mailto)}"


class DoiResolver:
    """Resolves a DOI for a paper from its title via open metadata APIs.

    Used to backfill DOIs for papers whose source list carries none (notably
    USENIX/NSDI, which DBLP indexes without DOIs). Crossref's bibliographic
    query is the primary source; OpenAlex is the fallback. The candidate title
    must match the query title (guards against the API returning a near-miss).
    """

    def __init__(self, fetch_json: FetchJson | None = None, providers: list[str] | None = None) -> None:
        self.fetch_json = fetch_json or _default_fetch_json
        self.providers = providers or ["crossref", "openalex"]
        self._mailto = (os.getenv("WIRELESS_TAXONOMY_CONTACT_EMAIL") or "").strip()

    def resolve(self, title: str | None) -> DoiResult | None:
        if not title or not title.strip():
            return None
        for provider in self.providers:
            handler = getattr(self, f"_{provider}", None)
            if handler is None:
                continue
            try:
                result = handler(title)
            except Exception:
                result = None
            if result is not None:
                return result
        return None

    def _crossref(self, title: str) -> DoiResult | None:
        source_url = "https://api.crossref.org/works?" + urlencode(
            {"query.bibliographic": title, "rows": "1", "select": "DOI,title"}
        )
        payload = self.fetch_json(self._with_mailto(source_url))
        message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
        items = message.get("items") if isinstance(message.get("items"), list) else []
        first = items[0] if items and isinstance(items[0], dict) else {}
        candidate_titles = first.get("title") if isinstance(first.get("title"), list) else []
        candidate_title = _str(candidate_titles[0]) if candidate_titles else ""
        doi = _str(first.get("DOI"))
        if doi and candidate_title and title_matches(title, candidate_title):
            return DoiResult(doi.lower(), "crossref", source_url)
        return None

    def _openalex(self, title: str) -> DoiResult | None:
        source_url = "https://api.openalex.org/works?" + urlencode(
            {"search": title, "per-page": "1", "select": "doi,title"}
        )
        payload = self.fetch_json(self._with_mailto(source_url))
        results = payload.get("results") if isinstance(payload.get("results"), list) else []
        first = results[0] if results and isinstance(results[0], dict) else {}
        candidate_title = _str(first.get("title"))
        doi = _normalize_doi_url(_str(first.get("doi")))
        if doi and candidate_title and title_matches(title, candidate_title):
            return DoiResult(doi.lower(), "openalex", source_url)
        return None

    def _with_mailto(self, url: str) -> str:
        if not self._mailto:
            return url
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}mailto={quote(self._mailto)}"


def _normalize_doi_url(value: str) -> str:
    text = (value or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.lower().startswith(prefix):
            return text[len(prefix):]
    return text


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


_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_USENIX_FIELD_RE = re.compile(r"field-name-field-paper-description\b")
_H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
_USENIX_TRAILERS = ("BibTeX", "@inproceedings", "Open Access", "Presentation Video", "Download")
_USENIX_VENUE_TAIL_RE = re.compile(r"\s+[A-Z]{2,6}\s*['\u2019]?\s*\d{2}\s*$")


def _default_fetch_text(url: str) -> str:
    import time
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    headers = {
        "User-Agent": _BROWSER_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    attempts = max(1, int(os.getenv("WIRELESS_TAXONOMY_FETCH_MAX_RETRIES", "3")))
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlopen(Request(url, headers=headers), timeout=30) as response:
                return response.read().decode("utf-8", "ignore")
        except HTTPError as exc:
            last_error = exc
            if exc.code not in _RETRYABLE_STATUS:
                return ""
        except URLError as exc:
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(min(1.5 * (2**attempt), 20.0))
    if last_error is not None:
        return ""
    return ""


def _html_h1(html: str) -> str:
    match = _H1_RE.search(html or "")
    if not match:
        return ""
    return _strip_jats(match.group(1))


def _usenix_abstract(html: str) -> str:
    """Extract the abstract text from a USENIX paper page.

    The abstract sits in a ``field-name-field-paper-description`` block. We slice
    from that marker, strip tags, and cut at the page's trailing boilerplate
    (BibTeX, media links) so only the abstract prose remains.
    """
    if not html:
        return ""
    match = _USENIX_FIELD_RE.search(html)
    if not match:
        return ""
    # Start after the opening tag closes so the div's own class attribute text
    # isn't captured as prose.
    tag_end = html.find(">", match.end())
    start = tag_end + 1 if tag_end != -1 else match.end()
    segment = html[start : start + 12000]
    text = _strip_jats(segment)
    for trailer in _USENIX_TRAILERS:
        idx = text.find(trailer)
        if idx > 0:
            text = text[:idx]
    text = _USENIX_VENUE_TAIL_RE.sub("", text.strip())
    return text.strip()
