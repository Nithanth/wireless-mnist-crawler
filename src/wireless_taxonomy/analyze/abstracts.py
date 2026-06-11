from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode

from wireless_taxonomy.analyze.text_match import title_matches

FetchJson = Callable[[str], dict[str, Any]]
FetchJsonPost = Callable[[str, dict[str, Any]], Any]
FetchText = Callable[[str], str]

_JATS_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_MIN_ABSTRACT_CHARS = 40

# Semantic Scholar's batch endpoint returns abstracts for up to 500 DOIs in a
# single request. Querying it once per conference (instead of one GET per paper)
# is both far faster and, crucially, avoids the per-request 429 throttling that
# otherwise drops most abstracts on a shared egress IP.
_S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,abstract"
_S2_BATCH_SIZE = 400

_ARXIV_ENTRY_RE = re.compile(r"<entry>(.*?)</entry>", re.DOTALL)
_ARXIV_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.DOTALL)
_ARXIV_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)


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
        cache: Any | None = None,
        fetch_json_post: FetchJsonPost | None = None,
    ) -> None:
        self.fetch_json = fetch_json or _default_fetch_json
        self.fetch_text = fetch_text or _default_fetch_text
        self.fetch_json_post = fetch_json_post or _default_fetch_json_post
        self.cache = cache
        # Abstracts fetched up-front by the Semantic Scholar batch endpoint,
        # keyed by normalized DOI. Checked before the per-paper provider chain
        # so a single batched request serves the whole conference.
        self._batch_abstracts: dict[str, AbstractResult] = {}
        # "usenix" is a page-scrape fallback that only fires when a USENIX paper
        # URL is supplied; it's a cheap no-op otherwise, so trying it first lets
        # USENIX papers (NSDI/OSDI/ATC, which carry no DOI) resolve immediately
        # instead of burning 429-throttled title searches against the JSON APIs.
        # "arxiv" is a title search (preprint coverage); "acm" is an opt-in,
        # best-effort browser scrape (ACM is Cloudflare-protected) enabled via
        # WIRELESS_TAXONOMY_ACM_BROWSER=1.
        if providers is None:
            providers = ["usenix", "openalex", "crossref", "semantic_scholar", "arxiv"]
            if _acm_browser_enabled():
                providers.append("acm")
        self.providers = providers
        self._mailto = (os.getenv("WIRELESS_TAXONOMY_CONTACT_EMAIL") or "").strip()

    def fetch(
        self, title: str | None, doi: str | None, url: str | None = None
    ) -> AbstractResult | None:
        if self.cache is not None:
            cached = self.cache.get_abstract(title, doi)
            if cached is not None:
                # A cached miss (no provider had an abstract) is remembered so
                # re-runs don't pay the full no-hit chain cost again.
                if not cached.get("abstract") or cached.get("provider") == "miss":
                    return None
                return AbstractResult(
                    cached.get("abstract", ""),
                    cached.get("provider", "cache"),
                    cached.get("source_url", ""),
                )
        if doi:
            batched = self._batch_abstracts.get(_norm_doi(doi))
            if batched is not None:
                if self.cache is not None:
                    self.cache.set_abstract(
                        title,
                        doi,
                        {
                            "abstract": batched.abstract,
                            "provider": batched.provider,
                            "source_url": batched.source_url,
                        },
                    )
                return batched
        for provider in self.providers:
            handler = getattr(self, f"_{provider}", None)
            if handler is None:
                continue
            try:
                result = handler(title, doi, url)
            except Exception:
                result = None
            if result is not None:
                if self.cache is not None:
                    self.cache.set_abstract(
                        title,
                        doi,
                        {
                            "abstract": result.abstract,
                            "provider": result.provider,
                            "source_url": result.source_url,
                        },
                    )
                return result
        if self.cache is not None:
            self.cache.set_abstract(title, doi, {"abstract": "", "provider": "miss", "source_url": ""})
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

    def _arxiv(
        self, title: str | None, doi: str | None, url: str | None = None
    ) -> AbstractResult | None:
        """Search arXiv by title and return the abstract for a matching preprint.

        arXiv's API has no DOI lookup for non-arXiv DOIs, so we query by title
        words and guard with a title match (arXiv's relevance ranking otherwise
        returns an unrelated top hit). Helps preprint-heavy systems papers; ACM
        measurement papers are rarely on arXiv.
        """
        if not title or not title.strip():
            return None
        words = re.findall(r"[A-Za-z0-9]+", title)
        if not words:
            return None
        query = " AND ".join(f"all:{word}" for word in words[:8])
        source_url = "https://export.arxiv.org/api/query?" + urlencode(
            {"search_query": query, "max_results": "1"}
        )
        xml = self.fetch_text(source_url)
        if not xml:
            return None
        entry = _ARXIV_ENTRY_RE.search(xml)
        if not entry:
            return None
        block = entry.group(1)
        cand_title = _ARXIV_TITLE_RE.search(block)
        cand_summary = _ARXIV_SUMMARY_RE.search(block)
        if not cand_title or not cand_summary:
            return None
        if not title_matches(title, _WS_RE.sub(" ", cand_title.group(1)).strip()):
            return None
        abstract = _WS_RE.sub(" ", cand_summary.group(1)).strip()
        if abstract and len(abstract) >= _MIN_ABSTRACT_CHARS:
            return AbstractResult(abstract, "arxiv", source_url)
        return None

    def _acm(
        self, title: str | None, doi: str | None, url: str | None = None
    ) -> AbstractResult | None:
        """Best-effort ACM Digital Library abstract scrape via a headless browser.

        ACM (IMC/SIGCOMM/MobiCom) paywalls full text and, crucially, sits behind
        Cloudflare bot protection that blocks plain HTTP *and* automated browsers
        in most environments. This provider is therefore **opt-in** (set
        ``WIRELESS_TAXONOMY_ACM_BROWSER=1`` and install the optional ``[acm]``
        extra) and degrades to ``None`` when the challenge can't be cleared, so
        it never breaks the chain. The abstract lives in the ``#abstract`` block
        of the ``/doi/abs/<doi>`` page.
        """
        if not doi:
            return None
        page_url = f"https://dl.acm.org/doi/abs/{quote(doi, safe='')}"
        html = _fetch_acm_browser(page_url)
        if not html:
            return None
        abstract = _acm_abstract(html)
        if abstract and len(abstract) >= _MIN_ABSTRACT_CHARS:
            return AbstractResult(abstract, "acm", page_url)
        return None

    def prefetch_semantic_scholar(self, items: list[tuple[str | None, str | None]]) -> int:
        """Batch-fetch abstracts by DOI from Semantic Scholar in one request.

        ``items`` is a list of ``(title, doi)`` pairs. Papers whose DOI yields an
        abstract are stored (and cached, if a cache is attached) so the later
        per-paper ``fetch`` short-circuits without a network call. Papers the
        batch misses fall through to the normal provider chain. The batch call is
        best-effort: any error (including an unrecoverable 429) leaves the
        per-paper path untouched.
        """
        by_doi: dict[str, tuple[str | None, str]] = {}
        for title, doi in items:
            norm = _norm_doi(doi)
            if norm and norm not in by_doi:
                by_doi[norm] = (title, (doi or "").strip())
        if not by_doi:
            return 0
        ordered = list(by_doi.values())
        stored = 0
        for start in range(0, len(ordered), _S2_BATCH_SIZE):
            chunk = ordered[start : start + _S2_BATCH_SIZE]
            ids = [f"DOI:{doi}" for _title, doi in chunk]
            try:
                records = self.fetch_json_post(_S2_BATCH_URL, {"ids": ids})
            except Exception:
                records = None
            if not isinstance(records, list):
                continue
            for (title, doi), record in zip(chunk, records):
                if not isinstance(record, dict):
                    continue
                abstract = _str(record.get("abstract"))
                if not abstract or len(abstract) < _MIN_ABSTRACT_CHARS:
                    continue
                result = AbstractResult(abstract, "semantic_scholar", _S2_BATCH_URL)
                self._batch_abstracts[_norm_doi(doi)] = result
                if self.cache is not None:
                    self.cache.set_abstract(
                        title,
                        doi,
                        {
                            "abstract": abstract,
                            "provider": "semantic_scholar",
                            "source_url": _S2_BATCH_URL,
                        },
                    )
                stored += 1
        return stored

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

    def __init__(
        self,
        fetch_json: FetchJson | None = None,
        providers: list[str] | None = None,
        cache: Any | None = None,
    ) -> None:
        self.fetch_json = fetch_json or _default_fetch_json
        self.providers = providers or ["crossref", "openalex"]
        self.cache = cache
        self._mailto = (os.getenv("WIRELESS_TAXONOMY_CONTACT_EMAIL") or "").strip()

    def resolve(self, title: str | None) -> DoiResult | None:
        if not title or not title.strip():
            return None
        if self.cache is not None:
            cached = self.cache.get_doi(title)
            if cached is not None:
                if not cached.get("doi") or cached.get("provider") == "miss":
                    return None  # remembered miss: don't re-query
                return DoiResult(
                    cached.get("doi", ""),
                    cached.get("provider", "cache"),
                    cached.get("source_url", ""),
                )
        for provider in self.providers:
            handler = getattr(self, f"_{provider}", None)
            if handler is None:
                continue
            try:
                result = handler(title)
            except Exception:
                result = None
            if result is not None:
                if self.cache is not None:
                    self.cache.set_doi(
                        title,
                        {
                            "doi": result.doi,
                            "provider": result.provider,
                            "source_url": result.source_url,
                        },
                    )
                return result
        if self.cache is not None:
            self.cache.set_doi(title, {"doi": "", "provider": "miss", "source_url": ""})
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


def _norm_doi(doi: str | None) -> str:
    return (doi or "").strip().lower()


_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRY_WAIT = 30.0


def _retry_wait_seconds(exc: Exception, attempt: int) -> float:
    """Backoff before a retry, honoring a server ``Retry-After`` header on 429.

    Rate-limited APIs (OpenAlex, Semantic Scholar) often return ``Retry-After``;
    respecting it recovers throttled requests that plain exponential backoff
    would give up on, which is the difference between a paper getting an abstract
    and silently becoming a "miss".
    """
    headers = getattr(exc, "headers", None)
    if headers is not None:
        raw = headers.get("Retry-After")
        if raw:
            try:
                return min(float(raw), _MAX_RETRY_WAIT)
            except (TypeError, ValueError):
                pass
    return min(1.5 * (2**attempt), 20.0)


def _s2_headers() -> dict[str, str]:
    headers = {"User-Agent": "wireless-taxonomy/0.1"}
    api_key = (os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY") or "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _default_fetch_json(url: str) -> dict[str, Any]:
    import http.client
    import json as _json
    import time
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    headers = {"User-Agent": "wireless-taxonomy/0.1"}
    if "api.semanticscholar.org" in url:
        headers = _s2_headers()
    attempts = max(1, int(os.getenv("WIRELESS_TAXONOMY_FETCH_MAX_RETRIES", "3")))
    last_error: Exception | None = None
    for attempt in range(attempts):
        wait = min(1.5 * (2**attempt), 20.0)
        try:
            with urlopen(Request(url, headers=headers), timeout=30) as response:
                payload = _json.loads(response.read().decode("utf-8"))
            return payload if isinstance(payload, dict) else {}
        except HTTPError as exc:
            last_error = exc
            if exc.code not in _RETRYABLE_STATUS:
                raise
            wait = _retry_wait_seconds(exc, attempt)
        except (URLError, http.client.HTTPException, ConnectionError) as exc:
            # Transient connection drops (e.g. RemoteDisconnected from
            # getresponse(), which urllib leaves unwrapped) are retried instead
            # of crashing a long multi-conference run.
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(wait)
    if last_error is not None:
        raise last_error
    return {}


def _default_fetch_json_post(url: str, body: dict[str, Any]) -> Any:
    import http.client
    import json as _json
    import time
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    headers = _s2_headers() if "api.semanticscholar.org" in url else {"User-Agent": "wireless-taxonomy/0.1"}
    headers["Content-Type"] = "application/json"
    data = _json.dumps(body).encode("utf-8")
    attempts = max(1, int(os.getenv("WIRELESS_TAXONOMY_FETCH_MAX_RETRIES", "3")))
    last_error: Exception | None = None
    for attempt in range(attempts):
        wait = min(1.5 * (2**attempt), 20.0)
        try:
            with urlopen(Request(url, data=data, headers=headers, method="POST"), timeout=60) as response:
                return _json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            last_error = exc
            if exc.code not in _RETRYABLE_STATUS:
                raise
            wait = _retry_wait_seconds(exc, attempt)
        except (URLError, http.client.HTTPException, ConnectionError) as exc:
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(wait)
    if last_error is not None:
        raise last_error
    return None


_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_USENIX_FIELD_RE = re.compile(r"field-name-field-paper-description\b")
_H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
_USENIX_TRAILERS = ("BibTeX", "@inproceedings", "Open Access", "Presentation Video", "Download")
_USENIX_VENUE_TAIL_RE = re.compile(r"\s+[A-Z]{2,6}\s*['\u2019]?\s*\d{2}\s*$")


def _default_fetch_text(url: str) -> str:
    import http.client
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
        except (URLError, http.client.HTTPException, ConnectionError) as exc:
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


_ACM_ABS_META_RE = re.compile(
    r'<meta[^>]+name=["\'](?:dc\.Description|description)["\'][^>]+content=["\'](.*?)["\']',
    re.IGNORECASE | re.DOTALL,
)
_ACM_ABS_BLOCK_RE = re.compile(
    r'(?:id=["\']abstract["\']|class=["\'][^"\']*abstractInFull[^"\']*["\'])(.*?)</section>',
    re.IGNORECASE | re.DOTALL,
)


def _acm_browser_enabled() -> bool:
    return (os.getenv("WIRELESS_TAXONOMY_ACM_BROWSER") or "").strip().lower() in {"1", "true", "yes"}


def _acm_abstract(html: str) -> str:
    """Pull the abstract from an ACM Digital Library paper page's HTML."""
    if not html:
        return ""
    block = _ACM_ABS_BLOCK_RE.search(html)
    if block:
        text = _strip_jats(block.group(1))
        if len(text) >= _MIN_ABSTRACT_CHARS:
            return text
    meta = _ACM_ABS_META_RE.search(html)
    if meta:
        return _strip_jats(meta.group(1))
    return ""


def _acm_chrome_executable() -> str | None:
    """Locate a Chrome/Chromium binary Playwright can drive, if any."""
    import glob
    import shutil

    explicit = (os.getenv("WIRELESS_TAXONOMY_CHROME_PATH") or "").strip()
    if explicit and os.path.exists(explicit):
        return explicit
    candidates = sorted(
        glob.glob("/opt/.devin/chrome/chrome/*/chrome-linux64/chrome"), reverse=True
    )
    for path in candidates:
        if os.path.exists(path):
            return path
    for name in ("google-chrome-stable", "google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _fetch_acm_browser(url: str) -> str:
    """Load an ACM page in a headless browser and return its HTML.

    Returns ``""`` when Playwright isn't installed, no Chrome is found, or the
    Cloudflare bot challenge can't be cleared (the common case) -- so the ACM
    provider degrades gracefully instead of raising.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return ""
    chrome = _acm_chrome_executable()
    timeout_ms = int(os.getenv("WIRELESS_TAXONOMY_ACM_TIMEOUT_MS", "60000"))
    launch_kwargs: dict[str, Any] = {
        "headless": True,
        "args": ["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"],
    }
    if chrome:
        launch_kwargs["executable_path"] = chrome
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(**launch_kwargs)
            try:
                page = browser.new_page(user_agent=_BROWSER_UA)
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                for _ in range(6):
                    page.wait_for_timeout(2000)
                    if "just a moment" not in (page.title() or "").lower():
                        break
                if "just a moment" in (page.title() or "").lower():
                    return ""  # Cloudflare challenge not cleared
                return page.content() or ""
            finally:
                browser.close()
    except Exception:
        return ""
