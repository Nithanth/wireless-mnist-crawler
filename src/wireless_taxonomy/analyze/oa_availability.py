
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode

from wireless_taxonomy.analyze.abstracts import (
    _default_fetch_json,
    _default_fetch_text,
    _str,
)
from wireless_taxonomy.analyze.text_match import title_matches
from wireless_taxonomy.llm import CreditExhaustedError

FetchJson = Callable[[str], dict[str, Any]]
FetchText = Callable[[str], str]

_WS_RE = re.compile(r"\s+")
_ARXIV_ENTRY_RE = re.compile(r"<entry>(.*?)</entry>", re.DOTALL)
_ARXIV_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.DOTALL)
_ARXIV_ID_RE = re.compile(r"<id>(.*?)</id>", re.DOTALL)


def _is_acm_blocked(url: str) -> bool:
    """True for URLs that resolve to ACM's programmatically-blocked PDFs.

    OpenAlex reports ``https://doi.org/10.1145/...`` as OA, but those DOIs
    redirect to dl.acm.org landing pages that the pipeline cannot download.
    """
    return "dl.acm.org" in url or "doi.org/10.1145" in url

# OpenAlex / Unpaywall report this on every work. Anything other than "closed"
# means a legally hosted copy exists somewhere (gold/hybrid = publisher, green =
# repository/preprint, bronze = free-to-read on the publisher site w/o a license).
_OA_STATUSES = {"gold", "green", "hybrid", "bronze", "diamond"}


@dataclass(frozen=True)
class OaResult:
    """Whether a legally fetchable open-access full text exists for a paper."""

    fetchable: bool
    oa_status: str
    license: str
    pdf_url: str
    provider: str
    source_url: str


_NOT_FETCHABLE = OaResult(False, "closed", "", "", "none", "")


class _WebSearchError(RuntimeError):
    """LLM web search failed to RUN (rate limit, no router, network error).

    Distinct from "ran and found nothing": a failed attempt must not be cached
    as ``web_search_attempted`` or the paper is permanently marked closed and
    the search is never retried.
    """


class OpenAccessResolver:
    """Detects a legally fetchable open-access full text for a paper.

    Reads open-access *status metadata* (it never scrapes paywalled PDFs):
    Unpaywall and OpenAlex expose ``is_oa`` + ``oa_status`` + a hosted PDF URL,
    Semantic Scholar exposes ``openAccessPdf``, and arXiv presence implies a
    freely hosted preprint. Providers are tried in order and the first that
    reports a hosted OA copy wins; a paper no provider can place is reported as
    closed (not fetchable).
    """

    def __init__(
        self,
        fetch_json: FetchJson | None = None,
        fetch_text: FetchText | None = None,
        providers: list[str] | None = None,
        cache: Any | None = None,
        web_search: bool = False,
        router: Any | None = None,
    ) -> None:
        self.fetch_json = fetch_json or _default_fetch_json
        self.fetch_text = fetch_text or _default_fetch_text
        self.cache = cache
        self._mailto = (os.getenv("WIRELESS_TAXONOMY_CONTACT_EMAIL") or "").strip()
        self._cse_key = (os.getenv("GOOGLE_CSE_API_KEY") or "").strip()
        self._cse_id = (os.getenv("GOOGLE_CSE_ID") or "").strip()
        self._core_key = (os.getenv("CORE_API_KEY") or "").strip()
        self._brave_key = (os.getenv("BRAVE_SEARCH_API_KEY") or "").strip()
        self._router = router
        if providers is None:
            # "usenix" is authoritative-and-free for NSDI/OSDI/ATC/Security
            # (USENIX hosts every paper open-access), so it runs first and only
            # fires for usenix.org URLs. Unpaywall is the canonical legal-OA
            # resolver but requires an email; without one it is skipped and the
            # others carry the load. CORE (free, ~300M works from institutional
            # repositories) runs after the indexes and before paid web search,
            # so it absorbs most of the residual for free.
            providers = ["usenix", "unpaywall", "openalex", "semantic_scholar", "arxiv"]
            if self._core_key:
                providers.append("core")
            if web_search:
                # Last-resort web search for author-hosted postprints the OA
                # indexes miss. Brave (whole-web index) is preferred; Google
                # CSE (domain-whitelisted since Google deprecated whole-web
                # CSEs) runs as an extra layer if configured; Gemini grounding
                # only when neither is available. Either way, a candidate URL
                # is ONLY trusted after downloading the PDF and verifying the
                # paper title appears in its first pages.
                if self._brave_key:
                    providers.append("brave_search")
                if self._cse_key and self._cse_id:
                    providers.append("google_cse")
                if not self._brave_key and not (self._cse_key and self._cse_id):
                    providers.append("llm_web_search")
        self.providers = providers

    _WEB_SEARCH_PROVIDERS = frozenset({"llm_web_search", "google_cse", "brave_search"})

    def resolve(self, title: str | None, doi: str | None, url: str | None = None) -> OaResult:
        ws_providers = sorted(self._WEB_SEARCH_PROVIDERS & set(self.providers))
        has_web_search = bool(ws_providers)
        if self.cache is not None:
            cached = self.cache.get_oa(title, doi)
            if cached is not None:
                # A cached "closed" verdict is stale if web search is now
                # enabled and either (a) it was never web-searched, or (b) it
                # was searched with a different provider set (e.g. the old
                # Gemini grounding) — re-resolve so the new providers get a
                # chance at papers the old search missed.
                stale_closed = (
                    has_web_search
                    and not cached.get("fetchable")
                    and (
                        not cached.get("web_search_attempted")
                        or cached.get("web_search_providers", []) != ws_providers
                    )
                )
                if not stale_closed:
                    return OaResult(
                        bool(cached.get("fetchable")),
                        cached.get("oa_status", "closed"),
                        cached.get("license", ""),
                        cached.get("pdf_url", ""),
                        cached.get("provider", "cache"),
                        cached.get("source_url", ""),
                    )
        result = _NOT_FETCHABLE
        web_search_attempted = has_web_search
        for provider in self.providers:
            handler = getattr(self, f"_{provider}", None)
            if handler is None:
                continue
            try:
                found = handler(title, doi, url)
            except _WebSearchError as wse:
                # Search didn't run (rate limit / config / budget) — leave the
                # cache entry retryable rather than poisoning it as "attempted".
                found = None
                web_search_attempted = False
                msg = str(wse)
                if "budget exhausted" in msg.lower() and not OpenAccessResolver._brave_cap_notified:
                    OpenAccessResolver._brave_cap_notified = True
                    print(
                        "\n  ⚠ BRAVE SEARCH BUDGET EXHAUSTED. Papers beyond this point "
                        "will fall through to CSE/Gemini or stay retryable. Add credits or "
                        "increase WIRELESS_TAXONOMY_BRAVE_MAX_QUERIES to resume Brave usage.",
                        file=sys.stderr,
                    )
                elif "not configured" in msg.lower() and provider == "brave_search":
                    pass  # normal if key absent
            except CreditExhaustedError:
                raise  # checkpoint upstream; never cache a closed verdict
            except Exception:
                found = None
            if found is not None and found.fetchable:
                result = found
                # If the URL points to dl.acm.org, keep trying other providers
                # for a non-ACM mirror (arxiv, institutional repo, etc.) since
                # ACM blocks programmatic PDF downloads.
                if "dl.acm.org" not in (found.pdf_url or ""):
                    break
        if self.cache is not None:
            self.cache.set_oa(
                title,
                doi,
                {
                    "fetchable": result.fetchable,
                    "oa_status": result.oa_status,
                    "license": result.license,
                    "pdf_url": result.pdf_url,
                    "provider": result.provider,
                    "source_url": result.source_url,
                    "web_search_attempted": web_search_attempted,
                    "web_search_providers": ws_providers if web_search_attempted else [],
                },
            )
        return result

    def _usenix(self, title: str | None, doi: str | None, url: str | None) -> OaResult | None:
        # USENIX (NSDI/OSDI/ATC/Security) publishes every paper open-access on
        # its own site. DBLP links the presentation page; we scrape the actual
        # PDF download link from it (pattern: /system/files/<conf>-paper-*.pdf).
        if not url or "usenix.org" not in url:
            return None
        pdf_url = _usenix_pdf_url(url, self.fetch_text)
        if not pdf_url:
            # Fallback: the page itself is still legally fetchable (full text in HTML)
            pdf_url = url
        return OaResult(True, "gold", "usenix-open-access", pdf_url, "usenix", url)

    def _unpaywall(self, title: str | None, doi: str | None, url: str | None) -> OaResult | None:
        if not doi or not self._mailto:
            return None
        source_url = f"https://api.unpaywall.org/v2/{quote(doi, safe='')}?" + urlencode({"email": self._mailto})
        payload = self.fetch_json(source_url)
        if not payload.get("is_oa"):
            return None
        location = payload.get("best_oa_location") if isinstance(payload.get("best_oa_location"), dict) else {}
        pdf_url = _str(location.get("url_for_pdf")) or _str(location.get("url"))
        if not pdf_url or _is_acm_blocked(pdf_url):
            return None
        return OaResult(
            True,
            _str(payload.get("oa_status")) or "green",
            _str(location.get("license")),
            pdf_url,
            "unpaywall",
            source_url,
        )

    def _openalex(self, title: str | None, doi: str | None, url: str | None) -> OaResult | None:
        payload: dict[str, Any] = {}
        source_url = ""
        if doi:
            source_url = f"https://api.openalex.org/works/https://doi.org/{quote(doi, safe='')}"
            payload = self.fetch_json(self._with_mailto(source_url))
        if not payload and title:
            source_url = "https://api.openalex.org/works?" + urlencode({"search": title, "per-page": "1"})
            search = self.fetch_json(self._with_mailto(source_url))
            results = search.get("results") if isinstance(search.get("results"), list) else []
            payload = results[0] if results and isinstance(results[0], dict) else {}
            if payload and not title_matches(title, _str(payload.get("title"))):
                return None
        oa = payload.get("open_access") if isinstance(payload.get("open_access"), dict) else {}
        if not oa.get("is_oa"):
            return None
        location = payload.get("best_oa_location") if isinstance(payload.get("best_oa_location"), dict) else {}
        pdf_url = _str(location.get("pdf_url")) or _str(oa.get("oa_url"))
        if not pdf_url or _is_acm_blocked(pdf_url):
            return None
        return OaResult(
            True,
            _str(oa.get("oa_status")) or "green",
            _str(location.get("license")),
            pdf_url,
            "openalex",
            source_url,
        )

    def _semantic_scholar(self, title: str | None, doi: str | None, url: str | None) -> OaResult | None:
        fields = "externalIds,openAccessPdf"
        payload: dict[str, Any] = {}
        source_url = ""
        if doi:
            source_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{quote(doi, safe='')}?fields={fields}"
            payload = self.fetch_json(source_url)
        if not payload.get("openAccessPdf") and title:
            source_url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urlencode(
                {"query": title, "limit": "1", "fields": f"title,{fields}"}
            )
            search = self.fetch_json(source_url)
            data = search.get("data") if isinstance(search.get("data"), list) else []
            first = data[0] if data and isinstance(data[0], dict) else {}
            if first and not title_matches(title, _str(first.get("title"))):
                return None
            payload = first
        oa_pdf = payload.get("openAccessPdf") if isinstance(payload.get("openAccessPdf"), dict) else {}
        pdf_url = _str(oa_pdf.get("url"))
        if not pdf_url or _is_acm_blocked(pdf_url):
            return None
        status = _str(oa_pdf.get("status")).lower()
        return OaResult(
            True,
            status if status in _OA_STATUSES else "green",
            _str(oa_pdf.get("license")),
            pdf_url,
            "semantic_scholar",
            source_url,
        )

    def _arxiv(self, title: str | None, doi: str | None, url: str | None) -> OaResult | None:
        # A DBLP-supplied arXiv link is already a legally hosted preprint.
        if url and "arxiv.org" in url:
            return OaResult(True, "green", "arxiv", _arxiv_pdf_url(url), "arxiv", url)
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
        cand_id = _ARXIV_ID_RE.search(block)
        if not cand_title or not cand_id:
            return None
        if not title_matches(title, _WS_RE.sub(" ", cand_title.group(1)).strip()):
            return None
        return OaResult(
            True,
            "green",
            "arxiv",
            _arxiv_pdf_url(_WS_RE.sub(" ", cand_id.group(1)).strip()),
            "arxiv",
            source_url,
        )

    # CORE's free tier allows ~5 search requests per 10 seconds. A process-wide
    # throttle keeps concurrent workers under that ceiling so CORE lookups
    # succeed instead of burning 429 retries (tune via env if you have a paid key).
    _core_lock = threading.Lock()
    _core_last_call = 0.0

    @classmethod
    def _core_throttle(cls) -> None:
        interval = float(os.getenv("WIRELESS_TAXONOMY_CORE_MIN_INTERVAL_SECONDS", "2.5"))
        with cls._core_lock:
            wait = cls._core_last_call + interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            cls._core_last_call = time.monotonic()

    def _core(self, title: str | None, doi: str | None, url: str | None) -> OaResult | None:
        """CORE (core.ac.uk): free aggregator of ~300M institutional-repository works.

        Searches by title, requires a fuzzy title match on the result metadata,
        then downloads + title-verifies the hosted PDF before trusting it —
        the same guarantee as every other provider.
        """
        if not title or not title.strip():
            return None
        if not self._core_key:
            return None
        source_url = "https://api.core.ac.uk/v3/search/works/?" + urlencode(
            {"q": f'title:"{title}"', "limit": "5"}
        )
        self._core_throttle()
        payload = self.fetch_json(source_url)
        results = payload.get("results") if isinstance(payload.get("results"), list) else []
        from wireless_taxonomy.analyze.dataset_extractor import _fetch_pdf_bytes

        for item in results[:5]:
            if not isinstance(item, dict):
                continue
            if not title_matches(title, _str(item.get("title"))):
                continue
            candidates = [_str(item.get("downloadUrl"))]
            urls = item.get("sourceFulltextUrls")
            if isinstance(urls, list):
                candidates.extend(_str(u) for u in urls)
            for candidate in candidates:
                if not candidate.lower().startswith("http"):
                    continue
                if _is_acm_blocked(candidate) or "ieeexplore.ieee.org" in candidate:
                    continue
                # VERIFIED APPROVAL: download + title-check before trusting.
                if _fetch_pdf_bytes(candidate, expected_title=title) is None:
                    continue
                return OaResult(True, "green", "", candidate, "core", source_url)
        return None

    # Brave's free "Data for Search" tier allows 1 request/second. A
    # process-wide throttle keeps concurrent workers under that ceiling.
    _brave_lock = threading.Lock()
    _brave_last_call = 0.0
    _brave_queries = 0
    _brave_cap_notified = False

    @classmethod
    def _brave_throttle(cls) -> None:
        interval = float(os.getenv("WIRELESS_TAXONOMY_BRAVE_MIN_INTERVAL_SECONDS", "1.1"))
        with cls._brave_lock:
            wait = cls._brave_last_call + interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            cls._brave_last_call = time.monotonic()

    @classmethod
    def _check_brave_cap(cls) -> None:
        max_q = int(os.getenv("WIRELESS_TAXONOMY_BRAVE_MAX_QUERIES", "3000"))
        if max_q <= 0:
            return
        with cls._brave_lock:
            cls._brave_queries += 1
            if cls._brave_queries > max_q:
                raise _WebSearchError(
                    f"Brave Search query budget exhausted ({max_q}/{max_q}). "
                    "Add BRAVE_SEARCH_API_KEY credits or increase WIRELESS_TAXONOMY_BRAVE_MAX_QUERIES."
                )

    def _brave_search(self, title: str | None, doi: str | None, url: str | None) -> OaResult | None:
        """Last-resort provider: Brave Search (whole-web index) finds a PDF URL.

        Runs the same query a human would (``"<title>" filetype:pdf``) against
        Brave's independent web index and tries the top hits in order. A hit is
        ONLY accepted after downloading it and verifying the paper title
        appears in the PDF's first pages (``_fetch_pdf_bytes(expected_title=...)``)
        — a wrong search result can never inject someone else's paper.
        """
        if not title or not title.strip():
            return None
        if not self._brave_key:
            raise _WebSearchError("Brave Search not configured")
        from wireless_taxonomy.analyze.dataset_extractor import _fetch_pdf_bytes

        for query in (f'"{title}" filetype:pdf', f"{title} pdf"):
            source_url = "https://api.search.brave.com/res/v1/web/search?" + urlencode(
                {"q": query, "count": "5"}
            )
            self._brave_throttle()
            self._check_brave_cap()
            try:
                payload = self.fetch_json(source_url)
            except Exception as exc:
                # Search never ran (rate limit / quota / network) — keep the
                # verdict retryable instead of caching it as attempted.
                raise _WebSearchError(str(exc)) from exc
            web = payload.get("web") if isinstance(payload.get("web"), dict) else {}
            results = web.get("results") if isinstance(web.get("results"), list) else []
            for item in results[:5]:
                candidate = _str(item.get("url") if isinstance(item, dict) else "")
                if not candidate.lower().startswith("http"):
                    continue
                if _is_acm_blocked(candidate) or "ieeexplore.ieee.org" in candidate or "researchgate.net" in candidate:
                    continue
                # VERIFIED APPROVAL: download + title-check before trusting.
                if _fetch_pdf_bytes(candidate, expected_title=title) is None:
                    continue
                return OaResult(True, "green", "", candidate, "brave_search", source_url)
        return None

    def _google_cse(self, title: str | None, doi: str | None, url: str | None) -> OaResult | None:
        """Last-resort provider: Google Programmable Search finds a PDF URL.

        Runs the same query a human would (``"<title>" filetype:pdf``) via the
        Custom Search JSON API and tries the top hits in order. A hit is ONLY
        accepted after downloading it and verifying the paper title appears in
        the PDF's first pages (``_fetch_pdf_bytes(expected_title=...)``) — a
        wrong search result can never inject someone else's paper.
        """
        if not title or not title.strip():
            return None
        if not (self._cse_key and self._cse_id):
            raise _WebSearchError("Google CSE not configured")
        from wireless_taxonomy.analyze.dataset_extractor import _fetch_pdf_bytes

        # Exact-phrase PDF query first; a looser query as fallback for titles
        # Google indexes with slightly different punctuation.
        for query in (f'"{title}" filetype:pdf', f"{title} pdf"):
            source_url = "https://www.googleapis.com/customsearch/v1?" + urlencode(
                {"key": self._cse_key, "cx": self._cse_id, "q": query, "num": "5"}
            )
            try:
                payload = self.fetch_json(source_url)
            except Exception as exc:
                # Search never ran (daily quota / network) — keep the verdict
                # retryable instead of caching it as attempted.
                raise _WebSearchError(str(exc)) from exc
            items = payload.get("items") if isinstance(payload.get("items"), list) else []
            for item in items[:5]:
                candidate = _str(item.get("link") if isinstance(item, dict) else "")
                if not candidate.lower().startswith("http"):
                    continue
                if _is_acm_blocked(candidate) or "ieeexplore.ieee.org" in candidate or "researchgate.net" in candidate:
                    continue
                # VERIFIED APPROVAL: download + title-check before trusting.
                if _fetch_pdf_bytes(candidate, expected_title=title) is None:
                    continue
                return OaResult(True, "green", "", candidate, "google_cse", source_url)
        return None

    def _llm_web_search(self, title: str | None, doi: str | None, url: str | None) -> OaResult | None:
        """Last-resort provider: LLM with web-search grounding finds a PDF URL.

        Asks Gemini (google_search tool) for a direct, publicly downloadable
        PDF of this exact paper — author homepages and university mirrors that
        the OA indexes miss. The returned URL is only trusted after actually
        downloading it and verifying the paper title appears in the first
        pages (the same wrong-paper guard extraction uses), so a bad search
        result can never inject someone else's paper.
        """
        if not title or not title.strip():
            return None
        router = self._get_router()
        if router is None:
            raise _WebSearchError("no LLM router available")
        from wireless_taxonomy.llm import LlmRequest

        prompt = (
            "Find a direct, publicly downloadable PDF for this exact research paper.\n"
            f"  Title: {title}\n"
            + (f"  DOI: {doi}\n" if doi else "")
            + "\n"
            "Search strategy:\n"
            "1. Search for the title in quotes plus 'pdf' or 'filetype:pdf'.\n"
            "2. Also try the title plus 'paper.pdf' or the last name of the first author.\n"
            "3. Examine the first 5 results, including any from .edu, university "
            "homepages, institutional repositories, or preprint servers.\n"
            "4. Prefer URLs that end in .pdf and are hosted on author/university pages.\n"
            "\n"
            "Do NOT return ResearchGate, IEEE Xplore, ACM DL, arXiv abstract pages, or "
            "any landing page that requires clicking. The URL must point directly at a "
            "downloadable .pdf file (or redirect immediately to one).\n"
            "\n"
            'Return JSON only: {"pdf_url": "<direct PDF URL or empty string if none found>"}'
        )
        try:
            response = router.complete(
                LlmRequest(
                    task="oa_pdf_web_search",
                    schema_name="PdfUrlSearch",
                    prompt=prompt,
                    use_web_search=True,
                )
            )
        except CreditExhaustedError:
            raise
        except Exception as exc:
            # The search never ran (rate limit, network) — signal so the
            # verdict is not cached as web_search_attempted.
            raise _WebSearchError(str(exc)) from exc
        parsed = response.parsed if isinstance(response.parsed, dict) else {}
        pdf_url = _str(parsed.get("pdf_url"))
        if not pdf_url or not pdf_url.lower().startswith("http"):
            return None
        if _is_acm_blocked(pdf_url) or "ieeexplore.ieee.org" in pdf_url or "researchgate.net" in pdf_url:
            return None
        # Verify by downloading and title-checking — never trust search blindly.
        from wireless_taxonomy.analyze.dataset_extractor import _fetch_pdf_bytes

        if _fetch_pdf_bytes(pdf_url, expected_title=title) is None:
            return None
        return OaResult(True, "green", "", pdf_url, "llm_web_search", "gemini:google_search")

    def _get_router(self) -> Any | None:
        if self._router is None:
            try:
                from wireless_taxonomy.config import load_dotenv, load_llm_settings
                from wireless_taxonomy.llm import LlmRouter

                load_dotenv()
                self._router = LlmRouter(load_llm_settings())
            except Exception:
                return None
        return self._router

    def _with_mailto(self, url: str) -> str:
        if not self._mailto:
            return url
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}mailto={quote(self._mailto)}"


_USENIX_PDF_RE = re.compile(r'href=["\']([^"\']*?/system/files/[^"\']*?\.pdf)', re.IGNORECASE)


def _usenix_pdf_url(page_url: str, fetch_text: Callable[[str], str]) -> str:
    """Scrape the direct PDF download link from a USENIX paper page.

    USENIX hosts PDFs at /system/files/<conf>-paper-<author>.pdf. The
    presentation page links to it. We grab the first .pdf href matching
    /system/files/ and skip slide PDFs (which contain 'slides' in the filename).
    """
    try:
        html = fetch_text(page_url)
    except Exception:
        return ""
    if not html:
        return ""
    matches = _USENIX_PDF_RE.findall(html)
    for href in matches:
        # Skip slides PDFs — we want the paper
        if "slide" in href.lower():
            continue
        # Make absolute if relative
        if href.startswith("/"):
            href = "https://www.usenix.org" + href
        return href
    # If all matches were slides, return the first one anyway (still a valid PDF)
    if matches:
        href = matches[0]
        if href.startswith("/"):
            href = "https://www.usenix.org" + href
        return href
    return ""


def _arxiv_pdf_url(value: str) -> str:
    """Turn an arXiv abs/landing URL or id into its PDF URL."""
    text = (value or "").strip()
    if not text:
        return ""
    text = text.replace("/abs/", "/pdf/")
    if "arxiv.org" in text:
        return text
    arxiv_id = text.rsplit("/", 1)[-1]
    return f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else ""


def summarize(papers: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-paper OA results into coverage counts + percentage."""
    total = len(papers)
    fetchable = sum(1 for p in papers if p.get("fetchable"))
    by_status: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for paper in papers:
        if not paper.get("fetchable"):
            continue
        status = paper.get("oa_status") or "unknown"
        source = paper.get("provider") or "unknown"
        by_status[status] = by_status.get(status, 0) + 1
        by_source[source] = by_source.get(source, 0) + 1
    pct = round(100.0 * fetchable / total, 1) if total else 0.0
    return {
        "total_papers": total,
        "fetchable": fetchable,
        "fetchable_pct": pct,
        "by_oa_status": dict(sorted(by_status.items())),
        "by_source": dict(sorted(by_source.items())),
    }
