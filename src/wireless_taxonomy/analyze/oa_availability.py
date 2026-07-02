
import os
import re
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
        self._router = router
        if providers is None:
            # "usenix" is authoritative-and-free for NSDI/OSDI/ATC/Security
            # (USENIX hosts every paper open-access), so it runs first and only
            # fires for usenix.org URLs. Unpaywall is the canonical legal-OA
            # resolver but requires an email; without one it is skipped and the
            # others carry the load.
            providers = ["usenix", "unpaywall", "openalex", "semantic_scholar", "arxiv"]
            if web_search:
                # Last-resort: LLM web-search grounding finds author-hosted
                # postprints the OA indexes miss. Runs only when everything
                # else failed, and its URL is verified by downloading the PDF
                # and title-checking the first pages before being trusted.
                providers.append("llm_web_search")
        self.providers = providers

    def resolve(self, title: str | None, doi: str | None, url: str | None = None) -> OaResult:
        if self.cache is not None:
            cached = self.cache.get_oa(title, doi)
            if cached is not None:
                # A cached "closed" verdict predates web search if web search is
                # now enabled — re-resolve so the new provider gets a chance.
                stale_closed = (
                    "llm_web_search" in self.providers
                    and not cached.get("fetchable")
                    and not cached.get("web_search_attempted")
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
        for provider in self.providers:
            handler = getattr(self, f"_{provider}", None)
            if handler is None:
                continue
            try:
                found = handler(title, doi, url)
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
                    "web_search_attempted": "llm_web_search" in self.providers,
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
            return None
        from wireless_taxonomy.llm import LlmRequest

        prompt = (
            "Find a direct, publicly downloadable PDF for this exact research paper:\n"
            f"  Title: {title}\n"
            + (f"  DOI: {doi}\n" if doi else "")
            + "\n"
            "Look for author-hosted copies (personal/university homepages), "
            "institutional repositories, or preprint servers. The URL must point "
            "directly at a .pdf file (not a landing page, not ResearchGate, not "
            "IEEE Xplore or ACM DL which block downloads).\n\n"
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
        except Exception:
            return None
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
