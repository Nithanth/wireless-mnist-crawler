from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from wireless_taxonomy.analyze import full_text_resolvers as resolvers
from wireless_taxonomy.analyze.pdf_text import (
    _unescape_pdf_string,
    extract_pdf_text,
    fallback_pdf_text as _fallback_pdf_text,
)
from wireless_taxonomy.analyze.paper_text import SNIPPET_TERMS, _classify_link
from wireless_taxonomy.analyze.text_match import (
    author_names as _author_names,
    author_overlap_score as _author_overlap_score,
    candidate_key as _candidate_key,
    last_name as _last_name,
    normalize_authors as _normalize_authors,
    normalize_doi as _normalize_doi,
    normalize_person_name as _normalize_person_name,
    normalize_title as _normalize_title,
    resolver_key as _resolver_key,
    same_doi as _same_doi,
    significant_title_tokens as _significant_title_tokens,
    text_matches_title as _text_matches_title,
    title_matches as _title_matches,
    unique as _unique,
)
from wireless_taxonomy.ingest.clean_page import fetch_clean_page
from wireless_taxonomy.models import PaperTextArtifact, PaperTextEnrichment, PaperTextLink, PaperTextSnippet


class _SemanticScholarRateLimiter:
    def __init__(self, interval_fn) -> None:
        self._interval_fn = interval_fn
        self._lock = threading.Lock()
        self._last_request_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait_seconds = self._interval_fn() - (now - self._last_request_at)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            self._last_request_at = time.monotonic()


_SEMANTIC_SCHOLAR_RATE_LIMITER = _SemanticScholarRateLimiter(lambda: _semantic_scholar_min_interval_seconds())
_OPENREVIEW_RATE_LIMITER = _SemanticScholarRateLimiter(lambda: _openreview_min_interval_seconds())


class FullTextDiscoverer:
    provider_name = "full_text_discovery_v0"
    max_pdf_fetch_attempts = 6

    def __init__(self, allow_remote: bool = False, candidate_cache: Any | None = None):
        self.allow_remote = allow_remote
        self.candidate_cache = candidate_cache

    def discover(self, paper: Mapping[str, Any]) -> PaperTextEnrichment:
        paper_id = int(paper["id"])
        paper_url = _optional(_get(paper, "paper_url"))
        pdf_url = _optional(_get(paper, "pdf_url"))
        doi = _optional(_get(paper, "doi"))
        artifacts: list[PaperTextArtifact] = []
        links: list[PaperTextLink] = []

        page_links: list[tuple[str | None, str]] = []
        title = _optional(_get(paper, "title"))
        authors = _optional(_get(paper, "authors"))
        metadata_links = self._metadata_candidates(doi, title, authors)
        doi_links = _doi_candidates(doi, paper_url)
        if paper_url:
            page_artifact, page_links = self._fetch_page(paper_id, paper_url)
            artifacts.append(page_artifact)
            links.extend(_links_from_candidates(paper_id, [("paper_url", paper_url), *page_links]))
        links.extend(_links_from_candidates(paper_id, [*metadata_links, *doi_links]))

        if pdf_url:
            links.append(PaperTextLink(paper_id, pdf_url, "pdf_url", "pdf", 0.95))

        pdf_candidates = _unique_candidates(
            [(label, url) for label, url in page_links if _classify_link(url, label)[0] == "pdf"]
            + [(label, url) for label, url in metadata_links if _classify_link(url, label)[0] == "pdf"]
            + [(label, url) for label, url in doi_links if _classify_link(url, label)[0] == "pdf"]
            + ([("pdf_url", pdf_url)] if pdf_url else [])
        )
        for label, candidate in _prioritize_pdf_candidates(pdf_candidates)[: self.max_pdf_fetch_attempts]:
            artifacts.append(self._fetch_pdf_text(paper_id, candidate, expected_title=title, source_label=label))

        snippets = _snippets_from_artifacts(paper_id, artifacts)
        return PaperTextEnrichment(paper_id, artifacts, _dedupe_links(links), snippets)

    def _fetch_page(self, paper_id: int, source_url: str) -> tuple[PaperTextArtifact, list[tuple[str | None, str]]]:
        if _is_remote_url(source_url) and not self.allow_remote:
            return _artifact(
                paper_id,
                "paper_landing_html",
                source_url,
                "remote_skipped",
                f"Remote full-text landing page skipped: {source_url}",
                None,
            ), []
        try:
            page = fetch_clean_page(source_url)
        except Exception as exc:
            return _artifact(paper_id, "paper_landing_html", source_url, "error", "", str(exc)), []
        return _artifact(paper_id, "paper_landing_html", page.source_url, "fetched", page.text, None), page.links

    def _fetch_pdf_text(
        self,
        paper_id: int,
        source_url: str,
        expected_title: str | None = None,
        source_label: str | None = None,
    ) -> PaperTextArtifact:
        if _is_remote_url(source_url) and not self.allow_remote:
            return _artifact(
                paper_id,
                "pdf_text",
                source_url,
                "remote_skipped",
                f"Remote PDF skipped: {source_url}",
                None,
            )
        try:
            data = _fetch_bytes(source_url)
            if _is_remote_url(source_url) and _looks_like_html_response(data):
                return _artifact(paper_id, "pdf_text", source_url, "error", "", "PDF URL returned HTML instead of PDF bytes")
            text = extract_pdf_text(data)
        except Exception as exc:
            return _artifact(paper_id, "pdf_text", source_url, "error", "", str(exc))
        if text.strip() and _requires_title_validation(source_label, source_url) and not _text_matches_title(expected_title, text):
            return _artifact(
                paper_id,
                "pdf_text",
                source_url,
                "rejected",
                "",
                "Fetched PDF text did not match the expected paper title closely enough",
            )
        status = "fetched" if text.strip() else "empty"
        return _artifact(paper_id, "pdf_text", source_url, status, text, None if text.strip() else "No text extracted from PDF")

    def _metadata_candidates(self, doi: str | None, title: str | None, authors: str | None = None) -> list[tuple[str | None, str]]:
        if not self.allow_remote:
            return []
        candidates: list[tuple[str | None, str]] = []
        for label, url in self._cached_candidates("openalex", _resolver_key(doi, title), lambda: _openalex_candidates(doi, title)):
            candidates.append((label, url))
        for label, url in self._cached_candidates("crossref", doi or "", lambda: _crossref_candidates(doi)):
            candidates.append((label, url))
        legacy_semantic_scholar_key = _resolver_key(doi, title)
        semantic_scholar_key = f"{legacy_semantic_scholar_key}|authors:{_normalize_authors(authors)}"
        semantic_scholar_links = self._cached_candidates(
            "semantic_scholar",
            semantic_scholar_key,
            lambda: _semantic_scholar_candidates(doi, title, authors),
        )
        if not semantic_scholar_links and authors and self.candidate_cache is not None:
            semantic_scholar_links = self.candidate_cache.get_candidates("semantic_scholar", legacy_semantic_scholar_key) or []
        for label, url in semantic_scholar_links:
            candidates.append((label, url))
        for label, url in self._cached_candidates("unpaywall", doi or "", lambda: _unpaywall_candidates(doi)):
            candidates.append((label, url))
        for label, url in self._cached_candidates("arxiv", _normalize_title(title or ""), lambda: _arxiv_candidates(title)):
            candidates.append((label, url))
        for label, url in self._cached_candidates("openreview", _normalize_title(title or ""), lambda: _openreview_candidates(title)):
            candidates.append((label, url))
        for label, url in self._cached_candidates("title_web_search", _normalize_title(title or ""), lambda: _web_search_candidates(title)):
            candidates.append((label, url))
        return _dedupe_candidates(candidates)

    def _cached_candidates(
        self,
        provider: str,
        cache_key: str,
        resolver,
    ) -> list[tuple[str | None, str]]:
        if not cache_key:
            return resolver()
        if self.candidate_cache is not None:
            cached = self.candidate_cache.get_candidates(provider, cache_key)
            if cached is not None:
                return cached
        try:
            candidates = resolver()
        except TransientResolverError:
            return []
        if self.candidate_cache is not None:
            self.candidate_cache.set_candidates(provider, cache_key, candidates)
        return candidates


def _fetch_bytes(source_url: str) -> bytes:
    if source_url.startswith("file://"):
        return Path(source_url.removeprefix("file://")).read_bytes()
    path = Path(source_url)
    if path.exists():
        return path.read_bytes()
    headers = {"User-Agent": "wireless-taxonomy/0.1"}
    if _is_openreview_url(source_url):
        return _fetch_openreview_bytes(source_url, headers)
    return _fetch_bytes_once(source_url, headers)


def _fetch_openreview_bytes(source_url: str, headers: dict[str, str]) -> bytes:
    attempts = _bounded_int_env("WIRELESS_TAXONOMY_OPENREVIEW_RETRIES", default=2, minimum=1, maximum=5)
    for attempt in range(attempts):
        _OPENREVIEW_RATE_LIMITER.wait()
        try:
            return _fetch_bytes_once(source_url, headers)
        except HTTPError as exc:
            if exc.code != 429 or attempt == attempts - 1:
                raise
            retry_after = _retry_after_seconds(exc.headers.get("Retry-After"))
            time.sleep(max(retry_after, _openreview_min_interval_seconds() * 2))
    return b""


def _fetch_bytes_once(source_url: str, headers: dict[str, str]) -> bytes:
    request = Request(source_url, headers=headers)
    with urlopen(request, timeout=30) as response:
        return response.read()


SEMANTIC_SCHOLAR_FIELDS = resolvers.SEMANTIC_SCHOLAR_FIELDS
TransientResolverError = resolvers.TransientResolverError


def _resolver_deps() -> resolvers.ResolverDeps:
    return resolvers.ResolverDeps(
        fetch_json=_fetch_json,
        fetch_bytes=_fetch_bytes,
        fetch_clean_page=fetch_clean_page,
        is_transient_resolver_error=_is_transient_resolver_error,
    )


def _openalex_candidates(doi: str | None, title: str | None = None) -> list[tuple[str | None, str]]:
    return resolvers.openalex_candidates(_resolver_deps(), doi, title)


def _openalex_payload_candidates(payload: dict[str, Any]) -> list[tuple[str | None, str]]:
    return resolvers.openalex_payload_candidates(payload)


def _crossref_candidates(doi: str | None) -> list[tuple[str | None, str]]:
    return resolvers.crossref_candidates(_resolver_deps(), doi)


def _semantic_scholar_candidates(doi: str | None, title: str | None, authors: str | None = None) -> list[tuple[str | None, str]]:
    return resolvers.semantic_scholar_candidates(_resolver_deps(), doi, title, authors)


def _semantic_scholar_payload_candidates(payload: dict[str, Any]) -> list[tuple[str | None, str]]:
    return resolvers.semantic_scholar_payload_candidates(payload)


def _semantic_scholar_match_score(payload: dict[str, Any], doi: str | None, title: str | None, authors: str | None) -> float:
    return resolvers.semantic_scholar_match_score(payload, doi, title, authors)


def _dedupe_semantic_scholar_payloads(payloads) -> list[dict[str, Any]]:
    return resolvers.dedupe_semantic_scholar_payloads(payloads)


def _semantic_scholar_author_names(payload: dict[str, Any]) -> list[str]:
    return resolvers.semantic_scholar_author_names(payload)


def _unpaywall_candidates(doi: str | None) -> list[tuple[str | None, str]]:
    return resolvers.unpaywall_candidates(_resolver_deps(), doi)


def _unpaywall_location_candidates(label: str, location: dict[str, Any]) -> list[tuple[str | None, str]]:
    return resolvers.unpaywall_location_candidates(label, location)


def _arxiv_candidates(title: str | None) -> list[tuple[str | None, str]]:
    return resolvers.arxiv_candidates(_resolver_deps(), title)


def _arxiv_title_queries(title: str) -> list[str]:
    return resolvers.arxiv_title_queries(title)


def _arxiv_candidates_from_atom(data: str, title: str) -> list[tuple[str | None, str]]:
    return resolvers.arxiv_candidates_from_atom(data, title)


def _openreview_candidates(title: str | None) -> list[tuple[str | None, str]]:
    return resolvers.openreview_candidates(_resolver_deps(), title)


def _openreview_content_value(note: dict[str, Any], key: str) -> str | None:
    return resolvers.openreview_content_value(note, key)


def _web_search_candidates(title: str | None) -> list[tuple[str | None, str]]:
    return resolvers.web_search_candidates(_resolver_deps(), title)


def _web_search_queries(title: str) -> list[str]:
    return resolvers.web_search_queries(title)


def _duckduckgo_candidates(query: str, title: str) -> list[tuple[str | None, str]]:
    return resolvers.duckduckgo_candidates(_resolver_deps(), query, title)


def _duckduckgo_result_urls(html: str) -> list[str]:
    return resolvers.duckduckgo_result_urls(html)


_DuckDuckGoResultParser = resolvers.DuckDuckGoResultParser


def _web_result_candidates(url: str, title: str) -> list[tuple[str | None, str]]:
    return resolvers.web_result_candidates(_resolver_deps(), url, title)


def _looks_like_pdf_url(url: str) -> bool:
    return resolvers.looks_like_pdf_url(url)


def _is_obvious_non_paper_url(url: str) -> bool:
    return resolvers.is_obvious_non_paper_url(url)


def _looks_like_scholarly_landing(url: str) -> bool:
    return resolvers.looks_like_scholarly_landing(url)


def _location_candidates(label: str, location: dict[str, Any]) -> list[tuple[str | None, str]]:
    return resolvers.location_candidates(label, location)


def _looks_like_html_response(data: bytes) -> bool:
    head = data[:500].lstrip().lower()
    return head.startswith((b"<!doctype html", b"<html", b"<head", b"<body")) or b"<html" in head[:200]


def _fetch_json(url: str) -> dict[str, Any]:
    headers = {"User-Agent": "wireless-taxonomy/0.1"}
    if "api.semanticscholar.org" in url:
        api_key = _optional(os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY"))
        if api_key:
            headers["x-api-key"] = api_key
        return _fetch_semantic_scholar_json(url, headers)
    if _is_openreview_url(url):
        return _fetch_openreview_json(url, headers)
    return _fetch_json_once(url, headers)


def _fetch_semantic_scholar_json(url: str, headers: dict[str, str]) -> dict[str, Any]:
    attempts = _bounded_int_env("WIRELESS_TAXONOMY_SEMANTIC_SCHOLAR_RETRIES", default=2, minimum=1, maximum=5)
    for attempt in range(attempts):
        _SEMANTIC_SCHOLAR_RATE_LIMITER.wait()
        try:
            return _fetch_json_once(url, headers)
        except HTTPError as exc:
            if exc.code != 429 or attempt == attempts - 1:
                raise
            retry_after = _retry_after_seconds(exc.headers.get("Retry-After"))
            time.sleep(max(retry_after, _semantic_scholar_min_interval_seconds() * 2))
    return {}


def _fetch_json_once(url: str, headers: dict[str, str]) -> dict[str, Any]:
    request = Request(url, headers=headers)
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def _fetch_openreview_json(url: str, headers: dict[str, str]) -> dict[str, Any]:
    attempts = _bounded_int_env("WIRELESS_TAXONOMY_OPENREVIEW_RETRIES", default=2, minimum=1, maximum=5)
    for attempt in range(attempts):
        _OPENREVIEW_RATE_LIMITER.wait()
        try:
            return _fetch_json_once(url, headers)
        except HTTPError as exc:
            if exc.code != 429 or attempt == attempts - 1:
                raise
            retry_after = _retry_after_seconds(exc.headers.get("Retry-After"))
            time.sleep(max(retry_after, _openreview_min_interval_seconds() * 2))
    return {}


def _semantic_scholar_min_interval_seconds() -> float:
    raw_value = os.getenv("WIRELESS_TAXONOMY_SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS")
    if raw_value:
        try:
            return max(1.05, float(raw_value))
        except ValueError:
            pass
    return 1.10


def _openreview_min_interval_seconds() -> float:
    raw_value = os.getenv("WIRELESS_TAXONOMY_OPENREVIEW_MIN_INTERVAL_SECONDS")
    if raw_value:
        try:
            return max(1.00, float(raw_value))
        except ValueError:
            pass
    return 1.25


def _is_openreview_url(url: str) -> bool:
    return urlparse(url).netloc.lower().endswith("openreview.net")


def _retry_after_seconds(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        return _semantic_scholar_min_interval_seconds() * 2


def _bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw_value = os.getenv(name)
    if raw_value:
        try:
            return min(max(int(raw_value), minimum), maximum)
        except ValueError:
            pass
    return default


def _is_transient_resolver_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
    return isinstance(exc, URLError)


def _doi_candidates(doi: str | None, paper_url: str | None) -> list[tuple[str | None, str]]:
    if not doi:
        return []
    candidates: list[tuple[str | None, str]] = [("doi_landing", f"https://doi.org/{doi}")]
    if doi.startswith("10.1145/") or (paper_url and "dl.acm.org/doi/" in paper_url):
        candidates.append(("acm_pdf_candidate", f"https://dl.acm.org/doi/pdf/{doi}"))
    return candidates


def _prioritize_pdf_candidates(candidates: list[tuple[str | None, str]]) -> list[tuple[str | None, str]]:
    def score(candidate: tuple[str | None, str]) -> int:
        label, url = candidate
        lower = url.lower()
        if "arxiv.org/pdf" in lower:
            return 0
        if "semanticscholar.org" in lower or "openalex" in lower:
            return 1
        if "unpaywall" in lower:
            return 2
        if label and label.startswith("web_search"):
            return 3
        if lower.endswith(".pdf"):
            return 4
        if "dl.acm.org/doi/pdf" in lower:
            return 8
        return 5

    return sorted(candidates, key=score)


def _links_from_candidates(paper_id: int, candidates: list[tuple[str | None, str]]) -> list[PaperTextLink]:
    links: list[PaperTextLink] = []
    for label, url in candidates:
        link_type, confidence = _classify_link(url, label)
        links.append(PaperTextLink(paper_id, url, label, link_type, confidence))
    return _dedupe_links(links)


def _dedupe_links(links: list[PaperTextLink]) -> list[PaperTextLink]:
    deduped: list[PaperTextLink] = []
    seen: set[str] = set()
    for link in links:
        if link.url in seen:
            continue
        seen.add(link.url)
        deduped.append(link)
    return deduped


def _dedupe_candidates(candidates: list[tuple[str | None, str]]) -> list[tuple[str | None, str]]:
    result: list[tuple[str | None, str]] = []
    seen: set[str] = set()
    for label, url in candidates:
        key = _candidate_key(url)
        if key in seen:
            continue
        seen.add(key)
        result.append((label, url))
    return result


def _unique_candidates(candidates: list[tuple[str | None, str]]) -> list[tuple[str | None, str]]:
    return _dedupe_candidates(candidates)


def _requires_title_validation(label: str | None, url: str) -> bool:
    if label and label.startswith("web_search"):
        return True
    lower = url.lower()
    return "duckduckgo" in lower


def _semantic_scholar_author_names(payload: dict[str, Any]) -> list[str]:
    raw_authors = payload.get("authors") if isinstance(payload.get("authors"), list) else []
    names: list[str] = []
    for author in raw_authors:
        if isinstance(author, dict):
            normalized = _normalize_person_name(_optional(author.get("name")) or "")
            if normalized:
                names.append(normalized)
    return _unique(names)


def _snippets_from_artifacts(paper_id: int, artifacts: list[PaperTextArtifact]) -> list[PaperTextSnippet]:
    snippets: list[PaperTextSnippet] = []
    seen: set[str] = set()
    for artifact in artifacts:
        if artifact.fetch_status != "fetched" or not artifact.content_text:
            continue
        for match in SNIPPET_TERMS.finditer(artifact.content_text):
            start = max(0, match.start() - 180)
            end = min(len(artifact.content_text), match.end() + 220)
            text = " ".join(artifact.content_text[start:end].split())
            if text.lower() in seen:
                continue
            seen.add(text.lower())
            snippets.append(
                PaperTextSnippet(
                    paper_id=paper_id,
                    snippet_type="full_text_context",
                    snippet_text=text,
                    source_url=artifact.source_url,
                    start_char=start,
                    end_char=end,
                    confidence=0.85,
                )
            )
            if len(snippets) >= 12:
                return snippets
    return snippets


def _artifact(
    paper_id: int,
    source_type: str,
    source_url: str | None,
    fetch_status: str,
    content_text: str,
    error_message: str | None,
) -> PaperTextArtifact:
    from wireless_taxonomy.analyze.paper_text import _artifact as make_artifact

    return make_artifact(paper_id, source_type, source_url, fetch_status, content_text, error_message)


def _optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _get(paper: Mapping[str, Any], key: str) -> Any:
    return paper[key]


def _is_remote_url(url: str) -> bool:
    return urlparse(url).scheme in {"http", "https"}
