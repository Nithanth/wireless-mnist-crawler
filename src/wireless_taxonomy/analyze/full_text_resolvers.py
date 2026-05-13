from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse

from wireless_taxonomy.analyze.paper_text import _classify_link
from wireless_taxonomy.analyze.text_match import (
    author_names,
    author_overlap_score,
    candidate_key,
    normalize_doi,
    normalize_person_name,
    normalize_title,
    same_doi,
    significant_title_tokens,
    text_matches_title,
    title_matches,
    unique,
)
from wireless_taxonomy.ingest.clean_page import CleanPage


class TransientResolverError(Exception):
    pass


@dataclass(frozen=True)
class ResolverDeps:
    fetch_json: Callable[[str], dict[str, Any]]
    fetch_bytes: Callable[[str], bytes]
    fetch_clean_page: Callable[[str], CleanPage]
    is_transient_resolver_error: Callable[[Exception], bool]


SEMANTIC_SCHOLAR_FIELDS = "paperId,url,title,year,authors,openAccessPdf,externalIds,isOpenAccess"


def openalex_candidates(deps: ResolverDeps, doi: str | None, title: str | None = None) -> list[tuple[str | None, str]]:
    payload: dict[str, Any] = {}
    if doi:
        url = f"https://api.openalex.org/works/https://doi.org/{quote(doi, safe='')}"
        try:
            payload = deps.fetch_json(url)
        except Exception:
            payload = {}
    if not payload and title:
        url = "https://api.openalex.org/works?" + urlencode({"search": title, "per-page": "1"})
        try:
            search_payload = deps.fetch_json(url)
        except Exception:
            search_payload = {}
        results = search_payload.get("results") if isinstance(search_payload.get("results"), list) else []
        payload = results[0] if results and isinstance(results[0], dict) else {}
        if payload and not title_matches(title, optional(payload.get("title"))):
            payload = {}
    if not payload:
        return []
    return openalex_payload_candidates(payload)


def openalex_payload_candidates(payload: dict[str, Any]) -> list[tuple[str | None, str]]:
    candidates: list[tuple[str | None, str]] = []
    open_access = payload.get("open_access") if isinstance(payload.get("open_access"), dict) else {}
    oa_url = optional(open_access.get("oa_url"))
    if oa_url:
        candidates.append(("openalex_oa_url", oa_url))
    for location_key in ["primary_location", "best_oa_location"]:
        location = payload.get(location_key) if isinstance(payload.get(location_key), dict) else {}
        candidates.extend(location_candidates(f"openalex_{location_key}", location))
    for location in payload.get("locations") or []:
        if isinstance(location, dict):
            candidates.extend(location_candidates("openalex_location", location))
    return dedupe_candidates(candidates)


def crossref_candidates(deps: ResolverDeps, doi: str | None) -> list[tuple[str | None, str]]:
    if not doi:
        return []
    url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
    try:
        payload = deps.fetch_json(url)
    except Exception:
        return []
    message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
    candidates: list[tuple[str | None, str]] = []
    for link in message.get("link") or []:
        if not isinstance(link, dict):
            continue
        link_url = optional(link.get("URL"))
        content_type = str(link.get("content-type") or "").lower()
        if link_url and ("pdf" in content_type or link_url.lower().endswith(".pdf")):
            candidates.append(("crossref_pdf", link_url))
    return dedupe_candidates(candidates)


def semantic_scholar_candidates(
    deps: ResolverDeps,
    doi: str | None,
    title: str | None,
    authors: str | None = None,
) -> list[tuple[str | None, str]]:
    payloads: list[dict[str, Any]] = []
    if doi:
        url = (
            "https://api.semanticscholar.org/graph/v1/paper/DOI:"
            f"{quote(doi, safe='')}?fields={quote(SEMANTIC_SCHOLAR_FIELDS, safe=',')}"
        )
        try:
            payload = deps.fetch_json(url)
        except Exception as exc:
            if deps.is_transient_resolver_error(exc):
                raise TransientResolverError(str(exc)) from exc
            payload = {}
        if payload:
            payloads.append(payload)
    if title:
        search_url = (
            "https://api.semanticscholar.org/graph/v1/paper/search?"
            + urlencode({"query": title, "limit": "10", "fields": SEMANTIC_SCHOLAR_FIELDS})
        )
        try:
            search_payload = deps.fetch_json(search_url)
        except Exception as exc:
            if deps.is_transient_resolver_error(exc):
                raise TransientResolverError(str(exc)) from exc
            search_payload = {}
        data = search_payload.get("data") if isinstance(search_payload.get("data"), list) else []
        payloads.extend(item for item in data if isinstance(item, dict))

    accepted = dedupe_semantic_scholar_payloads(
        payload
        for payload in payloads
        if semantic_scholar_match_score(payload, doi, title, authors) >= 0.45
    )
    candidates: list[tuple[str | None, str]] = []
    for payload in accepted:
        candidates.extend(semantic_scholar_payload_candidates(payload))
    return dedupe_candidates(candidates)


def semantic_scholar_payload_candidates(payload: dict[str, Any]) -> list[tuple[str | None, str]]:
    candidates: list[tuple[str | None, str]] = []
    open_access = payload.get("openAccessPdf") if isinstance(payload.get("openAccessPdf"), dict) else {}
    pdf_url = optional(open_access.get("url"))
    paper_url = optional(payload.get("url"))
    if pdf_url:
        candidates.append(("semantic_scholar_open_access_pdf", pdf_url))
    if paper_url:
        candidates.append(("semantic_scholar_landing", paper_url))
    external_ids = payload.get("externalIds") if isinstance(payload.get("externalIds"), dict) else {}
    arxiv_id = optional(external_ids.get("ArXiv"))
    if arxiv_id:
        candidates.append(("semantic_scholar_arxiv_pdf", f"https://arxiv.org/pdf/{arxiv_id}"))
        candidates.append(("semantic_scholar_arxiv_landing", f"https://arxiv.org/abs/{arxiv_id}"))
    openreview_id = optional(external_ids.get("OpenReview"))
    if openreview_id:
        candidates.append(("semantic_scholar_openreview_landing", f"https://openreview.net/forum?id={openreview_id}"))
        candidates.append(("semantic_scholar_openreview_pdf", f"https://openreview.net/pdf?id={openreview_id}"))
    return candidates


def semantic_scholar_match_score(payload: dict[str, Any], doi: str | None, title: str | None, authors: str | None) -> float:
    score = 0.0
    payload_title = optional(payload.get("title"))
    payload_doi = optional((payload.get("externalIds") or {}).get("DOI")) if isinstance(payload.get("externalIds"), dict) else None

    if doi and payload_doi and same_doi(doi, payload_doi):
        score += 0.60
    if title and payload_title:
        expected_norm = normalize_title(title)
        actual_norm = normalize_title(payload_title)
        if expected_norm and expected_norm == actual_norm:
            score += 0.45
        elif title_matches(title, payload_title):
            score += 0.35

    author_score = author_overlap_score(author_names(authors), semantic_scholar_author_names(payload))
    score += min(author_score, 0.20)

    return min(score, 1.0)


def dedupe_semantic_scholar_payloads(payloads: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for payload in payloads:
        paper_id = optional(payload.get("paperId"))
        external_ids = payload.get("externalIds") if isinstance(payload.get("externalIds"), dict) else {}
        arxiv_id = optional(external_ids.get("ArXiv"))
        doi = optional(external_ids.get("DOI"))
        url = optional(payload.get("url"))
        key = paper_id or (f"arxiv:{arxiv_id.lower()}" if arxiv_id else None) or (f"doi:{normalize_doi(doi)}" if doi else None) or url
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(payload)
    return deduped


def semantic_scholar_author_names(payload: dict[str, Any]) -> list[str]:
    raw_authors = payload.get("authors") if isinstance(payload.get("authors"), list) else []
    names: list[str] = []
    for author in raw_authors:
        if isinstance(author, dict):
            normalized = normalize_person_name(optional(author.get("name")) or "")
            if normalized:
                names.append(normalized)
    return unique(names)


def unpaywall_candidates(deps: ResolverDeps, doi: str | None) -> list[tuple[str | None, str]]:
    if not doi:
        return []
    email = optional(os.getenv("WIRELESS_TAXONOMY_UNPAYWALL_EMAIL") or os.getenv("UNPAYWALL_EMAIL"))
    if not email:
        return []
    url = f"https://api.unpaywall.org/v2/{quote(doi, safe='')}?" + urlencode({"email": email})
    try:
        payload = deps.fetch_json(url)
    except Exception:
        return []
    candidates: list[tuple[str | None, str]] = []
    best = payload.get("best_oa_location") if isinstance(payload.get("best_oa_location"), dict) else {}
    candidates.extend(unpaywall_location_candidates("unpaywall_best_oa", best))
    for location in payload.get("oa_locations") or []:
        if isinstance(location, dict):
            candidates.extend(unpaywall_location_candidates("unpaywall_oa_location", location))
    return dedupe_candidates(candidates)


def unpaywall_location_candidates(label: str, location: dict[str, Any]) -> list[tuple[str | None, str]]:
    candidates: list[tuple[str | None, str]] = []
    pdf_url = optional(location.get("url_for_pdf"))
    landing_url = optional(location.get("url_for_landing_page"))
    if pdf_url:
        candidates.append((f"{label}_pdf", pdf_url))
    if landing_url:
        candidates.append((f"{label}_landing", landing_url))
    return candidates


def arxiv_candidates(deps: ResolverDeps, title: str | None) -> list[tuple[str | None, str]]:
    if not title:
        return []
    candidates: list[tuple[str | None, str]] = []
    for query in arxiv_title_queries(title):
        url = "https://export.arxiv.org/api/query?" + urlencode({"search_query": query, "start": "0", "max_results": "10"})
        try:
            data = deps.fetch_bytes(url).decode("utf-8", errors="replace")
        except Exception:
            continue
        candidates.extend(arxiv_candidates_from_atom(data, title))
        if candidates:
            break
    return dedupe_candidates(candidates)


def arxiv_title_queries(title: str) -> list[str]:
    queries = [f'ti:"{title}"']
    tokens = significant_title_tokens(title)
    if tokens:
        queries.append(" AND ".join(f"ti:{token}" for token in tokens[:8]))
        queries.append(" AND ".join(f"all:{token}" for token in tokens[:8]))
    return unique(queries)


def arxiv_candidates_from_atom(data: str, title: str) -> list[tuple[str | None, str]]:
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return []
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    candidates: list[tuple[str | None, str]] = []
    for entry in root.findall("atom:entry", namespace):
        entry_title = optional(" ".join((entry.findtext("atom:title", default="", namespaces=namespace) or "").split()))
        if not title_matches(title, entry_title):
            continue
        landing_url = optional(entry.findtext("atom:id", default="", namespaces=namespace))
        pdf_url = None
        for link in entry.findall("atom:link", namespace):
            attrs = link.attrib
            if attrs.get("title") == "pdf" or attrs.get("type") == "application/pdf":
                pdf_url = optional(attrs.get("href"))
                break
        if landing_url:
            candidates.append(("arxiv_landing", landing_url))
            arxiv_id = landing_url.rstrip("/").rsplit("/", 1)[-1]
            if arxiv_id and not pdf_url:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        if pdf_url:
            candidates.append(("arxiv_pdf", pdf_url))
    return candidates


def openreview_candidates(deps: ResolverDeps, title: str | None) -> list[tuple[str | None, str]]:
    if not title:
        return []
    url = "https://api2.openreview.net/notes/search?" + urlencode(
        {
            "term": title,
            "content": "title",
            "type": "exact",
            "limit": "5",
        }
    )
    try:
        payload = deps.fetch_json(url)
    except Exception:
        return []
    notes = payload.get("notes") if isinstance(payload.get("notes"), list) else []
    candidates: list[tuple[str | None, str]] = []
    for note in notes:
        if not isinstance(note, dict):
            continue
        note_title = openreview_content_value(note, "title")
        if not title_matches(title, note_title):
            continue
        note_id = optional(note.get("forum") or note.get("id"))
        if not note_id:
            continue
        candidates.append(("openreview_landing", f"https://openreview.net/forum?id={note_id}"))
        candidates.append(("openreview_pdf", f"https://openreview.net/pdf?id={note_id}"))
    return dedupe_candidates(candidates)


def openreview_content_value(note: dict[str, Any], key: str) -> str | None:
    content = note.get("content") if isinstance(note.get("content"), dict) else {}
    value = content.get(key)
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value)
    return optional(value)


def web_search_candidates(deps: ResolverDeps, title: str | None) -> list[tuple[str | None, str]]:
    if not title:
        return []
    candidates: list[tuple[str | None, str]] = []
    for query in web_search_queries(title):
        candidates.extend(duckduckgo_candidates(deps, query, title))
        if len(candidates) >= 8:
            break
    return dedupe_candidates(candidates)


def web_search_queries(title: str) -> list[str]:
    return [
        f'"{title}" filetype:pdf',
        f'"{title}" arxiv',
        f'"{title}" pdf',
        f'"{title}" author manuscript',
    ]


def duckduckgo_candidates(deps: ResolverDeps, query: str, title: str) -> list[tuple[str | None, str]]:
    url = "https://duckduckgo.com/html/?" + urlencode({"q": query})
    try:
        html = deps.fetch_bytes(url).decode("utf-8", errors="replace")
    except Exception:
        return []
    candidates: list[tuple[str | None, str]] = []
    for result_url in duckduckgo_result_urls(html)[:8]:
        if is_obvious_non_paper_url(result_url):
            continue
        candidates.extend(web_result_candidates(deps, result_url, title))
        if len(candidates) >= 8:
            break
    return dedupe_candidates(candidates)


def duckduckgo_result_urls(html: str) -> list[str]:
    parser = DuckDuckGoResultParser()
    parser.feed(html)
    urls: list[str] = []
    for href in parser.hrefs:
        parsed = urlparse(href)
        if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
            params = parse_qs(parsed.query)
            target = params.get("uddg", [None])[0]
            if target:
                href = unquote(target)
        if href.startswith("//"):
            href = "https:" + href
        if not href.startswith(("http://", "https://")):
            continue
        if any(blocked in href.lower() for blocked in ("duckduckgo.com", "google.com/search", "bing.com/search")):
            continue
        urls.append(href)
    return unique(urls)


class DuckDuckGoResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        class_name = attrs_dict.get("class") or ""
        if href and ("result__a" in class_name or "/l/?" in href):
            self.hrefs.append(href)


def web_result_candidates(deps: ResolverDeps, url: str, title: str) -> list[tuple[str | None, str]]:
    parsed = urlparse(url)
    lower = url.lower()
    if "arxiv.org/abs/" in lower:
        arxiv_id = parsed.path.rstrip("/").rsplit("/", 1)[-1]
        return [("web_search_arxiv_landing", url), ("web_search_arxiv_pdf", f"https://arxiv.org/pdf/{arxiv_id}")]
    if "arxiv.org/pdf/" in lower:
        arxiv_id = parsed.path.rstrip("/").rsplit("/", 1)[-1]
        return [("web_search_arxiv_pdf", url), ("web_search_arxiv_landing", f"https://arxiv.org/abs/{arxiv_id}")]
    if looks_like_pdf_url(url):
        return [("web_search_result_pdf", url)]
    if not looks_like_scholarly_landing(url):
        return [("web_search_result_landing", url)]
    try:
        page = deps.fetch_clean_page(url)
    except Exception:
        return [("web_search_result_landing", url)]
    candidates: list[tuple[str | None, str]] = [("web_search_result_landing", page.source_url)]
    if not text_matches_title(title, page.text):
        return candidates
    for link_text, link_url in page.links[:40]:
        if _classify_link(link_url, link_text)[0] == "pdf" or looks_like_pdf_url(link_url):
            candidates.append(("web_search_landing_pdf", link_url))
        if "arxiv.org/abs/" in link_url.lower():
            arxiv_id = urlparse(link_url).path.rstrip("/").rsplit("/", 1)[-1]
            candidates.append(("web_search_arxiv_landing", link_url))
            candidates.append(("web_search_arxiv_pdf", f"https://arxiv.org/pdf/{arxiv_id}"))
    return dedupe_candidates(candidates)


def looks_like_pdf_url(url: str) -> bool:
    parsed = urlparse(url)
    lower = url.lower()
    return parsed.path.lower().endswith(".pdf") or "/pdf/" in lower or "download" in parsed.path.lower()


def is_obvious_non_paper_url(url: str) -> bool:
    lower = unquote(url.lower())
    blocked_terms = ("resume", "curriculum-vitae", "curriculum_vitae", "publication-list", "publications.pdf")
    if any(term in lower for term in blocked_terms):
        return True
    return bool(re.search(r"(^|[/_\-\s])cv([._\-\s/]|$)", lower))


def looks_like_scholarly_landing(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    path = urlparse(url).path.lower()
    scholarly_hosts = (
        "arxiv.org",
        "github.io",
        "edu",
        "acm.org",
        "semanticscholar.org",
        "openreview.net",
        "usenix.org",
        "vtechworks.lib.vt.edu",
    )
    if any(token in host for token in scholarly_hosts):
        return True
    return any(token in path for token in ("/paper", "/publication", "/pubs", "/research", "/handle/", "/bitstream/"))


def location_candidates(label: str, location: dict[str, Any]) -> list[tuple[str | None, str]]:
    candidates: list[tuple[str | None, str]] = []
    pdf_url = optional(location.get("pdf_url"))
    landing_url = optional(location.get("landing_page_url"))
    if pdf_url:
        candidates.append((f"{label}_pdf", pdf_url))
    if landing_url:
        candidates.append((f"{label}_landing", landing_url))
    return candidates


def dedupe_candidates(candidates: list[tuple[str | None, str]]) -> list[tuple[str | None, str]]:
    result: list[tuple[str | None, str]] = []
    seen: set[str] = set()
    for label, url in candidates:
        key = candidate_key(url)
        if key in seen:
            continue
        seen.add(key)
        result.append((label, url))
    return result


def optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
