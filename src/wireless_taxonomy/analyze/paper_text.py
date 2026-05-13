from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from wireless_taxonomy.ingest.clean_page import fetch_clean_page
from wireless_taxonomy.models import PaperTextArtifact, PaperTextEnrichment, PaperTextLink, PaperTextSnippet


SNIPPET_TERMS = re.compile(
    r"(?i)\b(dataset|data set|data|artifact|benchmark|trace|traces|measurement|measurements|corpus|repository|available|download)\b"
)
URL_PATTERN = re.compile(r"https?://[^\s)\]>\"']+")


class PaperTextFetcher:
    provider_name = "paper_text_fetcher_v1"

    def __init__(self, allow_remote: bool = False):
        self.allow_remote = allow_remote

    def enrich(self, paper: Mapping[str, Any]) -> PaperTextEnrichment:
        paper_id = int(_value(paper, "id"))
        title = str(_value(paper, "title") or "")
        abstract = _optional(_value(paper, "abstract"))
        paper_url = _optional(_value(paper, "paper_url"))
        pdf_url = _optional(_value(paper, "pdf_url"))

        artifacts: list[PaperTextArtifact] = []
        page_links: list[tuple[str | None, str]] = []

        if title or abstract:
            artifacts.append(_artifact(paper_id, "abstract", None, "available", self.fetch_text(title, abstract, None), None))

        if paper_url:
            artifact, links = self._fetch_page_artifact(paper_id, "paper_url", paper_url)
            artifacts.append(artifact)
            page_links.extend(links)

        if pdf_url:
            if _looks_like_pdf(pdf_url):
                text = f"PDF available: {pdf_url}"
                artifacts.append(_artifact(paper_id, "pdf_reference", pdf_url, "reference_only", text, None))
            else:
                artifact, links = self._fetch_page_artifact(paper_id, "pdf_url", pdf_url)
                artifacts.append(artifact)
                page_links.extend(links)

        links = self._extract_links(paper_id, paper_url, pdf_url, page_links, artifacts)
        snippets = self._extract_snippets(paper_id, artifacts)
        return PaperTextEnrichment(paper_id, artifacts, links, snippets)

    def fetch_text(self, title: str, abstract: str | None = None, pdf_url: str | None = None) -> str:
        return "\n\n".join(part for part in [title, abstract or "", f"PDF: {pdf_url}" if pdf_url else ""] if part)

    def _fetch_page_artifact(self, paper_id: int, source_type: str, source_url: str) -> tuple[PaperTextArtifact, list[tuple[str | None, str]]]:
        if _is_remote_url(source_url) and not self.allow_remote:
            text = f"Remote paper page available: {source_url}"
            return _artifact(paper_id, source_type, source_url, "remote_skipped", text, None), []
        try:
            page = fetch_clean_page(source_url)
        except Exception as exc:
            return _artifact(paper_id, source_type, source_url, "error", "", str(exc)), []
        return _artifact(paper_id, source_type, page.source_url, "fetched", page.text, None), page.links

    def _extract_links(
        self,
        paper_id: int,
        paper_url: str | None,
        pdf_url: str | None,
        page_links: list[tuple[str | None, str]],
        artifacts: list[PaperTextArtifact],
    ) -> list[PaperTextLink]:
        candidates: list[tuple[str | None, str]] = []
        if paper_url:
            candidates.append(("paper_url", paper_url))
        if pdf_url:
            candidates.append(("pdf_url", pdf_url))
        candidates.extend(page_links)
        for artifact_item in artifacts:
            candidates.extend((None, match.group(0).rstrip(".,;")) for match in URL_PATTERN.finditer(artifact_item.content_text))

        links: list[PaperTextLink] = []
        seen: set[str] = set()
        for label, url in candidates:
            normalized = url.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            link_type, confidence = _classify_link(normalized, label)
            links.append(PaperTextLink(paper_id, normalized, label, link_type, confidence))
        return links

    def _extract_snippets(self, paper_id: int, artifacts: list[PaperTextArtifact]) -> list[PaperTextSnippet]:
        snippets: list[PaperTextSnippet] = []
        seen: set[str] = set()
        for artifact_item in artifacts:
            if not artifact_item.content_text or artifact_item.fetch_status not in {"available", "fetched"}:
                continue
            for match in SNIPPET_TERMS.finditer(artifact_item.content_text):
                start = max(0, match.start() - 180)
                end = min(len(artifact_item.content_text), match.end() + 220)
                snippet = " ".join(artifact_item.content_text[start:end].split())
                key = snippet.lower()
                if key in seen:
                    continue
                seen.add(key)
                snippets.append(
                    PaperTextSnippet(
                        paper_id=paper_id,
                        snippet_type="dataset_context",
                        snippet_text=snippet,
                        source_url=artifact_item.source_url,
                        start_char=start,
                        end_char=end,
                        confidence=0.75 if artifact_item.fetch_status in {"available", "fetched"} else 0.55,
                    )
                )
                if len(snippets) >= 8:
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
    return PaperTextArtifact(
        paper_id=paper_id,
        source_type=source_type,
        source_url=source_url,
        fetch_status=fetch_status,
        content_text=content_text,
        content_sha256=hashlib.sha256(content_text.encode("utf-8")).hexdigest(),
        error_message=error_message,
    )


def _classify_link(url: str, label: str | None) -> tuple[str, float]:
    lower_url = url.lower()
    lower_label = (label or "").lower()
    if lower_url.endswith(".pdf") or "/pdf/" in lower_url or "pdf" in lower_label:
        return "pdf", 0.95
    if any(host in lower_url for host in ["github.com", "gitlab.com"]):
        return "repository", 0.80
    if any(host in lower_url for host in ["zenodo", "figshare", "dataverse", "osf.io", "kaggle.com", "huggingface.co"]):
        return "dataset_or_artifact", 0.85
    if "doi.org" in lower_url or "dl.acm.org/doi" in lower_url or "arxiv.org/abs/" in lower_url:
        return "paper_landing", 0.90
    if any(term in lower_label for term in ["data", "dataset", "artifact", "code"]):
        return "dataset_or_artifact", 0.70
    return "other", 0.50


def _looks_like_pdf(url: str) -> bool:
    return url.lower().split("?", 1)[0].endswith(".pdf")


def _is_remote_url(url: str) -> bool:
    return urlparse(url).scheme in {"http", "https"}


def _optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _value(paper: Mapping[str, Any], key: str) -> Any:
    return paper[key]
