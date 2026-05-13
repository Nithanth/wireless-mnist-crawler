from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PaperInputReadiness:
    paper_id: int
    has_abstract: bool
    has_fetched_text: bool
    has_pdf_link: bool
    has_artifact_link: bool
    snippet_count: int
    usable_text_chars: int
    readiness_level: str
    should_analyze: bool
    limitations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "has_abstract": self.has_abstract,
            "has_fetched_text": self.has_fetched_text,
            "has_pdf_link": self.has_pdf_link,
            "has_artifact_link": self.has_artifact_link,
            "snippet_count": self.snippet_count,
            "usable_text_chars": self.usable_text_chars,
            "readiness_level": self.readiness_level,
            "should_analyze": self.should_analyze,
            "limitations": self.limitations,
        }


class PaperInputReadinessAssessor:
    provider_name = "paper_input_readiness_v0"

    def assess(
        self,
        paper: dict[str, Any],
        artifacts: list[dict[str, Any]],
        links: list[dict[str, Any]],
        snippets: list[dict[str, Any]],
    ) -> PaperInputReadiness:
        paper_id = int(paper["id"])
        has_abstract = any(
            artifact.get("source_type") == "abstract"
            and artifact.get("fetch_status") == "available"
            and str(artifact.get("content_text") or "").strip()
            for artifact in artifacts
        )
        fetched_artifacts = [
            artifact
            for artifact in artifacts
            if artifact.get("fetch_status") == "fetched"
            and artifact.get("source_type") in {"paper_url", "pdf_url", "full_text", "paper_landing_html", "html_full_text", "pdf_text"}
        ]
        has_fetched_text = bool(fetched_artifacts)
        has_pdf_link = any(link.get("link_type") == "pdf" for link in links) or any(
            artifact.get("source_type") in {"pdf_reference", "pdf_url"} for artifact in artifacts
        )
        has_artifact_link = any(link.get("link_type") in {"dataset_or_artifact", "repository"} for link in links)
        usable_text_chars = sum(
            len(str(artifact.get("content_text") or ""))
            for artifact in artifacts
            if artifact.get("fetch_status") in {"available", "fetched"}
        )
        snippet_count = len(snippets)
        limitations: list[str] = []
        if not has_fetched_text:
            limitations.append("No fetched full-text or paper landing-page text.")
        if not has_pdf_link:
            limitations.append("No direct PDF link discovered.")
        if not has_artifact_link:
            limitations.append("No artifact, repository, or dataset link discovered.")
        if snippet_count == 0:
            limitations.append("No dataset/data/artifact snippets were extracted.")
        if not has_abstract and not has_fetched_text:
            limitations.append("No usable abstract or fetched text.")

        if has_fetched_text and (has_pdf_link or has_artifact_link):
            readiness_level = "full_text_plus_links"
        elif has_fetched_text:
            readiness_level = "full_text"
        elif has_abstract and (has_pdf_link or has_artifact_link):
            readiness_level = "abstract_plus_links"
        elif has_abstract:
            readiness_level = "abstract_only"
        else:
            readiness_level = "insufficient"

        should_analyze = readiness_level != "insufficient" and usable_text_chars > 0
        return PaperInputReadiness(
            paper_id=paper_id,
            has_abstract=has_abstract,
            has_fetched_text=has_fetched_text,
            has_pdf_link=has_pdf_link,
            has_artifact_link=has_artifact_link,
            snippet_count=snippet_count,
            usable_text_chars=usable_text_chars,
            readiness_level=readiness_level,
            should_analyze=should_analyze,
            limitations=limitations,
        )
