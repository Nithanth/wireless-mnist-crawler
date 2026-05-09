from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from wireless_taxonomy.config import LlmSettings
from wireless_taxonomy.ingest.clean_page import fetch_clean_page
from wireless_taxonomy.llm import LlmRequest, LlmRouter


@dataclass(frozen=True)
class VerificationIssue:
    field: str
    paper_id: int | None
    paper_title: str | None
    severity: str
    message: str
    suggested_value: str | None
    confidence: float
    evidence: str | None = None
    source_url: str | None = None


@dataclass(frozen=True)
class PaperListVerificationReport:
    paper_count: int
    missing_authors_count: int
    missing_abstract_count: int
    missing_doi_count: int
    duplicate_title_count: int
    low_confidence_count: int
    external_checked_count: int
    external_mismatch_count: int
    llm_correction_count: int
    final_confidence: float
    issues: list[VerificationIssue]
    external_results: list[dict[str, Any]]
    llm_corrections: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_count": self.paper_count,
            "missing_authors_count": self.missing_authors_count,
            "missing_abstract_count": self.missing_abstract_count,
            "missing_doi_count": self.missing_doi_count,
            "duplicate_title_count": self.duplicate_title_count,
            "low_confidence_count": self.low_confidence_count,
            "external_checked_count": self.external_checked_count,
            "external_mismatch_count": self.external_mismatch_count,
            "llm_correction_count": self.llm_correction_count,
            "final_confidence": self.final_confidence,
            "issues": [issue.__dict__ for issue in self.issues],
            "external_results": self.external_results,
            "llm_corrections": self.llm_corrections,
        }


class PaperListVerifier:
    def __init__(self, llm_settings: LlmSettings | None = None):
        self.llm_settings = llm_settings

    def verify(
        self,
        papers: list[dict[str, Any]],
        source_url: str | None,
        run_external: bool = False,
        run_llm: bool = False,
    ) -> PaperListVerificationReport:
        issues = _deterministic_issues(papers)
        external_results: list[dict[str, Any]] = []
        llm_corrections: list[dict[str, Any]] = []
        if run_external:
            external_results, external_issues = self._external_check(papers)
            issues.extend(external_issues)
        if run_llm and source_url and self.llm_settings:
            llm_corrections, llm_issues = self._llm_verify(papers, source_url)
            issues.extend(llm_issues)
        counts = _counts(papers, issues, external_results, llm_corrections)
        final_confidence = _final_confidence(counts, len(issues))
        return PaperListVerificationReport(
            paper_count=counts["paper_count"],
            missing_authors_count=counts["missing_authors_count"],
            missing_abstract_count=counts["missing_abstract_count"],
            missing_doi_count=counts["missing_doi_count"],
            duplicate_title_count=counts["duplicate_title_count"],
            low_confidence_count=counts["low_confidence_count"],
            external_checked_count=counts["external_checked_count"],
            external_mismatch_count=counts["external_mismatch_count"],
            llm_correction_count=counts["llm_correction_count"],
            final_confidence=final_confidence,
            issues=issues,
            external_results=external_results,
            llm_corrections=llm_corrections,
        )

    def _external_check(self, papers: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[VerificationIssue]]:
        results: list[dict[str, Any]] = []
        issues: list[VerificationIssue] = []
        for paper in papers:
            result = crossref_check(paper)
            if result is None:
                continue
            results.append(result)
            if result["status"] == "mismatch":
                issues.append(
                    VerificationIssue(
                        field="External Metadata",
                        paper_id=paper["id"],
                        paper_title=paper["title"],
                        severity="review",
                        message="External metadata title differs from extracted title",
                        suggested_value=result.get("external_title"),
                        confidence=result.get("similarity", 0.0),
                        evidence=json.dumps(result, ensure_ascii=False),
                        source_url=result.get("source_url"),
                    )
                )
        return results, issues

    def _llm_verify(self, papers: list[dict[str, Any]], source_url: str) -> tuple[list[dict[str, Any]], list[VerificationIssue]]:
        page = fetch_clean_page(source_url)
        prompt = _llm_verifier_prompt(page.text, page.links, papers)
        response = LlmRouter(self.llm_settings).complete(
            LlmRequest(task="paper_list_verification", schema_name="PaperListCorrections", prompt=prompt)
        )
        payload = response.parsed if isinstance(response.parsed, dict) else {}
        corrections = payload.get("corrections", []) if isinstance(payload, dict) else []
        if not isinstance(corrections, list):
            corrections = []
        issues = [
            VerificationIssue(
                field=str(correction.get("field") or "Paper List"),
                paper_id=None,
                paper_title=correction.get("paper_title"),
                severity="review",
                message=str(correction.get("issue") or "LLM verifier suggested a correction"),
                suggested_value=correction.get("suggested_value"),
                confidence=float(correction.get("confidence") or 0.80),
                evidence=correction.get("evidence_text"),
                source_url=source_url,
            )
            for correction in corrections
            if isinstance(correction, dict)
        ]
        return corrections, issues


def crossref_check(paper: dict[str, Any]) -> dict[str, Any] | None:
    doi = paper.get("doi")
    if doi:
        url = f"https://api.crossref.org/works/{quote(str(doi), safe='')}"
    else:
        title = paper.get("title")
        if not title:
            return None
        url = f"https://api.crossref.org/works?rows=1&query.title={quote(str(title))}"
    request = Request(url, headers={"User-Agent": "wireless-taxonomy/0.1 (mailto:research@example.invalid)"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    message = payload.get("message", {})
    item = message if doi else (message.get("items") or [{}])[0]
    titles = item.get("title") or []
    external_title = titles[0] if titles else ""
    similarity = _title_similarity(str(paper.get("title") or ""), external_title)
    return {
        "paper_id": paper.get("id"),
        "title": paper.get("title"),
        "doi": doi or item.get("DOI"),
        "external_title": external_title,
        "similarity": similarity,
        "status": "match" if similarity >= 0.92 else "mismatch",
        "source": "crossref",
        "source_url": url,
    }


def _deterministic_issues(papers: list[dict[str, Any]]) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    seen: dict[str, list[dict[str, Any]]] = {}
    for paper in papers:
        normalized = _normalize_title(str(paper.get("title") or ""))
        seen.setdefault(normalized, []).append(paper)
        for field, label in [("authors", "Authors"), ("abstract", "Abstract"), ("doi", "DOI")]:
            if not str(paper.get(field) or "").strip():
                issues.append(
                    VerificationIssue(label, paper.get("id"), paper.get("title"), "review", f"Missing {label}", None, 0.0)
                )
        if float(paper.get("source_confidence") or 0) < 0.90:
            issues.append(
                VerificationIssue(
                    "PaperSeed",
                    paper.get("id"),
                    paper.get("title"),
                    "review",
                    "Extraction confidence below threshold",
                    None,
                    float(paper.get("source_confidence") or 0),
                )
            )
    for same_title in seen.values():
        if same_title[0].get("title") and len(same_title) > 1:
            for paper in same_title:
                issues.append(
                    VerificationIssue("Paper Title", paper.get("id"), paper.get("title"), "review", "Duplicate title", paper.get("title"), 0.50)
                )
    return issues


def _counts(
    papers: list[dict[str, Any]],
    issues: list[VerificationIssue],
    external_results: list[dict[str, Any]],
    llm_corrections: list[dict[str, Any]],
) -> dict[str, int]:
    return {
        "paper_count": len(papers),
        "missing_authors_count": sum(1 for issue in issues if issue.message == "Missing Authors"),
        "missing_abstract_count": sum(1 for issue in issues if issue.message == "Missing Abstract"),
        "missing_doi_count": sum(1 for issue in issues if issue.message == "Missing DOI"),
        "duplicate_title_count": sum(1 for issue in issues if issue.message == "Duplicate title"),
        "low_confidence_count": sum(1 for issue in issues if issue.message == "Extraction confidence below threshold"),
        "external_checked_count": len(external_results),
        "external_mismatch_count": sum(1 for result in external_results if result.get("status") == "mismatch"),
        "llm_correction_count": len(llm_corrections),
    }


def _final_confidence(counts: dict[str, int], issue_count: int) -> float:
    paper_count = max(counts["paper_count"], 1)
    issue_penalty = min(0.60, issue_count / paper_count * 0.08)
    external_penalty = min(0.20, counts["external_mismatch_count"] / paper_count * 0.15)
    llm_penalty = min(0.20, counts["llm_correction_count"] / paper_count * 0.10)
    return round(max(0.0, 1.0 - issue_penalty - external_penalty - llm_penalty), 4)


def _title_similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, _normalize_title(left), _normalize_title(right)).ratio()


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _llm_verifier_prompt(page_text: str, links: list[tuple[str, str]], papers: list[dict[str, Any]]) -> str:
    paper_summary = json.dumps(
        [
            {
                "id": paper.get("id"),
                "title": paper.get("title"),
                "authors": paper.get("authors"),
                "doi": paper.get("doi"),
                "paper_url": paper.get("paper_url"),
                "abstract": paper.get("abstract"),
            }
            for paper in papers
        ],
        ensure_ascii=False,
    )
    link_summary = "\n".join(f"- text={label!r} url={url}" for label, url in links[:500])
    return f"""
You verify conference paper-list extraction accuracy.

Return JSON only:
{{
  "corrections": [
    {{
      "paper_title": "affected paper title, or null if a missing/extra record",
      "field": "title|authors|abstract|doi|paper_url|pdf_url|missing_record|extra_record",
      "issue": "concise issue",
      "suggested_value": "correct value or null",
      "confidence": 0.0,
      "evidence_text": "short quote or page snippet"
    }}
  ]
}}

Look for missing papers, extra non-paper records, wrong DOI/link mappings, truncated abstracts, and title/author errors.
If the extraction looks correct, return {{"corrections": []}}.

Extracted papers:
<<<EXTRACTED
{paper_summary}
EXTRACTED

Cleaned page text:
<<<PAGE
{page_text[:120000]}
PAGE

Preserved links:
<<<LINKS
{link_summary}
LINKS
""".strip()
