from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wireless_taxonomy.analyze.full_text import _artifact, _normalize_title, _snippets_from_artifacts, _text_matches_title, extract_pdf_text
from wireless_taxonomy.models import PaperTextEnrichment, PaperTextLink


@dataclass(frozen=True)
class UnmatchedPdf:
    path: Path
    reason: str


@dataclass(frozen=True)
class LocalPdfImportResult:
    enrichments: list[PaperTextEnrichment]
    unmatched: list[UnmatchedPdf]


class LocalPdfImporter:
    provider_name = "local_pdf_import_v0"

    def import_directory(self, papers: list[dict[str, Any]], directory: str | Path) -> LocalPdfImportResult:
        root = Path(directory)
        pdf_paths = sorted(path for path in root.rglob("*.pdf") if path.is_file())
        enrichments: list[PaperTextEnrichment] = []
        unmatched: list[UnmatchedPdf] = []
        matched_paths: set[Path] = set()
        for path in pdf_paths:
            try:
                data = path.read_bytes()
                text = extract_pdf_text(data)
            except Exception as exc:
                unmatched.append(UnmatchedPdf(path, f"Could not extract PDF text: {exc}"))
                continue
            match = _best_match(papers, path, text)
            if match is None:
                unmatched.append(UnmatchedPdf(path, "No paper title or DOI matched this PDF"))
                continue
            paper, confidence, reason = match
            source_url = path.resolve().as_uri()
            artifact = _artifact(
                int(paper["id"]),
                "local_pdf_text",
                source_url,
                "fetched" if text.strip() else "empty",
                text,
                None if text.strip() else "No text extracted from local PDF",
            )
            snippets = _snippets_from_artifacts(int(paper["id"]), [artifact])
            enrichments.append(
                PaperTextEnrichment(
                    int(paper["id"]),
                    [artifact],
                    [PaperTextLink(int(paper["id"]), source_url, f"local_pdf:{reason}", "pdf", confidence)],
                    snippets,
                )
            )
            matched_paths.add(path)
        return LocalPdfImportResult(enrichments, [item for item in unmatched if item.path not in matched_paths])


def _best_match(papers: list[dict[str, Any]], path: Path, text: str) -> tuple[dict[str, Any], float, str] | None:
    best: tuple[dict[str, Any], float, str] | None = None
    lower_text = text[:12000].lower()
    for paper in papers:
        title = str(paper.get("title") or "")
        doi = str(paper.get("doi") or "").strip().lower()
        score = 0.0
        reason = ""
        if doi and doi in lower_text:
            score = 1.0
            reason = "doi"
        elif title and _text_matches_title(title, text):
            score = 0.95
            reason = "title"
        elif title:
            score = _filename_title_score(path, title)
            reason = "filename"
        if score >= 0.75 and (best is None or score > best[1]):
            best = (paper, score, reason)
    return best


def _filename_title_score(path: Path, title: str) -> float:
    filename_tokens = set(_normalize_title(path.stem).split())
    title_tokens = {token for token in _normalize_title(title).split() if len(token) >= 3}
    if not filename_tokens or not title_tokens:
        return 0.0
    return len(filename_tokens & title_tokens) / max(len(title_tokens), 1)
