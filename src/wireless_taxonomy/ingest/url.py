
import re
from typing import Any

from wireless_taxonomy.config import LlmSettings
from wireless_taxonomy.ingest.adapters import detect_source_hint
from wireless_taxonomy.ingest.base import IngestAdapter
from wireless_taxonomy.ingest.clean_page import CleanPage, fetch_clean_page
from wireless_taxonomy.llm import LlmRequest, LlmRouter
from wireless_taxonomy.models import PaperSeed


class UrlIngestAdapter(IngestAdapter):
    source_method = "url"

    def __init__(self, venue: str, year: int, source_url: str, llm_settings: LlmSettings | None = None):
        self.venue = venue
        self.year = year
        self.source_url = source_url
        self.llm_settings = llm_settings

    def fetch(self) -> list[PaperSeed]:
        page = fetch_clean_page(self.source_url)
        hint = detect_source_hint(self.source_url, page.text)
        deterministic = extract_paper_seeds_deterministic(page, self.venue, self.year, hint)
        if deterministic:
            return deterministic
        if self.llm_settings:
            return extract_paper_seeds_llm(page, self.venue, self.year, hint, self.llm_settings)
        return []


def extract_paper_seeds_llm(page: CleanPage, venue: str, year: int, source_hint: str, llm_settings: LlmSettings) -> list[PaperSeed]:
    response = LlmRouter(llm_settings).complete(
        LlmRequest(
            task="paper_list_extraction",
            schema_name="PaperSeedList",
            prompt=_paper_list_prompt(page, venue, year, source_hint),
            metadata={"venue": venue, "year": year, "source_url": page.source_url},
        )
    )
    return paper_seeds_from_llm_payload(response.parsed, venue, year, page.source_url, f"{source_hint}_llm_structured_extraction")


def paper_seeds_from_llm_payload(payload: dict[str, Any] | list[Any] | None, venue: str, year: int, source_url: str, source_method: str) -> list[PaperSeed]:
    if payload is None:
        return []
    papers = payload.get("papers") if isinstance(payload, dict) else payload
    if not isinstance(papers, list):
        return []
    seeds: list[PaperSeed] = []
    for item in papers:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        authors_value = item.get("authors") or []
        if isinstance(authors_value, str):
            authors = [authors_value.strip()] if authors_value.strip() else []
        elif isinstance(authors_value, list):
            authors = [str(author).strip() for author in authors_value if str(author).strip()]
        else:
            authors = []
        confidence = _float_or_default(item.get("confidence"), 0.80)
        seeds.append(
            PaperSeed(
                title=title,
                authors=authors,
                venue=str(item.get("venue") or venue),
                year=int(item.get("year") or year),
                source_url=source_url,
                abstract=_optional_str(item.get("abstract")),
                doi=_optional_str(item.get("doi")),
                paper_url=_optional_str(item.get("paper_url") or item.get("acm_url")),
                pdf_url=_optional_str(item.get("pdf_url")),
                session=_optional_str(item.get("session")),
                source_method=source_method,
                source_confidence=confidence,
                evidence_text=_optional_str(item.get("evidence_text")),
            )
        )
    return seeds


def extract_paper_seeds_deterministic(page: CleanPage, venue: str, year: int, source_hint: str = "generic") -> list[PaperSeed]:
    """Fast deterministic extraction for simple fixtures and known program-page patterns."""
    if "--- PAPER ---" not in page.text and not re.search(r"(?im)^Title:\s+", page.text):
        program_style = _extract_program_style(page, venue, year, source_hint)
        if program_style:
            return program_style
    blocks = _split_blocks(page.text)
    seeds: list[PaperSeed] = []
    for block in blocks:
        title = _field(block, "Title")
        authors = _field(block, "Authors")
        if not title:
            continue
        doi = _field(block, "DOI") or None
        abstract = _field(block, "Abstract") or None
        session = _field(block, "Session") or None
        paper_url = _first_link(block, ["ACM", "Paper", "DOI"])
        pdf_url = _first_link(block, ["PDF"])
        seeds.append(
            PaperSeed(
                title=title,
                authors=[a.strip() for a in re.split(r",| and ", authors) if a.strip()],
                venue=venue,
                year=year,
                source_url=page.source_url,
                abstract=abstract,
                doi=doi,
                paper_url=paper_url,
                pdf_url=pdf_url,
                session=session,
                source_method=f"{source_hint}_labeled_block_parser",
                source_confidence=0.95 if authors else 0.60,
                evidence_text=block,
            )
        )
    return seeds


def _extract_program_style(page: CleanPage, venue: str, year: int, source_hint: str) -> list[PaperSeed]:
    lines = [line.strip() for line in page.text.splitlines() if line.strip()]
    if not any(line.startswith("Abstract:") for line in lines):
        return []
    start = _program_start(lines)
    acm_links = [
        url
        for _, url in page.links
        if "dl.acm.org/doi/10." in url and "/doi/proceedings/" not in url
    ]
    seeds: list[PaperSeed] = []
    i = start
    session: str | None = None
    while i < len(lines):
        line = lines[i]
        if _is_session_line(line):
            session = line.split("|", 1)[1].strip() if "|" in line else line
            i += 1
            continue
        if _is_control_line(line):
            i += 1
            continue
        abstract_idx = _next_abstract_index(lines, i)
        if abstract_idx is None:
            i += 1
            continue
        title = line
        authors_text = " ".join(lines[i + 1 : abstract_idx])
        j = abstract_idx + 1
        abstract_parts = [lines[abstract_idx].removeprefix("Abstract:").strip()]
        while j < len(lines) and not _is_control_line(lines[j]) and not _looks_like_record_start(lines, j):
            abstract_parts.append(lines[j])
            j += 1
        acm_url = acm_links[len(seeds)] if len(seeds) < len(acm_links) else None
        seeds.append(
            PaperSeed(
                title=title,
                authors=[authors_text],
                venue=venue,
                year=year,
                source_url=page.source_url,
                abstract=" ".join(part for part in abstract_parts if part).strip() or None,
                doi=_doi_from_acm_url(acm_url),
                paper_url=acm_url,
                pdf_url=None,
                session=session,
                source_method=f"{source_hint}_program_page_parser",
                source_confidence=0.93 if authors_text and abstract_parts else 0.70,
                evidence_text="\n".join(lines[i:j]),
            )
        )
        i = j
    return seeds


def _split_blocks(text: str) -> list[str]:
    if "--- PAPER ---" in text:
        return [part.strip() for part in text.split("--- PAPER ---") if part.strip()]
    # Fallback for simple lists that use repeated Title:/Authors: pairs.
    starts = [m.start() for m in re.finditer(r"(?im)^Title:\s+", text)]
    if not starts:
        return [text]
    starts.append(len(text))
    return [text[starts[i] : starts[i + 1]].strip() for i in range(len(starts) - 1)]


def _field(block: str, name: str) -> str:
    pattern = rf"(?ims)^{re.escape(name)}:\s*(.*?)(?=^\w[\w /()-]*:\s|\Z)"
    match = re.search(pattern, block)
    return " ".join(match.group(1).split()) if match else ""


def _first_link(block: str, labels: list[str]) -> str | None:
    for label in labels:
        pattern = rf"{re.escape(label)}[^\[]*\[(https?://[^\]]+|file://[^\]]+)\]"
        match = re.search(pattern, block, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _program_start(lines: list[str]) -> int:
    for marker in ("Papers Info", "Accepted Papers"):
        if marker in lines:
            return lines.index(marker) + 1
    return 0


def _next_abstract_index(lines: list[str], title_index: int) -> int | None:
    for idx in range(title_index + 2, min(title_index + 6, len(lines))):
        if lines[idx].startswith("Abstract:"):
            return idx
    return None


def _looks_like_record_start(lines: list[str], idx: int) -> bool:
    return (
        idx + 2 < len(lines)
        and not _is_control_line(lines[idx])
        and not lines[idx].startswith("Abstract:")
        and _looks_like_author_line(lines[idx + 1])
        and _next_abstract_index(lines, idx) is not None
    )


def _is_control_line(line: str) -> bool:
    return (
        line.startswith("Day ")
        or line.startswith("Session Chair:")
        or _is_session_line(line)
        or line in {"Papers Info"}
        or line.startswith("Proceedings of ")
        or line.startswith("Welcome to ")
        or "Best Paper Award" in line
    )


def _is_session_line(line: str) -> bool:
    return bool(re.search(r"\d{1,2}:\d{2}\s*[—-]\s*\d{1,2}:\d{2}\s*\|", line))


def _looks_like_author_line(line: str) -> bool:
    return "(" in line or ";" in line


def _doi_from_acm_url(url: str | None) -> str | None:
    if not url:
        return None
    match = re.search(r"/doi/(10\.\d{4,9}/[^/?#]+)", url)
    return match.group(1) if match else None


def _paper_list_prompt(page: CleanPage, venue: str, year: int, source_hint: str) -> str:
    max_chars = 120_000
    text = page.text[:max_chars]
    links = "\n".join(f"- text={label!r} url={url}" for label, url in page.links[:500])
    return f"""
You extract accepted conference paper records from heterogeneous conference webpages.

Venue: {venue}
Year: {year}
Source type hint: {source_hint}
Source URL: {page.source_url}

Return JSON only, with this exact top-level shape:
{{
  "papers": [
    {{
      "title": "full paper title",
      "authors": ["author names or the exact author line if individual names are ambiguous"],
      "venue": "{venue}",
      "year": {year},
      "session": "session name if visible, otherwise null",
      "abstract": "full abstract if visible, otherwise null",
      "doi": "DOI if visible or inferable from ACM/IEEE DOI URL, otherwise null",
      "paper_url": "canonical paper landing page URL if visible, otherwise null",
      "pdf_url": "PDF URL if visible, otherwise null",
      "confidence": 0.0,
      "evidence_text": "short source snippet proving this record"
    }}
  ]
}}

Extraction rules:
- Extract accepted/research/full conference papers, not navigation, sessions, awards without paper records, keynotes, chairs, workshops, or videos alone.
- Preserve titles exactly.
- Preserve author information with high fidelity; if individual author splitting is uncertain, put the exact author line as one array item.
- Include all visible abstracts and continue across wrapped lines.
- Use preserved links to assign DOI, paper URL, and PDF URL when possible.
- If uncertain about a record, include it with confidence below 0.90 rather than omitting it.
- Do not invent missing metadata.

Cleaned page text:
<<<PAGE_TEXT
{text}
PAGE_TEXT

Preserved links:
<<<LINKS
{links}
LINKS
""".strip()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
