from __future__ import annotations

import re
from pathlib import Path

from wireless_taxonomy.ingest.base import IngestAdapter
from wireless_taxonomy.models import PaperSeed


class BibtexIngestAdapter(IngestAdapter):
    source_method = "bibtex"

    def __init__(self, venue: str, year: int, path: str):
        self.venue = venue
        self.year = year
        self.path = Path(path)

    def fetch(self) -> list[PaperSeed]:
        text = self.path.read_text(encoding="utf-8")
        entries = re.split(r"(?=@\w+\{)", text)
        seeds: list[PaperSeed] = []
        for entry in entries:
            if not entry.strip().startswith("@"):
                continue
            title = _bib_field(entry, "title")
            authors = _bib_field(entry, "author")
            doi = _bib_field(entry, "doi") or None
            abstract = _bib_field(entry, "abstract") or None
            seeds.append(
                PaperSeed(
                    title=title,
                    authors=[a.strip() for a in authors.split(" and ") if a.strip()],
                    venue=self.venue,
                    year=int(_bib_field(entry, "year") or self.year),
                    source_url=str(self.path),
                    abstract=abstract,
                    doi=doi,
                    source_method=self.source_method,
                    source_confidence=0.99 if title and authors else 0.60,
                    evidence_text=entry.strip(),
                )
            )
        return seeds


def _bib_field(entry: str, field: str) -> str:
    match = re.search(rf"(?is)\b{re.escape(field)}\s*=\s*[\{{\"](.*?)[\}}\"]\s*,", entry)
    return " ".join(match.group(1).replace("\n", " ").split()) if match else ""
