
import csv
from pathlib import Path

from wireless_taxonomy.ingest.base import IngestAdapter
from wireless_taxonomy.models import PaperSeed


class CsvIngestAdapter(IngestAdapter):
    source_method = "csv"

    def __init__(self, venue: str, year: int, path: str):
        self.venue = venue
        self.year = year
        self.path = Path(path)

    def fetch(self) -> list[PaperSeed]:
        with self.path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            seeds = []
            for row in reader:
                title = row.get("Paper Title") or row.get("title") or row.get("Title") or ""
                authors = row.get("Authors") or row.get("authors") or ""
                seeds.append(
                    PaperSeed(
                        title=title.strip(),
                        authors=[a.strip() for a in authors.replace(" and ", ",").split(",") if a.strip()],
                        venue=row.get("Conference") or self.venue,
                        year=int(row.get("Year") or self.year),
                        source_url=str(self.path),
                        source_method=self.source_method,
                        source_confidence=0.99 if title and authors else 0.60,
                        evidence_text=str(dict(row)),
                    )
                )
            return seeds
