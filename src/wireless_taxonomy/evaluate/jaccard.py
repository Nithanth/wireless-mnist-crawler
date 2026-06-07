from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wireless_taxonomy.evaluate.matching import MatchPair, PaperRecord, make_record, match_papers
from wireless_taxonomy.export.paper_set import PaperSetExporter

# Candidate header names (compared case-insensitively after trimming) for the
# columns in a manually curated CSV.
_TITLE_COLUMN_CANDIDATES = ("title", "paper title", "paper_title", "papertitle", "paper")
_CONFERENCE_COLUMN_CANDIDATES = ("conference", "venue")
_YEAR_COLUMN_CANDIDATES = ("year",)
_AUTHORS_COLUMN_CANDIDATES = ("authors", "author")


@dataclass(frozen=True)
class JaccardReport:
    """Jaccard (IoU) comparison between the automated and manual paper sets.

    Papers are matched one-to-one by normalized title, optionally with fuzzy
    title similarity boosted by author overlap. `missed_by_cli` are papers in the
    manual set the pipeline did not capture; `extra_from_cli` are papers the
    pipeline captured that are absent from the manual set.
    """

    run_id: int
    venue: str
    year: int
    wireless_only: bool
    fuzzy: bool
    title_column: str
    authors_column: str | None
    conference_filtered: bool
    jaccard_index: float
    intersection_count: int
    union_count: int
    automated_count: int
    manual_count: int
    matched: list[str]
    fuzzy_matches: list[MatchPair]
    missed_by_cli: list[str]
    extra_from_cli: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "venue": self.venue,
            "year": self.year,
            "wireless_only": self.wireless_only,
            "fuzzy": self.fuzzy,
            "title_column": self.title_column,
            "authors_column": self.authors_column,
            "conference_filtered": self.conference_filtered,
            "jaccard_index": self.jaccard_index,
            "counts": {
                "automated": self.automated_count,
                "manual": self.manual_count,
                "intersection": self.intersection_count,
                "union": self.union_count,
                "missed_by_cli": len(self.missed_by_cli),
                "extra_from_cli": len(self.extra_from_cli),
            },
            "matched": self.matched,
            "fuzzy_matches": [
                {
                    "manual_title": pair.manual_title,
                    "automated_title": pair.automated_title,
                    "title_similarity": round(pair.title_similarity, 4),
                    "author_overlap": round(pair.author_overlap, 4),
                    "shared_authors": pair.shared_authors,
                }
                for pair in self.fuzzy_matches
            ],
            "missed_by_cli": self.missed_by_cli,
            "extra_from_cli": self.extra_from_cli,
        }


def _detect_column(headers: list[str], candidates: tuple[str, ...], override: str | None, kind: str, required: bool) -> str | None:
    normalized = {(header or "").strip().lower(): header for header in headers}
    if override:
        actual = normalized.get(override.strip().lower())
        if actual is None:
            raise ValueError(f"{kind} column {override!r} not found. Available columns: {', '.join(headers)}")
        return actual
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    if required:
        raise ValueError(
            f"Could not auto-detect a {kind} column. "
            f"Available columns: {', '.join(headers)}. Pass the matching --*-col option."
        )
    return None


def detect_title_column(headers: list[str], override: str | None = None) -> str:
    column = _detect_column(headers, _TITLE_COLUMN_CANDIDATES, override, "title", required=True)
    assert column is not None  # required=True guarantees a value
    return column


def load_manual_records(
    manual_csv: str | Path,
    title_col: str | None = None,
    authors_col: str | None = None,
    conference_col: str | None = None,
    year_col: str | None = None,
    venue: str | None = None,
    year: int | None = None,
) -> tuple[list[PaperRecord], str, str | None, bool]:
    """Load a manual CSV into `PaperRecord`s (title + authors for matching).

    When `venue`/`year` are given and the CSV exposes conference/year columns, rows
    are filtered to that conference instance so a multi-conference sheet compares
    like-for-like against a single run. utf-8-sig handles the BOM that spreadsheet
    exports (e.g. Google Sheets) commonly prepend. Returns the records, the resolved
    title column, the resolved authors column (if any), and whether conference
    filtering was applied.
    """

    path = Path(manual_csv)
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        headers = list(reader.fieldnames or [])
        if not headers:
            raise ValueError(f"Manual CSV {path} has no header row")
        title_column = detect_title_column(headers, title_col)
        authors_column = _detect_column(headers, _AUTHORS_COLUMN_CANDIDATES, authors_col, "authors", required=False)
        conference_column = _detect_column(headers, _CONFERENCE_COLUMN_CANDIDATES, conference_col, "conference", required=False)
        year_column = _detect_column(headers, _YEAR_COLUMN_CANDIDATES, year_col, "year", required=False)
        apply_filter = venue is not None and year is not None and conference_column is not None and year_column is not None
        records: list[PaperRecord] = []
        for row in reader:
            if apply_filter:
                row_venue = (row.get(conference_column) or "").strip().lower()
                row_year = (row.get(year_column) or "").strip()
                if row_venue != venue.strip().lower() or row_year != str(year):
                    continue
            title = (row.get(title_column) or "").strip()
            if not title:
                continue
            authors = (row.get(authors_column) or "") if authors_column else ""
            records.append(make_record(title, authors))
    return records, title_column, authors_column, apply_filter


def compute_paper_list_jaccard(
    conn: sqlite3.Connection,
    run_id: int,
    manual_csv: str | Path,
    title_col: str | None = None,
    authors_col: str | None = None,
    conference_col: str | None = None,
    year_col: str | None = None,
    wireless_only: bool = True,
    wireless_source: str = "classify",
    conference_filter: bool = True,
    fuzzy: bool = True,
) -> JaccardReport:
    exporter = PaperSetExporter(conn)
    ref = exporter.conference_ref(run_id)

    automated = [
        make_record(row["title"], row["authors"])
        for row in exporter.rows(run_id, wireless_only=wireless_only, wireless_source=wireless_source)
    ]

    filter_venue = ref.venue if conference_filter else None
    filter_year = ref.year if conference_filter else None
    manual, title_column, authors_column, conference_filtered = load_manual_records(
        manual_csv,
        title_col=title_col,
        authors_col=authors_col,
        conference_col=conference_col,
        year_col=year_col,
        venue=filter_venue,
        year=filter_year,
    )

    result = match_papers(manual, automated, fuzzy=fuzzy)

    # Counts use deduplicated normalized-title keys per side.
    automated_count = len({record.key for record in automated if record.key})
    manual_count = len({record.key for record in manual if record.key})
    intersection_count = len(result.matched)
    union_count = automated_count + manual_count - intersection_count
    jaccard_index = (intersection_count / union_count) if union_count else 1.0

    fuzzy_matches = [pair for pair in result.matched if pair.method == "fuzzy"]

    return JaccardReport(
        run_id=run_id,
        venue=ref.venue,
        year=ref.year,
        wireless_only=wireless_only,
        fuzzy=fuzzy,
        title_column=title_column,
        authors_column=authors_column,
        conference_filtered=conference_filtered,
        jaccard_index=jaccard_index,
        intersection_count=intersection_count,
        union_count=union_count,
        automated_count=automated_count,
        manual_count=manual_count,
        matched=[pair.manual_title for pair in result.matched],
        fuzzy_matches=fuzzy_matches,
        missed_by_cli=result.missed_by_cli,
        extra_from_cli=result.extra_from_cli,
    )


def write_jaccard_report(report: JaccardReport, output: str | Path) -> Path:
    output = Path(output)
    if output.suffix.lower() != ".json":
        output = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return output
