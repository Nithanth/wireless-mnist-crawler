from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wireless_taxonomy.analyze.text_match import normalize_title
from wireless_taxonomy.export.paper_set import PaperSetExporter

# Candidate header names (compared case-insensitively after trimming) for the
# columns in a manually curated CSV.
_TITLE_COLUMN_CANDIDATES = ("title", "paper title", "paper_title", "papertitle", "paper")
_CONFERENCE_COLUMN_CANDIDATES = ("conference", "venue")
_YEAR_COLUMN_CANDIDATES = ("year",)


@dataclass(frozen=True)
class JaccardReport:
    """Jaccard (IoU) comparison between the automated and manual paper sets.

    Keys are normalized titles. `missed_by_cli` are papers in the manual set the
    pipeline did not capture; `extra_from_cli` are papers the pipeline captured
    that are absent from the manual set.
    """

    run_id: int
    venue: str
    year: int
    wireless_only: bool
    title_column: str
    conference_filtered: bool
    jaccard_index: float
    intersection_count: int
    union_count: int
    automated_count: int
    manual_count: int
    matched: list[str]
    missed_by_cli: list[str]
    extra_from_cli: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "venue": self.venue,
            "year": self.year,
            "wireless_only": self.wireless_only,
            "title_column": self.title_column,
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


def load_manual_paper_keys(
    manual_csv: str | Path,
    title_col: str | None = None,
    conference_col: str | None = None,
    year_col: str | None = None,
    venue: str | None = None,
    year: int | None = None,
) -> tuple[dict[str, str], str, bool]:
    """Load a manual CSV into a mapping of normalized title -> original title.

    When `venue`/`year` are given and the CSV exposes conference/year columns, rows
    are filtered to that conference instance so a multi-conference sheet compares
    like-for-like against a single run. utf-8-sig handles the BOM that spreadsheet
    exports (e.g. Google Sheets) commonly prepend. Returns the mapping, the resolved
    title column, and whether conference filtering was applied.
    """

    path = Path(manual_csv)
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        headers = list(reader.fieldnames or [])
        if not headers:
            raise ValueError(f"Manual CSV {path} has no header row")
        title_column = detect_title_column(headers, title_col)
        conference_column = _detect_column(headers, _CONFERENCE_COLUMN_CANDIDATES, conference_col, "conference", required=False)
        year_column = _detect_column(headers, _YEAR_COLUMN_CANDIDATES, year_col, "year", required=False)
        apply_filter = venue is not None and year is not None and conference_column is not None and year_column is not None
        keys: dict[str, str] = {}
        for row in reader:
            if apply_filter:
                row_venue = (row.get(conference_column) or "").strip().lower()
                row_year = (row.get(year_column) or "").strip()
                if row_venue != venue.strip().lower() or row_year != str(year):
                    continue
            title = (row.get(title_column) or "").strip()
            if not title:
                continue
            key = normalize_title(title)
            if not key:
                continue
            keys.setdefault(key, title)
    return keys, title_column, apply_filter


def compute_paper_list_jaccard(
    conn: sqlite3.Connection,
    run_id: int,
    manual_csv: str | Path,
    title_col: str | None = None,
    conference_col: str | None = None,
    year_col: str | None = None,
    wireless_only: bool = True,
    wireless_source: str = "classify",
    conference_filter: bool = True,
) -> JaccardReport:
    exporter = PaperSetExporter(conn)
    ref = exporter.conference_ref(run_id)

    automated: dict[str, str] = {}
    for row in exporter.rows(run_id, wireless_only=wireless_only, wireless_source=wireless_source):
        key = row["match_key"]
        if key:
            automated.setdefault(key, row["title"])

    filter_venue = ref.venue if conference_filter else None
    filter_year = ref.year if conference_filter else None
    manual, title_column, conference_filtered = load_manual_paper_keys(
        manual_csv,
        title_col=title_col,
        conference_col=conference_col,
        year_col=year_col,
        venue=filter_venue,
        year=filter_year,
    )

    automated_keys = set(automated)
    manual_keys = set(manual)
    intersection = automated_keys & manual_keys
    union = automated_keys | manual_keys
    jaccard_index = (len(intersection) / len(union)) if union else 1.0

    matched = sorted(manual[key] for key in intersection)
    missed_by_cli = sorted(manual[key] for key in (manual_keys - automated_keys))
    extra_from_cli = sorted(automated[key] for key in (automated_keys - manual_keys))

    return JaccardReport(
        run_id=run_id,
        venue=ref.venue,
        year=ref.year,
        wireless_only=wireless_only,
        title_column=title_column,
        conference_filtered=conference_filtered,
        jaccard_index=jaccard_index,
        intersection_count=len(intersection),
        union_count=len(union),
        automated_count=len(automated_keys),
        manual_count=len(manual_keys),
        matched=matched,
        missed_by_cli=missed_by_cli,
        extra_from_cli=extra_from_cli,
    )


def write_jaccard_report(report: JaccardReport, output: str | Path) -> Path:
    output = Path(output)
    if output.suffix.lower() != ".json":
        output = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return output
