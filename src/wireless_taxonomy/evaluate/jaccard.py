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


@dataclass(frozen=True)
class JaccardAggregate:
    """Per-conference Jaccard reports plus two roll-ups across the whole sheet.

    micro pools every paper (sum of intersections / sum of unions) so large
    conferences dominate; macro averages the per-conference indices so each
    conference counts equally. `skipped` records conference instances that could
    not be scored (e.g. no wireless classification run yet).
    """

    reports: list[JaccardReport]
    skipped: list[dict[str, str]]

    @property
    def micro_index(self) -> float:
        intersection = sum(report.intersection_count for report in self.reports)
        union = sum(report.union_count for report in self.reports)
        return (intersection / union) if union else 1.0

    @property
    def macro_index(self) -> float:
        if not self.reports:
            return 1.0
        return sum(report.jaccard_index for report in self.reports) / len(self.reports)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conferences": len(self.reports),
            "micro_jaccard_index": self.micro_index,
            "macro_jaccard_index": self.macro_index,
            "totals": {
                "automated": sum(r.automated_count for r in self.reports),
                "manual": sum(r.manual_count for r in self.reports),
                "intersection": sum(r.intersection_count for r in self.reports),
                "union": sum(r.union_count for r in self.reports),
            },
            "skipped": self.skipped,
            "reports": [report.to_dict() for report in self.reports],
        }


def list_conference_runs(conn: sqlite3.Connection) -> list[tuple[int, str, int]]:
    """Return (latest_run_id, venue, year) for every conference instance that has a run."""
    rows = conn.execute(
        """
        SELECT MAX(r.id) AS run_id, v.name AS venue, ci.year AS year
        FROM conference_instances ci
        JOIN venues v ON v.id = ci.venue_id
        JOIN pipeline_runs r ON r.conference_instance_id = ci.id
        GROUP BY ci.id
        ORDER BY v.name, ci.year
        """
    ).fetchall()
    return [(int(row["run_id"]), str(row["venue"]), int(row["year"])) for row in rows]


def compute_paper_list_jaccard_all(
    conn: sqlite3.Connection,
    manual_csv: str | Path,
    title_col: str | None = None,
    authors_col: str | None = None,
    conference_col: str | None = None,
    year_col: str | None = None,
    wireless_only: bool = True,
    wireless_source: str = "classify",
    fuzzy: bool = True,
) -> JaccardAggregate:
    """Compute a per-conference Jaccard for every conference instance in the DB.

    Each instance is compared against the same manual CSV, self-filtered to that
    instance's venue+year (conference filtering is always on here — that is the
    point of the aggregate). Instances that cannot be scored are collected in
    `skipped` rather than aborting the whole run.
    """

    reports: list[JaccardReport] = []
    skipped: list[dict[str, str]] = []
    for run_id, venue, year in list_conference_runs(conn):
        try:
            reports.append(
                compute_paper_list_jaccard(
                    conn,
                    run_id,
                    manual_csv,
                    title_col=title_col,
                    authors_col=authors_col,
                    conference_col=conference_col,
                    year_col=year_col,
                    wireless_only=wireless_only,
                    wireless_source=wireless_source,
                    conference_filter=True,
                    fuzzy=fuzzy,
                )
            )
        except ValueError as exc:
            skipped.append({"venue": venue, "year": str(year), "reason": str(exc)})
    return JaccardAggregate(reports=reports, skipped=skipped)


def write_jaccard_aggregate(aggregate: JaccardAggregate, output: str | Path) -> Path:
    output = Path(output)
    if output.suffix.lower() != ".json":
        output = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return output
