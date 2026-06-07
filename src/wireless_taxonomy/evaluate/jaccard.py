from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wireless_taxonomy.analyze.text_match import normalize_title
from wireless_taxonomy.evaluate.matching import MatchPair, MatchResult, PaperRecord, make_record, match_papers
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
    mean_wireless_confidence: float | None = None

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
            "mean_wireless_confidence": self.mean_wireless_confidence,
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


@dataclass(frozen=True)
class Evaluation:
    """Everything produced by matching one run against the manual CSV.

    Carries the rich automated rows (including each paper's wireless label and
    classifier confidence) so both the `JaccardReport` summary and the per-paper
    comparison CSV can be derived from a single matching pass.
    """

    run_id: int
    venue: str
    year: int
    wireless_only: bool
    fuzzy: bool
    title_column: str
    authors_column: str | None
    conference_filtered: bool
    automated_rows: list[dict[str, Any]]
    automated: list[PaperRecord]
    manual: list[PaperRecord]
    result: MatchResult


def evaluate_run(
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
) -> Evaluation:
    exporter = PaperSetExporter(conn)
    ref = exporter.conference_ref(run_id)

    automated_rows = exporter.rows(run_id, wireless_only=wireless_only, wireless_source=wireless_source)
    automated = [make_record(row["title"], row["authors"]) for row in automated_rows]

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
    return Evaluation(
        run_id=run_id,
        venue=ref.venue,
        year=ref.year,
        wireless_only=wireless_only,
        fuzzy=fuzzy,
        title_column=title_column,
        authors_column=authors_column,
        conference_filtered=conference_filtered,
        automated_rows=automated_rows,
        automated=automated,
        manual=manual,
        result=result,
    )


def report_from_evaluation(evaluation: Evaluation) -> JaccardReport:
    result = evaluation.result
    # Counts use deduplicated normalized-title keys per side.
    automated_count = len({record.key for record in evaluation.automated if record.key})
    manual_count = len({record.key for record in evaluation.manual if record.key})
    intersection_count = len(result.matched)
    union_count = automated_count + manual_count - intersection_count
    jaccard_index = (intersection_count / union_count) if union_count else 1.0
    fuzzy_matches = [pair for pair in result.matched if pair.method == "fuzzy"]
    return JaccardReport(
        run_id=evaluation.run_id,
        venue=evaluation.venue,
        year=evaluation.year,
        wireless_only=evaluation.wireless_only,
        fuzzy=evaluation.fuzzy,
        title_column=evaluation.title_column,
        authors_column=evaluation.authors_column,
        conference_filtered=evaluation.conference_filtered,
        jaccard_index=jaccard_index,
        intersection_count=intersection_count,
        union_count=union_count,
        automated_count=automated_count,
        manual_count=manual_count,
        matched=[pair.manual_title for pair in result.matched],
        fuzzy_matches=fuzzy_matches,
        missed_by_cli=result.missed_by_cli,
        extra_from_cli=result.extra_from_cli,
        mean_wireless_confidence=_mean_confidence(evaluation.automated_rows),
    )


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
    evaluation = evaluate_run(
        conn,
        run_id,
        manual_csv,
        title_col=title_col,
        authors_col=authors_col,
        conference_col=conference_col,
        year_col=year_col,
        wireless_only=wireless_only,
        wireless_source=wireless_source,
        conference_filter=conference_filter,
        fuzzy=fuzzy,
    )
    return report_from_evaluation(evaluation)


# Per-paper comparison view (one row per paper, with a status column) for CSV.
COMPARISON_COLUMNS = [
    "venue",
    "year",
    "status",
    "manual_title",
    "automated_title",
    "title_similarity",
    "author_overlap",
    "shared_authors",
    "wireless_label",
    "wireless_confidence",
]


def comparison_rows(evaluation: Evaluation) -> list[dict[str, Any]]:
    """Flatten an evaluation into one row per paper with a `status` column.

    status is one of matched / fuzzy_matched / missed_by_cli / extra_from_cli.
    Automated-side rows carry the wireless label + classifier confidence.
    """

    auto_by_key = {normalize_title(row["title"]): row for row in evaluation.automated_rows}
    venue, year = evaluation.venue, evaluation.year
    rows: list[dict[str, Any]] = []

    def _auto_fields(title: str) -> tuple[Any, Any]:
        auto = auto_by_key.get(normalize_title(title), {})
        return auto.get("wireless_label", ""), auto.get("wireless_confidence", "")

    for pair in evaluation.result.matched:
        label, confidence = _auto_fields(pair.automated_title)
        rows.append(
            {
                "venue": venue,
                "year": year,
                "status": "matched" if pair.method == "exact" else "fuzzy_matched",
                "manual_title": pair.manual_title,
                "automated_title": pair.automated_title,
                "title_similarity": round(pair.title_similarity, 4),
                "author_overlap": round(pair.author_overlap, 4),
                "shared_authors": "; ".join(pair.shared_authors),
                "wireless_label": label,
                "wireless_confidence": confidence,
            }
        )
    for title in evaluation.result.missed_by_cli:
        rows.append(
            {
                "venue": venue,
                "year": year,
                "status": "missed_by_cli",
                "manual_title": title,
                "automated_title": "",
                "title_similarity": "",
                "author_overlap": "",
                "shared_authors": "",
                "wireless_label": "",
                "wireless_confidence": "",
            }
        )
    for title in evaluation.result.extra_from_cli:
        label, confidence = _auto_fields(title)
        rows.append(
            {
                "venue": venue,
                "year": year,
                "status": "extra_from_cli",
                "manual_title": "",
                "automated_title": title,
                "title_similarity": "",
                "author_overlap": "",
                "shared_authors": "",
                "wireless_label": label,
                "wireless_confidence": confidence,
            }
        )
    return rows


def write_comparison_csv(rows: list[dict[str, Any]], output: str | Path) -> Path:
    output = Path(output)
    if output.suffix.lower() != ".csv":
        output = output.with_suffix(".csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output


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


def comparison_rows_all(
    conn: sqlite3.Connection,
    manual_csv: str | Path,
    title_col: str | None = None,
    authors_col: str | None = None,
    conference_col: str | None = None,
    year_col: str | None = None,
    wireless_only: bool = True,
    wireless_source: str = "classify",
    fuzzy: bool = True,
) -> list[dict[str, Any]]:
    """Per-paper comparison rows for every scorable conference instance, combined.

    Mirrors `compute_paper_list_jaccard_all` but returns the flat per-paper view
    (venue/year columns distinguish conferences). Instances that can't be scored
    are simply omitted, matching the aggregate's `skipped` behaviour.
    """

    rows: list[dict[str, Any]] = []
    for run_id, _venue, _year in list_conference_runs(conn):
        try:
            evaluation = evaluate_run(
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
        except ValueError:
            continue
        rows.extend(comparison_rows(evaluation))
    return rows


def _mean_confidence(evaluation_rows: list[dict[str, Any]]) -> float | None:
    values = [
        float(row["wireless_confidence"])
        for row in evaluation_rows
        if row.get("wireless_confidence") not in ("", None)
    ]
    return sum(values) / len(values) if values else None


def format_report_summary(report: JaccardReport) -> str:
    """Readable multi-line summary of one conference's Jaccard result."""

    conf = "n/a" if report.mean_wireless_confidence is None else f"{report.mean_wireless_confidence:.2f}"
    lines = [
        f"{report.venue} {report.year}  —  Jaccard (IoU) = {report.jaccard_index:.4f}",
        f"  matched (intersection) : {report.intersection_count:>4}   (fuzzy: {len(report.fuzzy_matches)})",
        f"  union                  : {report.union_count:>4}",
        f"  automated / manual     : {report.automated_count:>4} / {report.manual_count}",
        f"  missed_by_cli          : {len(report.missed_by_cli):>4}  (curated wireless the CLI didn't flag)",
        f"  extra_from_cli         : {len(report.extra_from_cli):>4}  (CLI-flagged, not in your sheet)",
        f"  mean wireless confidence (automated): {conf}",
    ]
    return "\n".join(lines)


def format_aggregate_summary(aggregate: JaccardAggregate) -> str:
    """Readable summary table for `jaccard-all` across every conference."""

    lines: list[str] = []
    for report in aggregate.reports:
        lines.append(
            f"  {report.venue} {report.year}: index={report.jaccard_index:.4f} "
            f"matched={report.intersection_count} union={report.union_count} "
            f"missed={len(report.missed_by_cli)} extra={len(report.extra_from_cli)} "
            f"(fuzzy={len(report.fuzzy_matches)})"
        )
    for entry in aggregate.skipped:
        lines.append(f"  SKIPPED {entry['venue']} {entry['year']}: {entry['reason']}")
    header = (
        f"Aggregate coverage (Jaccard/IoU) over {len(aggregate.reports)} conference(s), "
        f"{len(aggregate.skipped)} skipped:"
    )
    footer = (
        f"  micro (pooled papers)            = {aggregate.micro_index:.4f}\n"
        f"  macro (mean of per-conference)   = {aggregate.macro_index:.4f}"
    )
    return "\n".join([header, *lines, footer])
