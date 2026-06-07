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
# column in a manually curated CSV that holds the paper title.
_TITLE_COLUMN_CANDIDATES = ("title", "paper title", "paper_title", "papertitle", "paper")


@dataclass(frozen=True)
class JaccardReport:
    """Jaccard (IoU) comparison between the automated and manual paper sets.

    Keys are normalized titles. `missed_by_cli` are papers in the manual set the
    pipeline did not fetch; `extra_from_cli` are papers the pipeline fetched that
    are absent from the manual set.
    """

    run_id: int
    title_column: str
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
            "title_column": self.title_column,
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


def detect_title_column(headers: list[str], override: str | None = None) -> str:
    """Return the actual header to read titles from.

    When `override` is provided it is matched case-insensitively against the
    headers. Otherwise the first known title-like header wins.
    """

    normalized = {(header or "").strip().lower(): header for header in headers}
    if override:
        actual = normalized.get(override.strip().lower())
        if actual is None:
            raise ValueError(
                f"Title column {override!r} not found. Available columns: {', '.join(headers)}"
            )
        return actual
    for candidate in _TITLE_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(
        "Could not auto-detect a title column. "
        f"Available columns: {', '.join(headers)}. Pass --title-col to choose one."
    )


def load_manual_paper_keys(manual_csv: str | Path, title_col: str | None = None) -> tuple[dict[str, str], str]:
    """Load a manual CSV into a mapping of normalized title -> original title.

    Returns the mapping plus the resolved title column. utf-8-sig handles the BOM
    that spreadsheet exports (e.g. Google Sheets) commonly prepend.
    """

    path = Path(manual_csv)
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        headers = list(reader.fieldnames or [])
        if not headers:
            raise ValueError(f"Manual CSV {path} has no header row")
        column = detect_title_column(headers, title_col)
        keys: dict[str, str] = {}
        for row in reader:
            title = (row.get(column) or "").strip()
            if not title:
                continue
            key = normalize_title(title)
            if not key:
                continue
            keys.setdefault(key, title)
    return keys, column


def compute_paper_list_jaccard(
    conn: sqlite3.Connection,
    run_id: int,
    manual_csv: str | Path,
    title_col: str | None = None,
) -> JaccardReport:
    automated: dict[str, str] = {}
    for row in PaperSetExporter(conn).rows(run_id):
        key = row["match_key"]
        if key:
            automated.setdefault(key, row["title"])

    manual, column = load_manual_paper_keys(manual_csv, title_col)

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
        title_column=column,
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
