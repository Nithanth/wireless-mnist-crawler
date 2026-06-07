from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from wireless_taxonomy.analyze.text_match import normalize_title

PAPER_SET_COLUMNS = ["match_key", "title", "abstract", "authors", "doi", "year", "venue"]


class PaperSetExporter:
    """Flat, conference-scoped export of the papers the pipeline fetched.

    The output is intentionally minimal so it can be set-compared (e.g. Jaccard)
    against a manually curated paper list. `match_key` is the normalized title used
    as the comparison key on both sides.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def rows(self, run_id: int) -> list[dict[str, Any]]:
        conference_instance_id = self._conference_instance_id(run_id)
        papers = self.conn.execute(
            """
            SELECT p.title, p.abstract, p.authors, p.doi, ci.year, v.name AS venue
            FROM papers p
            JOIN conference_instances ci ON ci.id = p.conference_instance_id
            JOIN venues v ON v.id = ci.venue_id
            WHERE p.conference_instance_id = ?
            ORDER BY p.title
            """,
            (conference_instance_id,),
        ).fetchall()
        return [
            {
                "match_key": normalize_title(paper["title"]),
                "title": paper["title"],
                "abstract": paper["abstract"] or "",
                "authors": paper["authors"] or "",
                "doi": paper["doi"] or "",
                "year": paper["year"],
                "venue": paper["venue"],
            }
            for paper in papers
        ]

    def export(self, run_id: int, output: str | Path, fmt: str = "csv") -> Path:
        if fmt not in {"csv", "json"}:
            raise ValueError("paper-set format must be csv or json")
        output = Path(output)
        rows = self.rows(run_id)
        if fmt == "json":
            if output.suffix.lower() != ".json":
                output = output.with_suffix(".json")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
            return output
        if output.suffix.lower() != ".csv":
            output = output.with_suffix(".csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=PAPER_SET_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return output

    def _conference_instance_id(self, run_id: int) -> int:
        run = self.conn.execute(
            "SELECT conference_instance_id FROM pipeline_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        if run["conference_instance_id"] is None:
            raise ValueError(f"Run {run_id} has no associated conference instance")
        return int(run["conference_instance_id"])
