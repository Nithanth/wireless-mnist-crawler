from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wireless_taxonomy.analyze.text_match import normalize_title

PAPER_SET_COLUMNS = ["match_key", "title", "abstract", "authors", "doi", "year", "venue"]

# Where the per-paper wireless decision comes from when restricting the set.
WIRELESS_SOURCES = ("classify", "agentic")


@dataclass(frozen=True)
class ConferenceRef:
    conference_instance_id: int
    venue: str
    year: int


class PaperSetExporter:
    """Flat, conference-scoped export of the papers the pipeline fetched.

    The output is intentionally minimal so it can be set-compared (e.g. Jaccard)
    against a manually curated paper list. `match_key` is the normalized title used
    as the comparison key on both sides. When `wireless_only` is set, the rows are
    restricted to papers the pipeline classified as wireless (from title+abstract),
    which is the apples-to-apples set against a hand-curated wireless list.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def conference_ref(self, run_id: int) -> ConferenceRef:
        row = self.conn.execute(
            """
            SELECT ci.id AS conference_instance_id, v.name AS venue, ci.year
            FROM pipeline_runs r
            JOIN conference_instances ci ON ci.id = r.conference_instance_id
            JOIN venues v ON v.id = ci.venue_id
            WHERE r.id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Run {run_id} not found or has no associated conference instance")
        return ConferenceRef(int(row["conference_instance_id"]), str(row["venue"]), int(row["year"]))

    def rows(self, run_id: int, wireless_only: bool = False, wireless_source: str = "classify") -> list[dict[str, Any]]:
        ref = self.conference_ref(run_id)
        papers = self.conn.execute(
            """
            SELECT p.id, p.title, p.abstract, p.authors, p.doi, ci.year, v.name AS venue
            FROM papers p
            JOIN conference_instances ci ON ci.id = p.conference_instance_id
            JOIN venues v ON v.id = ci.venue_id
            WHERE p.conference_instance_id = ?
            ORDER BY p.title
            """,
            (ref.conference_instance_id,),
        ).fetchall()
        if wireless_only:
            allowed = self._wireless_paper_ids(ref.conference_instance_id, wireless_source)
            papers = [paper for paper in papers if paper["id"] in allowed]
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

    def export(
        self,
        run_id: int,
        output: str | Path,
        fmt: str = "csv",
        wireless_only: bool = False,
        wireless_source: str = "classify",
    ) -> Path:
        if fmt not in {"csv", "json"}:
            raise ValueError("paper-set format must be csv or json")
        output = Path(output)
        rows = self.rows(run_id, wireless_only=wireless_only, wireless_source=wireless_source)
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

    def _wireless_paper_ids(self, conference_instance_id: int, wireless_source: str) -> set[int]:
        if wireless_source not in WIRELESS_SOURCES:
            raise ValueError(f"wireless_source must be one of {WIRELESS_SOURCES}")
        if wireless_source == "agentic":
            stage, table = "agentic-paper-analysis", "paper_agentic_analyses"
        else:
            stage, table = "classify-wireless", "paper_classifications"
        stage_run_id = self._latest_stage_run_id(conference_instance_id, stage)
        if stage_run_id is None:
            raise ValueError(
                f"No '{stage}' run found for this conference. Run `{stage} --run-id ...` first, "
                "or compare the full paper list with --all-papers."
            )
        rows = self.conn.execute(
            f"SELECT paper_id FROM {table} WHERE run_id = ? AND is_wireless = 1",
            (stage_run_id,),
        ).fetchall()
        return {int(row["paper_id"]) for row in rows}

    def _latest_stage_run_id(self, conference_instance_id: int, stage: str) -> int | None:
        row = self.conn.execute(
            """
            SELECT id FROM pipeline_runs
            WHERE conference_instance_id = ? AND stage = ? AND status = 'completed'
            ORDER BY id DESC LIMIT 1
            """,
            (conference_instance_id, stage),
        ).fetchone()
        return int(row["id"]) if row else None
