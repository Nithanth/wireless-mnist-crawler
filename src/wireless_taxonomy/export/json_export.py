from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class JsonExporter:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def export(self, run_id: int | None, output: str | Path) -> Path:
        output = Path(output)
        if output.suffix.lower() != ".json":
            output = output.with_suffix(".json")
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "runs": self._runs(run_id),
            "papers": self._papers(run_id),
            "datasets": self._datasets(),
            "paper_dataset_links": self._paper_dataset_links(),
            "review_items": self._review_items(run_id),
            "evidence_claims": self._evidence_claims(run_id),
            "paper_list_verification_reports": self._paper_list_verification_reports(run_id),
        }
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return output

    def _runs(self, run_id: int | None) -> list[dict[str, Any]]:
        params: tuple[Any, ...] = ()
        query = "SELECT * FROM pipeline_runs"
        if run_id is not None:
            related = self._related_run_ids(run_id)
            query += f" WHERE id IN ({','.join('?' for _ in related)})"
            params = tuple(related)
        return _rows(self.conn.execute(query + " ORDER BY id", params))

    def _papers(self, run_id: int | None) -> list[dict[str, Any]]:
        params: tuple[Any, ...] = ()
        query = """
            SELECT p.*, v.name AS venue, ci.year
            FROM papers p
            JOIN conference_instances ci ON ci.id = p.conference_instance_id
            JOIN venues v ON v.id = ci.venue_id
        """
        if run_id is not None:
            run = self.conn.execute("SELECT conference_instance_id FROM pipeline_runs WHERE id = ?", (run_id,)).fetchone()
            if run is not None:
                query += " WHERE p.conference_instance_id = ?"
                params = (run["conference_instance_id"],)
        papers = _rows(self.conn.execute(query + " ORDER BY p.id", params))
        for paper in papers:
            paper["paper_sources"] = _rows(
                self.conn.execute("SELECT * FROM paper_sources WHERE paper_id = ? ORDER BY id", (paper["id"],))
            )
            paper["classifications"] = _rows(
                self.conn.execute("SELECT * FROM paper_classifications WHERE paper_id = ? ORDER BY id", (paper["id"],))
            )
            paper["datasets"] = _rows(
                self.conn.execute(
                    """
                    SELECT d.*, pdl.relationship_type, pdl.confidence AS link_confidence,
                           pdl.evidence_text AS link_evidence_text, pdl.review_needed
                    FROM paper_dataset_links pdl
                    JOIN datasets d ON d.id = pdl.dataset_id
                    WHERE pdl.paper_id = ?
                    ORDER BY d.canonical_name
                    """,
                    (paper["id"],),
                )
            )
        return papers

    def _datasets(self) -> list[dict[str, Any]]:
        datasets = _rows(self.conn.execute("SELECT * FROM datasets ORDER BY id"))
        for dataset in datasets:
            dataset["links"] = _rows(
                self.conn.execute("SELECT * FROM dataset_links WHERE dataset_id = ? ORDER BY id", (dataset["id"],))
            )
            dataset["availability_checks"] = _rows(
                self.conn.execute("SELECT * FROM availability_checks WHERE dataset_id = ? ORDER BY id", (dataset["id"],))
            )
            dataset["use_count"] = self.conn.execute(
                "SELECT COUNT(DISTINCT paper_id) AS count FROM paper_dataset_links WHERE dataset_id = ?",
                (dataset["id"],),
            ).fetchone()["count"]
        return datasets

    def _paper_dataset_links(self) -> list[dict[str, Any]]:
        return _rows(self.conn.execute("SELECT * FROM paper_dataset_links ORDER BY id"))

    def _review_items(self, run_id: int | None) -> list[dict[str, Any]]:
        params: tuple[Any, ...] = ()
        query = "SELECT * FROM review_items"
        if run_id is not None:
            related = self._related_run_ids(run_id)
            query += f" WHERE run_id IN ({','.join('?' for _ in related)})"
            params = tuple(related)
        return _rows(self.conn.execute(query + " ORDER BY id", params))

    def _evidence_claims(self, run_id: int | None) -> list[dict[str, Any]]:
        params: tuple[Any, ...] = ()
        query = "SELECT * FROM evidence_claims"
        if run_id is not None:
            related = self._related_run_ids(run_id)
            query += f" WHERE run_id IN ({','.join('?' for _ in related)})"
            params = tuple(related)
        return _rows(self.conn.execute(query + " ORDER BY id", params))

    def _paper_list_verification_reports(self, run_id: int | None) -> list[dict[str, Any]]:
        params: tuple[Any, ...] = ()
        query = "SELECT * FROM paper_list_verification_reports"
        if run_id is not None:
            related = self._related_run_ids(run_id)
            query += f" WHERE run_id IN ({','.join('?' for _ in related)})"
            params = tuple(related)
        reports = _rows(self.conn.execute(query + " ORDER BY id", params))
        for report in reports:
            raw = report.get("report_json")
            if isinstance(raw, str):
                try:
                    report["report"] = json.loads(raw)
                except json.JSONDecodeError:
                    report["report"] = None
        return reports

    def _related_run_ids(self, run_id: int) -> list[int]:
        run = self.conn.execute("SELECT conference_instance_id FROM pipeline_runs WHERE id = ?", (run_id,)).fetchone()
        if run is None or run["conference_instance_id"] is None:
            return [run_id]
        rows = self.conn.execute("SELECT id FROM pipeline_runs WHERE conference_instance_id = ?", (run["conference_instance_id"],))
        return [int(row["id"]) for row in rows] or [run_id]


def _rows(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    return [dict(row) for row in cursor.fetchall()]
