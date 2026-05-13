from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class JsonExporter:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def export(self, run_id: int | None, output: str | Path, scope: str = "related") -> Path:
        if scope not in {"related", "exact"}:
            raise ValueError("JSON export scope must be related or exact")
        output = Path(output)
        if output.suffix.lower() != ".json":
            output = output.with_suffix(".json")
        output.parent.mkdir(parents=True, exist_ok=True)
        run_ids = self._run_ids(run_id, scope)
        payload = {
            "run_id": run_id,
            "scope": scope,
            "runs": self._runs(run_ids),
            "papers": self._papers(run_id, run_ids),
            "datasets": self._datasets(run_ids),
            "paper_dataset_links": self._paper_dataset_links(run_ids),
            "paper_agentic_analyses": self._paper_agentic_analyses(run_ids),
            "paper_analysis_reflections": self._paper_analysis_reflections(run_ids),
            "paper_analysis_dataset_claims": self._paper_analysis_dataset_claims(run_ids),
            "paper_text_artifacts": self._paper_text_artifacts(run_ids),
            "paper_text_links": self._paper_text_links(run_ids),
            "paper_text_snippets": self._paper_text_snippets(run_ids),
            "paper_input_readiness": self._paper_input_readiness(run_ids),
            "review_items": self._review_items(run_ids),
            "evidence_claims": self._evidence_claims(run_ids),
            "scope_assessments": self._scope_assessments(run_ids),
            "paper_list_verification_reports": self._paper_list_verification_reports(run_ids),
        }
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return output

    def _runs(self, run_ids: list[int] | None) -> list[dict[str, Any]]:
        params: tuple[Any, ...] = ()
        query = "SELECT * FROM pipeline_runs"
        if run_ids is not None:
            query += f" WHERE id IN ({','.join('?' for _ in run_ids)})"
            params = tuple(run_ids)
        return _rows(self.conn.execute(query + " ORDER BY id", params))

    def _papers(self, run_id: int | None, run_ids: list[int] | None) -> list[dict[str, Any]]:
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
                self.conn.execute(
                    _where_run("SELECT * FROM paper_sources WHERE paper_id = ?", run_ids) + " ORDER BY id",
                    _paper_params(paper["id"], run_ids),
                )
            )
            paper["classifications"] = _rows(
                self.conn.execute(
                    _where_run("SELECT * FROM paper_classifications WHERE paper_id = ?", run_ids) + " ORDER BY id",
                    _paper_params(paper["id"], run_ids),
                )
            )
            paper["datasets"] = _rows(
                self.conn.execute(
                    _where_run(
                        """
                    SELECT d.*, pdl.relationship_type, pdl.confidence AS link_confidence,
                           pdl.evidence_text AS link_evidence_text, pdl.review_needed
                    FROM paper_dataset_links pdl
                    JOIN datasets d ON d.id = pdl.dataset_id
                    WHERE pdl.paper_id = ?
                    """,
                        run_ids,
                        table_alias="pdl",
                    )
                    + " ORDER BY d.canonical_name",
                    _paper_params(paper["id"], run_ids),
                )
            )
            paper["text_artifacts"] = _rows(
                self.conn.execute(
                    _where_run("SELECT * FROM paper_text_artifacts WHERE paper_id = ?", run_ids) + " ORDER BY id",
                    _paper_params(paper["id"], run_ids),
                )
            )
            paper["text_links"] = _rows(
                self.conn.execute(
                    _where_run("SELECT * FROM paper_text_links WHERE paper_id = ?", run_ids) + " ORDER BY id",
                    _paper_params(paper["id"], run_ids),
                )
            )
            paper["text_snippets"] = _rows(
                self.conn.execute(
                    _where_run("SELECT * FROM paper_text_snippets WHERE paper_id = ?", run_ids) + " ORDER BY id",
                    _paper_params(paper["id"], run_ids),
                )
            )
            paper["input_readiness"] = _rows(
                self.conn.execute(
                    _where_run("SELECT * FROM paper_input_readiness WHERE paper_id = ?", run_ids) + " ORDER BY id",
                    _paper_params(paper["id"], run_ids),
                )
            )
            for readiness in paper["input_readiness"]:
                _parse_json_field(readiness, "limitations_json", "limitations")
                _parse_json_field(readiness, "report_json", "report")
        return papers

    def _datasets(self, run_ids: list[int] | None) -> list[dict[str, Any]]:
        if run_ids is None:
            datasets = _rows(self.conn.execute("SELECT * FROM datasets ORDER BY id"))
        else:
            placeholders = ",".join("?" for _ in run_ids)
            datasets = _rows(
                self.conn.execute(
                    f"""
                    SELECT DISTINCT d.*
                    FROM datasets d
                    LEFT JOIN paper_dataset_links pdl ON pdl.dataset_id = d.id
                    LEFT JOIN availability_checks ac ON ac.dataset_id = d.id
                    WHERE pdl.run_id IN ({placeholders}) OR ac.run_id IN ({placeholders})
                    ORDER BY d.id
                    """,
                    (*run_ids, *run_ids),
                )
            )
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

    def _paper_dataset_links(self, run_ids: list[int] | None) -> list[dict[str, Any]]:
        return self._rows_for_runs("paper_dataset_links", run_ids)

    def _paper_agentic_analyses(self, run_ids: list[int] | None) -> list[dict[str, Any]]:
        rows = self._rows_for_runs("paper_agentic_analyses", run_ids)
        for row in rows:
            _parse_json_field(row, "modalities_json", "modalities")
            _parse_json_field(row, "osi_layers_json", "osi_layers")
            _parse_json_field(row, "analysis_json", "analysis")
        return rows

    def _paper_analysis_dataset_claims(self, run_ids: list[int] | None) -> list[dict[str, Any]]:
        rows = self._rows_for_runs("paper_analysis_dataset_claims", run_ids)
        for row in rows:
            _parse_json_field(row, "modalities_json", "modalities")
            _parse_json_field(row, "osi_layers_json", "osi_layers")
        return rows

    def _paper_analysis_reflections(self, run_ids: list[int] | None) -> list[dict[str, Any]]:
        rows = self._rows_for_runs("paper_analysis_reflections", run_ids)
        for row in rows:
            _parse_json_field(row, "issues_json", "issues")
            _parse_json_field(row, "reflection_json", "reflection")
        return rows

    def _paper_text_artifacts(self, run_ids: list[int] | None) -> list[dict[str, Any]]:
        return self._rows_for_runs("paper_text_artifacts", run_ids)

    def _paper_text_links(self, run_ids: list[int] | None) -> list[dict[str, Any]]:
        return self._rows_for_runs("paper_text_links", run_ids)

    def _paper_text_snippets(self, run_ids: list[int] | None) -> list[dict[str, Any]]:
        return self._rows_for_runs("paper_text_snippets", run_ids)

    def _paper_input_readiness(self, run_ids: list[int] | None) -> list[dict[str, Any]]:
        rows = self._rows_for_runs("paper_input_readiness", run_ids)
        for row in rows:
            _parse_json_field(row, "limitations_json", "limitations")
            _parse_json_field(row, "report_json", "report")
        return rows

    def _review_items(self, run_ids: list[int] | None) -> list[dict[str, Any]]:
        return self._rows_for_runs("review_items", run_ids)

    def _evidence_claims(self, run_ids: list[int] | None) -> list[dict[str, Any]]:
        return self._rows_for_runs("evidence_claims", run_ids)

    def _paper_list_verification_reports(self, run_ids: list[int] | None) -> list[dict[str, Any]]:
        reports = self._rows_for_runs("paper_list_verification_reports", run_ids)
        for report in reports:
            raw = report.get("report_json")
            if isinstance(raw, str):
                try:
                    report["report"] = json.loads(raw)
                except json.JSONDecodeError:
                    report["report"] = None
        return reports

    def _scope_assessments(self, run_ids: list[int] | None) -> list[dict[str, Any]]:
        reports = self._rows_for_runs("scope_assessments", run_ids)
        for report in reports:
            raw = report.get("report_json")
            if isinstance(raw, str):
                try:
                    report["report"] = json.loads(raw)
                except json.JSONDecodeError:
                    report["report"] = None
        return reports

    def _run_ids(self, run_id: int | None, scope: str) -> list[int] | None:
        if run_id is None:
            return None
        if scope == "exact":
            return [run_id]
        return self._related_run_ids(run_id)

    def _related_run_ids(self, run_id: int) -> list[int]:
        run = self.conn.execute("SELECT conference_instance_id FROM pipeline_runs WHERE id = ?", (run_id,)).fetchone()
        if run is None or run["conference_instance_id"] is None:
            return [run_id]
        rows = self.conn.execute("SELECT id FROM pipeline_runs WHERE conference_instance_id = ?", (run["conference_instance_id"],))
        return [int(row["id"]) for row in rows] or [run_id]

    def _rows_for_runs(self, table: str, run_ids: list[int] | None) -> list[dict[str, Any]]:
        params: tuple[Any, ...] = ()
        query = f"SELECT * FROM {table}"
        if run_ids is not None:
            query += f" WHERE run_id IN ({','.join('?' for _ in run_ids)})"
            params = tuple(run_ids)
        return _rows(self.conn.execute(query + " ORDER BY id", params))


def _rows(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    return [dict(row) for row in cursor.fetchall()]


def _parse_json_field(row: dict[str, Any], raw_key: str, parsed_key: str) -> None:
    raw = row.get(raw_key)
    if isinstance(raw, str):
        try:
            row[parsed_key] = json.loads(raw)
        except json.JSONDecodeError:
            row[parsed_key] = None


def _where_run(query: str, run_ids: list[int] | None, table_alias: str | None = None) -> str:
    if run_ids is None:
        return query
    column = f"{table_alias}.run_id" if table_alias else "run_id"
    return query + f" AND {column} IN ({','.join('?' for _ in run_ids)})"


def _paper_params(paper_id: int, run_ids: list[int] | None) -> tuple[Any, ...]:
    if run_ids is None:
        return (paper_id,)
    return (paper_id, *run_ids)
