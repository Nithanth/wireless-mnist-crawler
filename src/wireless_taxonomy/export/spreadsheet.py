from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path

from openpyxl import Workbook

from wireless_taxonomy.export.schemas import (
    BIBTEX_SHEET,
    DATASETS_SHEET,
    EVIDENCE_SHEET,
    PAPER_DATASET_LINKS_SHEET,
    PAPERS_SHEET,
    REVIEW_SHEET,
    SHEETS,
)


class SpreadsheetExporter:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def export(self, run_id: int | None, output: str | Path, fmt: str = "xlsx") -> Path:
        output = Path(output)
        data = self._collect(run_id)
        if fmt == "csv":
            output.mkdir(parents=True, exist_ok=True)
            for sheet, rows in data.items():
                path = output / f"{_slug(sheet)}.csv"
                with path.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(SHEETS[sheet])
                    writer.writerows(rows)
            return output
        if fmt != "xlsx":
            raise ValueError("format must be xlsx, csv, or json")
        if output.suffix.lower() != ".xlsx":
            output = output.with_suffix(".xlsx")
        output.parent.mkdir(parents=True, exist_ok=True)
        wb = Workbook()
        default = wb.active
        wb.remove(default)
        for sheet, rows in data.items():
            ws = wb.create_sheet(sheet)
            ws.append(SHEETS[sheet])
            for row in rows:
                ws.append(row)
            ws.freeze_panes = "A2"
        wb.save(output)
        return output

    def _collect(self, run_id: int | None) -> dict[str, list[list[object]]]:
        return {
            PAPERS_SHEET: self._papers(),
            DATASETS_SHEET: self._datasets(),
            BIBTEX_SHEET: self._bibtex(),
            REVIEW_SHEET: self._review(run_id),
            EVIDENCE_SHEET: self._evidence(run_id),
            PAPER_DATASET_LINKS_SHEET: self._paper_dataset_links(),
        }

    def _papers(self) -> list[list[object]]:
        rows = self.conn.execute(
            """
            SELECT p.title, p.authors, v.name AS venue, ci.year, p.bibtex_key,
                   GROUP_CONCAT(d.canonical_name, ', ') AS datasets
            FROM papers p
            JOIN conference_instances ci ON ci.id = p.conference_instance_id
            JOIN venues v ON v.id = ci.venue_id
            LEFT JOIN paper_dataset_links pdl ON pdl.paper_id = p.id AND pdl.review_needed = 0
            LEFT JOIN datasets d ON d.id = pdl.dataset_id
            GROUP BY p.id
            ORDER BY ci.year, p.title
            """
        )
        return [[r["title"], r["authors"], r["venue"], r["year"], r["datasets"] or "", r["bibtex_key"] or ""] for r in rows]

    def _datasets(self) -> list[list[object]]:
        rows = self.conn.execute(
            """
            SELECT d.canonical_name, p.bibtex_key, d.availability_status,
                   GROUP_CONCAT(padc.modalities_json, '||') AS modalities_json,
                   GROUP_CONCAT(padc.osi_layers_json, '||') AS osi_layers_json,
                   COUNT(DISTINCT pdl.paper_id) AS use_count
            FROM datasets d
            LEFT JOIN papers p ON p.id = d.source_paper_id
            LEFT JOIN paper_dataset_links pdl ON pdl.dataset_id = d.id
            LEFT JOIN paper_analysis_dataset_claims padc ON padc.dataset_id = d.id
            GROUP BY d.id
            ORDER BY d.canonical_name
            """
        )
        return [
            [
                r["canonical_name"],
                r["bibtex_key"] or "",
                ", ".join(_json_list_union(r["osi_layers_json"])),
                ", ".join(_json_list_union(r["modalities_json"])),
                _open_yes_no(r["availability_status"]),
                r["availability_status"] or "",
                "",
                r["use_count"],
            ]
            for r in rows
        ]

    def _bibtex(self) -> list[list[object]]:
        rows = self.conn.execute("SELECT citation_key, doi, bibtex FROM bibtex_entries ORDER BY citation_key")
        return [[r["citation_key"], r["doi"] or "", r["bibtex"]] for r in rows]

    def _review(self, run_id: int | None) -> list[list[object]]:
        query = "SELECT * FROM review_items WHERE status = 'pending'"
        params: tuple[object, ...] = ()
        if run_id is not None:
            run_ids = self._related_run_ids(run_id)
            query += f" AND run_id IN ({','.join('?' for _ in run_ids)})"
            params = tuple(run_ids)
        rows = self.conn.execute(query + " ORDER BY id", params)
        return [[r["item_type"], r["paper_title"] or "", r["dataset_name"] or "", r["field"], r["suggested_value"] or "", r["confidence"], r["review_reason"], r["evidence"] or "", r["source_url"] or ""] for r in rows]

    def _evidence(self, run_id: int | None) -> list[list[object]]:
        query = """
            SELECT ec.*, p.title AS paper_title, d.canonical_name AS dataset_name
            FROM evidence_claims ec
            LEFT JOIN papers p ON p.id = ec.paper_id
            LEFT JOIN datasets d ON d.id = ec.dataset_id
        """
        params: tuple[object, ...] = ()
        if run_id is not None:
            run_ids = self._related_run_ids(run_id)
            query += f" WHERE ec.run_id IN ({','.join('?' for _ in run_ids)})"
            params = tuple(run_ids)
        rows = self.conn.execute(query + " ORDER BY ec.id", params)
        return [[r["claim_id"], r["paper_title"] or "", r["dataset_name"] or "", r["claim_type"], r["claim_value"], r["evidence_text"] or "", "", r["source_url"] or "", r["created_at"], r["confidence"]] for r in rows]

    def _paper_dataset_links(self) -> list[list[object]]:
        rows = self.conn.execute(
            """
            SELECT p.title, p.bibtex_key, d.canonical_name, pdl.relationship_type,
                   pdl.confidence, pdl.evidence_text, pdl.review_needed
            FROM paper_dataset_links pdl
            JOIN papers p ON p.id = pdl.paper_id
            JOIN datasets d ON d.id = pdl.dataset_id
            ORDER BY p.title, d.canonical_name
            """
        )
        return [[r["title"], r["bibtex_key"] or "", r["canonical_name"], r["relationship_type"], r["confidence"], r["evidence_text"] or "", "Yes" if r["review_needed"] else "No"] for r in rows]

    def _related_run_ids(self, run_id: int) -> list[int]:
        run = self.conn.execute("SELECT conference_instance_id FROM pipeline_runs WHERE id = ?", (run_id,)).fetchone()
        if run is None or run["conference_instance_id"] is None:
            return [run_id]
        rows = self.conn.execute(
            "SELECT id FROM pipeline_runs WHERE conference_instance_id = ?",
            (run["conference_instance_id"],),
        )
        return [int(row["id"]) for row in rows] or [run_id]


def _slug(sheet: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", sheet.lower()).strip("_")


def _open_yes_no(status: str | None) -> str:
    if not status:
        return ""
    if status == "open_downloadable":
        return "Yes"
    if status in {"not_available", "broken_link", "request_access", "metadata_only", "code_only"}:
        return "No"
    return "Unclear"


def _json_list_union(value: str | None) -> list[str]:
    if not value:
        return []
    items: set[str] = set()
    for raw in value.split("||"):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            items.update(str(item) for item in parsed if str(item).strip())
    return sorted(items)
