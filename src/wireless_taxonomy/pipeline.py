from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from wireless_taxonomy.analyze.availability import AvailabilityChecker
from wireless_taxonomy.analyze.datasets import DatasetExtractor
from wireless_taxonomy.analyze.paper_text import PaperTextFetcher
from wireless_taxonomy.analyze.wireless import WirelessClassifier
from wireless_taxonomy.config import Settings
from wireless_taxonomy.db import connect, migrate, transaction
from wireless_taxonomy.evidence import EvidenceLogger
from wireless_taxonomy.export.spreadsheet import SpreadsheetExporter
from wireless_taxonomy.export.json_export import JsonExporter
from wireless_taxonomy.ingest.base import validate_paper_seeds
from wireless_taxonomy.ingest.bibtex import BibtexIngestAdapter
from wireless_taxonomy.ingest.csv import CsvIngestAdapter
from wireless_taxonomy.ingest.url import UrlIngestAdapter
from wireless_taxonomy.ingest.verify import PaperListVerifier
from wireless_taxonomy.models import EvidenceClaim, PaperSeed, new_id, utc_now
from wireless_taxonomy.resolve.datasets import DatasetResolver, normalize_dataset_name
from wireless_taxonomy.resolve.reuse import compute_reuse_counts
from wireless_taxonomy.review.queue import insert_review_item


class Pipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        migrate(settings.db_path)
        self.conn = connect(settings.db_path)

    def close(self) -> None:
        self.conn.close()

    def init_db(self) -> None:
        migrate(self.settings.db_path)

    def ingest(self, venue: str, year: int, source_type: str, source_value: str) -> int:
        ci_id = self._conference_instance_id(venue, year, source_value if source_type == "url" else None)
        run_id = self._create_run(ci_id, "ingest", source_type, source_value)
        logger = EvidenceLogger(self.settings.evidence_dir, run_id)
        adapter = self._adapter(venue, year, source_type, source_value)
        seeds = adapter.fetch()
        review_items = validate_paper_seeds(seeds, self.settings.thresholds.wireless_inclusion)
        with transaction(self.conn):
            for seed in seeds:
                paper_id = self._upsert_paper(ci_id, seed)
                self.conn.execute(
                    """
                    INSERT INTO paper_sources
                    (paper_id, run_id, source_url, source_method, evidence_text, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (paper_id, run_id, seed.source_url, seed.source_method, seed.evidence_text, seed.source_confidence),
                )
                self._insert_evidence(
                    run_id,
                    paper_id,
                    None,
                    "paper_seed",
                    seed.title,
                    seed.evidence_text,
                    seed.source_url,
                    seed.source_confidence,
                    {"source_method": seed.source_method},
                )
            for item in review_items:
                insert_review_item(self.conn, run_id, item)
            self._complete_run(run_id, f"Ingested {len(seeds)} papers; {len(review_items)} review items.")
        logger.event("ingest_completed", {"paper_count": len(seeds), "review_count": len(review_items)})
        return run_id

    def run(self, venue: str, year: int, source_type: str, source_value: str, out: str | None = None) -> int:
        run_id = self.ingest(venue, year, source_type, source_value)
        self.verify_paper_list(run_id, run_external=False, run_llm=False)
        self.classify_wireless(run_id)
        self.extract_datasets(run_id)
        self.check_availability(run_id)
        self.resolve_reuse(run_id)
        if out:
            self.export(run_id, out, "xlsx")
        return run_id

    def classify_wireless(self, run_id: int) -> int:
        source_run = self._require_run(run_id)
        stage_run_id = self._create_run(source_run["conference_instance_id"], "classify-wireless", "run", str(run_id))
        logger = EvidenceLogger(self.settings.evidence_dir, stage_run_id)
        classifier = WirelessClassifier()
        rows = self.conn.execute(
            "SELECT * FROM papers WHERE conference_instance_id = ?", (source_run["conference_instance_id"],)
        ).fetchall()
        review_count = 0
        with transaction(self.conn):
            for paper in rows:
                result = classifier.classify(paper["id"], paper["title"], paper["abstract"])
                is_wireless = 1 if result.label == "yes" and result.confidence >= self.settings.thresholds.wireless_inclusion else 0 if result.label == "no" else None
                self.conn.execute(
                    """
                    INSERT INTO paper_classifications
                    (paper_id, run_id, is_wireless, label, confidence, evidence, model_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (paper["id"], stage_run_id, is_wireless, result.label, result.confidence, result.evidence, result.model_version),
                )
                self._insert_evidence(stage_run_id, paper["id"], None, "wireless_classification", result.label, result.evidence, None, result.confidence)
                if result.confidence < self.settings.thresholds.wireless_inclusion:
                    insert_review_item(
                        self.conn,
                        stage_run_id,
                        _review("paper", paper["title"], None, "Wireless Classification", result.label, result.confidence, "Wireless confidence below threshold", result.evidence, None),
                    )
                    review_count += 1
            self._complete_run(stage_run_id, f"Classified {len(rows)} papers; {review_count} review items.")
        logger.event("classify_wireless_completed", {"paper_count": len(rows), "review_count": review_count})
        return stage_run_id

    def verify_paper_list(self, run_id: int, run_external: bool = False, run_llm: bool = False) -> int:
        source_run = self._require_run(run_id)
        stage_run_id = self._create_run(source_run["conference_instance_id"], "verify-paper-list", "run", str(run_id))
        papers = [dict(row) for row in self.conn.execute(
            "SELECT * FROM papers WHERE conference_instance_id = ? ORDER BY id",
            (source_run["conference_instance_id"],),
        )]
        source_url = self._source_url_for_verification(source_run)
        verifier = PaperListVerifier(self.settings.llm)
        report = verifier.verify(papers, source_url, run_external=run_external, run_llm=run_llm)
        with transaction(self.conn):
            self.conn.execute(
                """
                INSERT INTO paper_list_verification_reports
                (run_id, conference_instance_id, paper_count, missing_authors_count,
                 missing_abstract_count, missing_doi_count, duplicate_title_count,
                 low_confidence_count, external_checked_count, external_mismatch_count,
                 llm_correction_count, final_confidence, report_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stage_run_id,
                    source_run["conference_instance_id"],
                    report.paper_count,
                    report.missing_authors_count,
                    report.missing_abstract_count,
                    report.missing_doi_count,
                    report.duplicate_title_count,
                    report.low_confidence_count,
                    report.external_checked_count,
                    report.external_mismatch_count,
                    report.llm_correction_count,
                    report.final_confidence,
                    json.dumps(report.to_dict(), ensure_ascii=False),
                ),
            )
            for issue in report.issues:
                insert_review_item(
                    self.conn,
                    stage_run_id,
                    _review(
                        "paper",
                        issue.paper_title,
                        None,
                        issue.field,
                        issue.suggested_value or "",
                        issue.confidence,
                        issue.message,
                        issue.evidence,
                        issue.source_url,
                    ),
                )
            self._insert_evidence(
                stage_run_id,
                None,
                None,
                "paper_list_verification",
                f"confidence={report.final_confidence}",
                json.dumps(report.to_dict(), ensure_ascii=False),
                source_url,
                report.final_confidence,
            )
            self._complete_run(
                stage_run_id,
                f"Verified {report.paper_count} papers; {len(report.issues)} review issues; confidence={report.final_confidence}.",
            )
        EvidenceLogger(self.settings.evidence_dir, stage_run_id).event("paper_list_verification_completed", report.to_dict())
        return stage_run_id

    def extract_datasets(self, run_id: int) -> int:
        source_run = self._require_run(run_id)
        stage_run_id = self._create_run(source_run["conference_instance_id"], "extract-datasets", "run", str(run_id))
        fetcher = PaperTextFetcher()
        extractor = DatasetExtractor()
        resolver = DatasetResolver(self.conn)
        rows = self.conn.execute(
            "SELECT * FROM papers WHERE conference_instance_id = ?", (source_run["conference_instance_id"],)
        ).fetchall()
        claim_count = 0
        review_count = 0
        with transaction(self.conn):
            for paper in rows:
                text = fetcher.fetch_text(paper["title"], paper["abstract"], paper["pdf_url"])
                for claim in extractor.extract(paper["id"], text, paper["paper_url"]):
                    decision = resolver.resolve(claim.dataset_name)
                    dataset_id = decision.canonical_dataset_id or self._create_dataset(claim.dataset_name, paper["id"])
                    review_needed = int(claim.confidence < self.settings.thresholds.dataset_use)
                    self.conn.execute(
                        """
                        INSERT OR IGNORE INTO paper_dataset_links
                        (paper_id, dataset_id, run_id, relationship_type, confidence, evidence_text, evidence_url, review_needed)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (paper["id"], dataset_id, stage_run_id, claim.relationship_type, claim.confidence, claim.evidence_text, claim.source_url, review_needed),
                    )
                    self._insert_evidence(stage_run_id, paper["id"], dataset_id, "dataset_claim", claim.dataset_name, claim.evidence_text, claim.source_url, claim.confidence)
                    claim_count += 1
                    if review_needed:
                        insert_review_item(
                            self.conn,
                            stage_run_id,
                            _review("dataset", paper["title"], claim.dataset_name, "Dataset Use", claim.dataset_name, claim.confidence, "Dataset extraction confidence below threshold", claim.evidence_text, claim.source_url),
                        )
                        review_count += 1
            self._complete_run(stage_run_id, f"Extracted {claim_count} dataset claims; {review_count} review items.")
        EvidenceLogger(self.settings.evidence_dir, stage_run_id).event("extract_datasets_completed", {"claim_count": claim_count})
        return stage_run_id

    def check_availability(self, run_id: int) -> int:
        source_run = self._require_run(run_id)
        stage_run_id = self._create_run(source_run["conference_instance_id"], "check-availability", "run", str(run_id))
        checker = AvailabilityChecker()
        rows = self.conn.execute(
            """
            SELECT dl.dataset_id, dl.url
            FROM dataset_links dl
            JOIN datasets d ON d.id = dl.dataset_id
            ORDER BY dl.id
            """
        ).fetchall()
        with transaction(self.conn):
            for row in rows:
                claim = checker.check(row["url"], row["dataset_id"])
                self.conn.execute(
                    """
                    INSERT INTO availability_checks
                    (dataset_id, run_id, url, availability_status, confidence, evidence_text, checked_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (row["dataset_id"], stage_run_id, claim.url, claim.availability_status, claim.confidence, claim.evidence_text, claim.checked_at),
                )
                self._insert_evidence(stage_run_id, None, row["dataset_id"], "availability", claim.availability_status, claim.evidence_text, claim.url, claim.confidence)
            self._complete_run(stage_run_id, f"Checked {len(rows)} dataset availability links.")
        EvidenceLogger(self.settings.evidence_dir, stage_run_id).event("availability_completed", {"link_count": len(rows)})
        return stage_run_id

    def resolve_reuse(self, run_id: int) -> int:
        source_run = self._require_run(run_id)
        stage_run_id = self._create_run(source_run["conference_instance_id"], "resolve-reuse", "run", str(run_id))
        counts = compute_reuse_counts(self.conn)
        with transaction(self.conn):
            self._complete_run(stage_run_id, f"Computed reuse counts for {len(counts)} datasets.")
        EvidenceLogger(self.settings.evidence_dir, stage_run_id).event("reuse_completed", {"reuse_counts": counts})
        return stage_run_id

    def export(self, run_id: int | None, out: str, fmt: str) -> Path:
        if fmt == "json":
            return JsonExporter(self.conn).export(run_id, out)
        return SpreadsheetExporter(self.conn).export(run_id, out, fmt)

    def status(self, run_id: int | None = None) -> list[sqlite3.Row]:
        if run_id is None:
            return list(self.conn.execute("SELECT * FROM pipeline_runs ORDER BY id"))
        return list(self.conn.execute("SELECT * FROM pipeline_runs WHERE id = ?", (run_id,)))

    def latest_paper_list_report(self, run_id: int | None = None) -> sqlite3.Row | None:
        params: tuple[object, ...] = ()
        query = "SELECT * FROM paper_list_verification_reports"
        if run_id is not None:
            run = self._require_run(run_id)
            query += " WHERE conference_instance_id = ?"
            params = (run["conference_instance_id"],)
        return self.conn.execute(query + " ORDER BY id DESC LIMIT 1", params).fetchone()

    def _adapter(self, venue: str, year: int, source_type: str, source_value: str):
        if source_type == "url":
            return UrlIngestAdapter(venue, year, source_value, self.settings.llm)
        if source_type == "bibtex":
            return BibtexIngestAdapter(venue, year, source_value)
        if source_type == "csv":
            return CsvIngestAdapter(venue, year, source_value)
        raise ValueError("source_type must be url, bibtex, or csv")

    def _conference_instance_id(self, venue: str, year: int, source_url: str | None = None) -> int:
        with transaction(self.conn):
            self.conn.execute("INSERT OR IGNORE INTO venues(name) VALUES (?)", (venue,))
            venue_id = self.conn.execute("SELECT id FROM venues WHERE name = ?", (venue,)).fetchone()["id"]
            self.conn.execute(
                "INSERT OR IGNORE INTO conference_instances(venue_id, year, official_url, proceedings_url) VALUES (?, ?, ?, ?)",
                (venue_id, year, source_url, source_url),
            )
            return self.conn.execute("SELECT id FROM conference_instances WHERE venue_id = ? AND year = ?", (venue_id, year)).fetchone()["id"]

    def _create_run(self, conference_instance_id: int | None, stage: str, source_type: str | None, source_value: str | None) -> int:
        with transaction(self.conn):
            cur = self.conn.execute(
                "INSERT INTO pipeline_runs(conference_instance_id, stage, status, source_type, source_value) VALUES (?, ?, 'running', ?, ?)",
                (conference_instance_id, stage, source_type, source_value),
            )
            return int(cur.lastrowid)

    def _complete_run(self, run_id: int, message: str) -> None:
        self.conn.execute(
            "UPDATE pipeline_runs SET status = 'completed', completed_at = ?, message = ? WHERE id = ?",
            (utc_now(), message, run_id),
        )

    def _require_run(self, run_id: int) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM pipeline_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise ValueError(f"Run {run_id} not found")
        return row

    def _source_url_for_verification(self, run: sqlite3.Row) -> str | None:
        if run["source_type"] == "url":
            return run["source_value"]
        row = self.conn.execute(
            """
            SELECT source_value FROM pipeline_runs
            WHERE conference_instance_id = ? AND source_type = 'url'
            ORDER BY id LIMIT 1
            """,
            (run["conference_instance_id"],),
        ).fetchone()
        return row["source_value"] if row else None

    def _upsert_paper(self, conference_instance_id: int, seed: PaperSeed) -> int:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO papers
            (conference_instance_id, title, authors, doi, abstract, paper_url, pdf_url, session, source_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (conference_instance_id, seed.title, ", ".join(seed.authors), seed.doi, seed.abstract, seed.paper_url, seed.pdf_url, seed.session, seed.source_confidence),
        )
        return self.conn.execute(
            "SELECT id FROM papers WHERE conference_instance_id = ? AND title = ?",
            (conference_instance_id, seed.title),
        ).fetchone()["id"]

    def _create_dataset(self, name: str, source_paper_id: int) -> int:
        normalized = normalize_dataset_name(name)
        self.conn.execute(
            "INSERT OR IGNORE INTO datasets(canonical_name, normalized_name, source_paper_id) VALUES (?, ?, ?)",
            (name, normalized, source_paper_id),
        )
        return self.conn.execute("SELECT id FROM datasets WHERE normalized_name = ?", (normalized,)).fetchone()["id"]

    def _insert_evidence(
        self,
        run_id: int,
        paper_id: int | None,
        dataset_id: int | None,
        claim_type: str,
        claim_value: str,
        evidence_text: str | None,
        source_url: str | None,
        confidence: float,
        payload: dict | None = None,
    ) -> None:
        claim = EvidenceClaim(new_id("claim"), run_id, claim_type, claim_value, evidence_text, source_url, confidence, payload=payload or {})
        self.conn.execute(
            """
            INSERT INTO evidence_claims
            (claim_id, run_id, paper_id, dataset_id, claim_type, claim_value, evidence_text,
             source_url, confidence, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (claim.claim_id, run_id, paper_id, dataset_id, claim_type, claim_value, evidence_text, source_url, confidence, json.dumps(payload or {})),
        )


def _review(item_type: str, paper_title: str | None, dataset_name: str | None, field: str, suggested: str, confidence: float, reason: str, evidence: str | None, source_url: str | None):
    from wireless_taxonomy.models import ReviewItem

    return ReviewItem(item_type, field, suggested, confidence, reason, paper_title, dataset_name, evidence, source_url)
