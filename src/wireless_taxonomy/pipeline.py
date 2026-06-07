from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from wireless_taxonomy.analyze.agentic_paper import DeterministicPaperAnalyzer, LlmPaperAnalyzer
from wireless_taxonomy.analyze.acm_browser import AuthenticatedAcmBrowserFetcher
from wireless_taxonomy.analyze.availability import AvailabilityChecker
from wireless_taxonomy.analyze.datasets import DatasetExtractor
from wireless_taxonomy.analyze.full_text import FullTextDiscoverer
from wireless_taxonomy.analyze.local_pdfs import LocalPdfImporter
from wireless_taxonomy.analyze.paper_text import PaperTextFetcher
from wireless_taxonomy.analyze.readiness import PaperInputReadinessAssessor
from wireless_taxonomy.analyze.reflection import DeterministicAnalysisReflector
from wireless_taxonomy.analyze.scope import ScopeAssessor
from wireless_taxonomy.analyze.wireless import WirelessClassifier
from wireless_taxonomy.config import Settings
from wireless_taxonomy.db import connect, migrate, transaction
from wireless_taxonomy.evidence import EvidenceLogger
from wireless_taxonomy.export.spreadsheet import SpreadsheetExporter
from wireless_taxonomy.export.json_export import JsonExporter
from wireless_taxonomy.export.paper_set import PaperSetExporter
from wireless_taxonomy.evaluate.jaccard import (
    JaccardAggregate,
    JaccardReport,
    compute_paper_list_jaccard,
    compute_paper_list_jaccard_all,
    list_conference_runs,
    write_jaccard_aggregate,
    write_jaccard_report,
)
from wireless_taxonomy.ingest.base import validate_paper_seeds
from wireless_taxonomy.ingest.bibtex import BibtexIngestAdapter
from wireless_taxonomy.ingest.csv import CsvIngestAdapter
from wireless_taxonomy.ingest.url import UrlIngestAdapter
from wireless_taxonomy.ingest.verify import PaperListVerifier
from wireless_taxonomy.models import EvidenceClaim, PaperSeed, PaperTextArtifact, PaperTextEnrichment, new_id, utc_now
from wireless_taxonomy.resolve.datasets import DatasetResolver, normalize_dataset_name
from wireless_taxonomy.resolve.reuse import compute_reuse_counts
from wireless_taxonomy.resolve.cache import SqliteResolverCache
from wireless_taxonomy.review.queue import insert_review_item


@dataclass(frozen=True)
class TextPersistenceResult:
    artifact_count: int
    link_count: int
    snippet_count: int
    failed_artifacts: list[PaperTextArtifact]
    fetched_artifacts: list[PaperTextArtifact]


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

    def run(self, venue: str, year: int, source_type: str, source_value: str, out: str | None = None, fmt: str = "csv", use_llm: bool = False) -> int:
        run_id = self.ingest(venue, year, source_type, source_value)
        self.assess_scope(run_id)
        self.verify_paper_list(run_id, run_external=False, run_llm=False)
        self.enrich_paper_text(run_id)
        self.discover_full_text(run_id)
        self.assess_paper_inputs(run_id)
        analysis_run_id = self.agentic_paper_analysis(run_id, use_llm=use_llm)
        self.reflect_paper_analysis(run_id, analysis_run_id=analysis_run_id)
        self.classify_wireless(run_id)
        self.extract_datasets(run_id)
        self.check_availability(run_id)
        self.resolve_reuse(run_id)
        if out:
            self.export(run_id, out, fmt)
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

    def assess_scope(self, run_id: int) -> int:
        source_run = self._require_run(run_id)
        stage_run_id = self._create_run(source_run["conference_instance_id"], "assess-scope", "run", str(run_id))
        papers = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM papers WHERE conference_instance_id = ? ORDER BY id",
                (source_run["conference_instance_id"],),
            )
        ]
        assessor = ScopeAssessor()
        assessment = assessor.assess(papers)
        with transaction(self.conn):
            self.conn.execute(
                """
                INSERT INTO scope_assessments
                (run_id, conference_instance_id, paper_count, networking_like_count,
                 wireless_like_count, malformed_count, networking_like_ratio,
                 wireless_like_ratio, should_proceed, decision, confidence, report_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stage_run_id,
                    source_run["conference_instance_id"],
                    assessment.paper_count,
                    assessment.networking_like_count,
                    assessment.wireless_like_count,
                    assessment.malformed_count,
                    assessment.networking_like_ratio,
                    assessment.wireless_like_ratio,
                    1 if assessment.should_proceed else 0,
                    assessment.decision,
                    assessment.confidence,
                    json.dumps(assessment.to_dict(), ensure_ascii=False),
                ),
            )
            for issue in assessment.issues:
                insert_review_item(
                    self.conn,
                    stage_run_id,
                    _review(
                        "paper" if issue.paper_id else "source",
                        issue.paper_title,
                        None,
                        issue.field,
                        assessment.decision,
                        issue.confidence,
                        issue.message,
                        issue.evidence,
                        source_run["source_value"] if source_run["source_type"] == "url" else None,
                    ),
                )
            self._insert_evidence(
                stage_run_id,
                None,
                None,
                "scope_assessment",
                assessment.decision,
                json.dumps(assessment.to_dict(), ensure_ascii=False),
                source_run["source_value"] if source_run["source_type"] == "url" else None,
                assessment.confidence,
            )
            self._complete_run(
                stage_run_id,
                f"Assessed {assessment.paper_count} papers; decision={assessment.decision}; "
                f"networking={assessment.networking_like_ratio}; wireless={assessment.wireless_like_ratio}; "
                f"malformed={assessment.malformed_count}.",
            )
        EvidenceLogger(self.settings.evidence_dir, stage_run_id).event("scope_assessment_completed", assessment.to_dict())
        return stage_run_id

    def enrich_paper_text(self, run_id: int) -> int:
        source_run = self._require_run(run_id)
        stage_run_id = self._create_run(source_run["conference_instance_id"], "enrich-paper-text", "run", str(run_id))
        fetcher = PaperTextFetcher(allow_remote=self.settings.enable_web_search)
        rows = self.conn.execute(
            "SELECT * FROM papers WHERE conference_instance_id = ? ORDER BY id",
            (source_run["conference_instance_id"],),
        ).fetchall()
        artifact_count = 0
        link_count = 0
        snippet_count = 0
        review_count = 0
        with transaction(self.conn):
            for paper in rows:
                enrichment = fetcher.enrich(paper)
                persisted = self._persist_paper_text_enrichment(
                    stage_run_id,
                    enrichment,
                    provider_name=fetcher.provider_name,
                    artifact_claim_type="paper_text_artifact",
                    snippet_claim_type="paper_text_snippet",
                    artifact_success_confidence=0.85,
                    artifact_failure_confidence=0.45,
                    artifact_success_statuses={"available", "fetched", "reference_only"},
                )
                artifact_count += persisted.artifact_count
                link_count += persisted.link_count
                snippet_count += persisted.snippet_count
                for artifact in enrichment.artifacts:
                    if artifact.fetch_status == "error":
                        insert_review_item(
                            self.conn,
                            stage_run_id,
                            _review(
                                "paper",
                                paper["title"],
                                None,
                                "Paper Text",
                                artifact.source_url or "",
                                0.40,
                                "Paper text source could not be fetched",
                                artifact.error_message,
                                artifact.source_url,
                            ),
                        )
                        review_count += 1
            self._complete_run(
                stage_run_id,
                f"Enriched {len(rows)} papers; {artifact_count} artifacts; {link_count} links; {snippet_count} snippets; {review_count} review items.",
            )
        EvidenceLogger(self.settings.evidence_dir, stage_run_id).event(
            "paper_text_enrichment_completed",
            {"paper_count": len(rows), "artifact_count": artifact_count, "link_count": link_count, "snippet_count": snippet_count},
        )
        return stage_run_id

    def discover_full_text(self, run_id: int, paper_id: int | None = None) -> int:
        source_run = self._require_run(run_id)
        stage_run_id = self._create_run(source_run["conference_instance_id"], "discover-full-text", "run", str(run_id))
        discoverer = FullTextDiscoverer(allow_remote=self.settings.enable_web_search, candidate_cache=SqliteResolverCache(self.conn))
        params: tuple[object, ...] = (source_run["conference_instance_id"],)
        query = "SELECT * FROM papers WHERE conference_instance_id = ?"
        if paper_id is not None:
            query += " AND id = ?"
            params = (source_run["conference_instance_id"], paper_id)
        rows = self.conn.execute(query + " ORDER BY id", params).fetchall()
        artifact_count = 0
        link_count = 0
        snippet_count = 0
        review_count = 0
        with transaction(self.conn):
            for paper in rows:
                enrichment = discoverer.discover(paper)
                persisted = self._persist_paper_text_enrichment(
                    stage_run_id,
                    enrichment,
                    provider_name=discoverer.provider_name,
                    artifact_claim_type="full_text_artifact",
                    snippet_claim_type="full_text_snippet",
                    artifact_success_confidence=0.90,
                    artifact_failure_confidence=0.45,
                )
                artifact_count += persisted.artifact_count
                link_count += persisted.link_count
                snippet_count += persisted.snippet_count
                full_text_artifacts = [
                    artifact
                    for artifact in persisted.fetched_artifacts
                    if artifact.source_type in {"pdf_text", "local_pdf_text"} and artifact.fetch_status == "fetched"
                ]
                pdf_links = [link for link in enrichment.links if link.link_type == "pdf"]
                if pdf_links and not full_text_artifacts:
                    summary = _full_text_failure_summary(persisted.failed_artifacts)
                    insert_review_item(
                        self.conn,
                        stage_run_id,
                        _review(
                            "paper",
                            paper["title"],
                            None,
                            "Full Text Discovery",
                            f"{len(pdf_links)} candidate PDF(s), 0 fetched",
                            0.40,
                            "No candidate PDF produced usable full text",
                            summary,
                            paper["paper_url"],
                        ),
                    )
                    review_count += 1
            self._complete_run(
                stage_run_id,
                f"Discovered full text for {len(rows)} papers; {artifact_count} artifacts; {link_count} links; {snippet_count} snippets; {review_count} review items.",
            )
        EvidenceLogger(self.settings.evidence_dir, stage_run_id).event(
            "full_text_discovery_completed",
            {"paper_count": len(rows), "artifact_count": artifact_count, "link_count": link_count, "snippet_count": snippet_count, "review_count": review_count},
        )
        return stage_run_id

    def add_pdfs(self, run_id: int, directory: str | Path) -> int:
        source_run = self._require_run(run_id)
        stage_run_id = self._create_run(source_run["conference_instance_id"], "add-pdfs", "directory", str(directory))
        rows = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM papers WHERE conference_instance_id = ? ORDER BY id",
                (source_run["conference_instance_id"],),
            )
        ]
        importer = LocalPdfImporter()
        result = importer.import_directory(rows, directory)
        artifact_count = 0
        link_count = 0
        snippet_count = 0
        review_count = 0
        title_by_id = {int(row["id"]): row["title"] for row in rows}
        with transaction(self.conn):
            for enrichment in result.enrichments:
                persisted = self._persist_paper_text_enrichment(
                    stage_run_id,
                    enrichment,
                    provider_name=importer.provider_name,
                    artifact_claim_type="local_pdf_artifact",
                    snippet_claim_type="local_pdf_snippet",
                    artifact_success_confidence=0.95,
                    artifact_failure_confidence=0.40,
                    default_snippets_to_first_artifact=False,
                )
                artifact_count += persisted.artifact_count
                link_count += persisted.link_count
                snippet_count += persisted.snippet_count
            for unmatched in result.unmatched:
                insert_review_item(
                    self.conn,
                    stage_run_id,
                    _review(
                        "paper",
                        None,
                        None,
                        "Local PDF Import",
                        str(unmatched.path),
                        0.30,
                        unmatched.reason,
                        None,
                        unmatched.path.resolve().as_uri(),
                    ),
                )
                review_count += 1
            self._complete_run(
                stage_run_id,
                f"Imported local PDFs; artifacts={artifact_count}; links={link_count}; snippets={snippet_count}; unmatched={review_count}.",
            )
        EvidenceLogger(self.settings.evidence_dir, stage_run_id).event(
            "local_pdf_import_completed",
            {
                "paper_count": len(title_by_id),
                "artifact_count": artifact_count,
                "link_count": link_count,
                "snippet_count": snippet_count,
                "unmatched_count": review_count,
            },
        )
        return stage_run_id

    def fetch_acm_browser(
        self,
        run_id: int,
        profile_dir: str | Path,
        paper_id: int | None = None,
        limit: int | None = None,
        headless: bool = False,
        browser_channel: str | None = None,
        cdp_url: str | None = None,
        delay_seconds: float | None = None,
        login_only: bool = False,
        login_url: str = "https://dl.acm.org/",
    ) -> int | None:
        source_run = self._require_run(run_id)
        fetcher = AuthenticatedAcmBrowserFetcher(
            profile_dir=profile_dir,
            headless=headless,
            browser_channel=browser_channel,
            cdp_url=cdp_url,
            delay_seconds=delay_seconds,
        )
        if login_only:
            fetcher.login(login_url)
            return None

        stage_run_id = self._create_run(source_run["conference_instance_id"], "fetch-acm-browser", "run", str(run_id))
        params: tuple[object, ...] = (source_run["conference_instance_id"],)
        query = "SELECT * FROM papers WHERE conference_instance_id = ? AND (doi LIKE '10.1145/%' OR paper_url LIKE '%dl.acm.org/doi/%')"
        if paper_id is not None:
            query += " AND id = ?"
            params = (source_run["conference_instance_id"], paper_id)
        rows = [dict(row) for row in self.conn.execute(query + " ORDER BY id", params).fetchall()]
        enrichments = fetcher.fetch_many(rows, limit=limit)
        artifact_count = 0
        link_count = 0
        snippet_count = 0
        review_count = 0
        title_by_id = {int(row["id"]): row["title"] for row in rows}
        with transaction(self.conn):
            for enrichment in enrichments:
                persisted = self._persist_paper_text_enrichment(
                    stage_run_id,
                    enrichment,
                    provider_name=fetcher.provider_name,
                    artifact_claim_type="acm_browser_artifact",
                    snippet_claim_type="acm_browser_snippet",
                    artifact_success_confidence=0.95,
                    artifact_failure_confidence=0.35,
                )
                artifact_count += persisted.artifact_count
                link_count += persisted.link_count
                snippet_count += persisted.snippet_count
                if not persisted.fetched_artifacts:
                    artifact = enrichment.artifacts[0] if enrichment.artifacts else None
                    insert_review_item(
                        self.conn,
                        stage_run_id,
                        _review(
                            "paper",
                            title_by_id.get(enrichment.paper_id),
                            None,
                            "ACM Browser Fetch",
                            "0 fetched",
                            0.35,
                            "Authenticated ACM browser fetch did not produce usable full text",
                            artifact.error_message if artifact else "No artifact recorded",
                            artifact.source_url if artifact else None,
                        ),
                    )
                    review_count += 1
            self._complete_run(
                stage_run_id,
                f"Fetched ACM browser full text for {len(enrichments)} papers; {artifact_count} artifacts; {link_count} links; {snippet_count} snippets; {review_count} review items.",
            )
        EvidenceLogger(self.settings.evidence_dir, stage_run_id).event(
            "acm_browser_fetch_completed",
            {"paper_count": len(enrichments), "artifact_count": artifact_count, "link_count": link_count, "snippet_count": snippet_count, "review_count": review_count},
        )
        return stage_run_id

    def assess_paper_inputs(self, run_id: int) -> int:
        source_run = self._require_run(run_id)
        stage_run_id = self._create_run(source_run["conference_instance_id"], "assess-paper-inputs", "run", str(run_id))
        assessor = PaperInputReadinessAssessor()
        text_run_ids = [
            run_id
            for run_id in [
                self._latest_stage_run_id(source_run["conference_instance_id"], "enrich-paper-text"),
                self._latest_stage_run_id(source_run["conference_instance_id"], "discover-full-text"),
            ]
            if run_id is not None
        ]
        papers = [dict(row) for row in self.conn.execute(
            "SELECT * FROM papers WHERE conference_instance_id = ? ORDER BY id",
            (source_run["conference_instance_id"],),
        )]
        ready_count = 0
        review_count = 0
        level_counts: dict[str, int] = {}
        with transaction(self.conn):
            for paper in papers:
                artifacts = [dict(row) for row in self.conn.execute(
                    """
                    SELECT * FROM paper_text_artifacts
                    WHERE paper_id = ? AND (? = 0 OR run_id IN (%s))
                    ORDER BY id
                    """ % _placeholders(text_run_ids),
                    (paper["id"], len(text_run_ids), *text_run_ids),
                )]
                links = [dict(row) for row in self.conn.execute(
                    """
                    SELECT * FROM paper_text_links
                    WHERE paper_id = ? AND (? = 0 OR run_id IN (%s))
                    ORDER BY id
                    """ % _placeholders(text_run_ids),
                    (paper["id"], len(text_run_ids), *text_run_ids),
                )]
                snippets = [dict(row) for row in self.conn.execute(
                    """
                    SELECT * FROM paper_text_snippets
                    WHERE paper_id = ? AND (? = 0 OR run_id IN (%s))
                    ORDER BY id
                    """ % _placeholders(text_run_ids),
                    (paper["id"], len(text_run_ids), *text_run_ids),
                )]
                readiness = assessor.assess(paper, artifacts, links, snippets)
                payload = readiness.to_dict()
                level_counts[readiness.readiness_level] = level_counts.get(readiness.readiness_level, 0) + 1
                ready_count += int(readiness.should_analyze)
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO paper_input_readiness
                    (paper_id, run_id, has_abstract, has_fetched_text, has_pdf_link,
                     has_artifact_link, snippet_count, usable_text_chars, readiness_level,
                     should_analyze, limitations_json, report_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper["id"],
                        stage_run_id,
                        int(readiness.has_abstract),
                        int(readiness.has_fetched_text),
                        int(readiness.has_pdf_link),
                        int(readiness.has_artifact_link),
                        readiness.snippet_count,
                        readiness.usable_text_chars,
                        readiness.readiness_level,
                        int(readiness.should_analyze),
                        json.dumps(readiness.limitations, ensure_ascii=False),
                        json.dumps(payload, ensure_ascii=False),
                    ),
                )
                self._insert_evidence(
                    stage_run_id,
                    paper["id"],
                    None,
                    "paper_input_readiness",
                    readiness.readiness_level,
                    json.dumps(payload, ensure_ascii=False),
                    paper.get("paper_url"),
                    0.90 if readiness.should_analyze else 0.40,
                    {"provider": assessor.provider_name},
                )
                if not readiness.should_analyze:
                    insert_review_item(
                        self.conn,
                        stage_run_id,
                        _review(
                            "paper",
                            paper["title"],
                            None,
                            "Paper Input Readiness",
                            readiness.readiness_level,
                            0.40,
                            "Paper lacks usable text inputs for analysis",
                            "; ".join(readiness.limitations),
                            paper.get("paper_url"),
                        ),
                    )
                    review_count += 1
            self._complete_run(
                stage_run_id,
                f"Assessed inputs for {len(papers)} papers; ready={ready_count}; levels={json.dumps(level_counts, sort_keys=True)}; review items={review_count}.",
            )
        EvidenceLogger(self.settings.evidence_dir, stage_run_id).event(
            "paper_input_readiness_completed",
            {"paper_count": len(papers), "ready_count": ready_count, "level_counts": level_counts, "review_count": review_count},
        )
        return stage_run_id

    def agentic_paper_analysis(self, run_id: int, paper_id: int | None = None, use_llm: bool = False) -> int:
        source_run = self._require_run(run_id)
        stage_run_id = self._create_run(source_run["conference_instance_id"], "agentic-paper-analysis", "run", str(run_id))
        analyzer = LlmPaperAnalyzer(self.settings.llm) if use_llm else DeterministicPaperAnalyzer()
        resolver = DatasetResolver(self.conn)
        params: tuple[object, ...] = (source_run["conference_instance_id"],)
        query = "SELECT * FROM papers WHERE conference_instance_id = ?"
        if paper_id is not None:
            query += " AND id = ?"
            params = (source_run["conference_instance_id"], paper_id)
        rows = [dict(row) for row in self.conn.execute(query + " ORDER BY id", params)]
        analysis_count = 0
        dataset_claim_count = 0
        review_count = 0
        with transaction(self.conn):
            for paper in rows:
                text = self._paper_text_for_analysis(paper["id"])
                analysis = analyzer.analyze(paper, text)
                analysis_payload = analysis.to_dict()
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO paper_agentic_analyses
                    (paper_id, run_id, provider_name, wireless_label, is_wireless,
                     wireless_confidence, wireless_evidence, modalities_json,
                     osi_layers_json, summary, analysis_json, review_needed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper["id"],
                        stage_run_id,
                        analysis.provider_name,
                        analysis.wireless_label,
                        None if analysis.is_wireless is None else int(analysis.is_wireless),
                        analysis.wireless_confidence,
                        analysis.wireless_evidence,
                        json.dumps(analysis.modalities, ensure_ascii=False),
                        json.dumps(analysis.osi_layers, ensure_ascii=False),
                        analysis.summary,
                        json.dumps(analysis_payload, ensure_ascii=False),
                        int(analysis.review_needed),
                    ),
                )
                self.conn.execute(
                    """
                    INSERT INTO paper_classifications
                    (paper_id, run_id, is_wireless, label, confidence, evidence, model_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper["id"],
                        stage_run_id,
                        None if analysis.is_wireless is None else int(analysis.is_wireless),
                        analysis.wireless_label,
                        analysis.wireless_confidence,
                        analysis.wireless_evidence,
                        analysis.provider_name,
                    ),
                )
                self._insert_evidence(
                    stage_run_id,
                    paper["id"],
                    None,
                    "agentic_wireless_classification",
                    analysis.wireless_label,
                    analysis.wireless_evidence,
                    paper.get("paper_url"),
                    analysis.wireless_confidence,
                    {"provider": analysis.provider_name},
                )
                if analysis.modalities:
                    self._insert_evidence(
                        stage_run_id,
                        paper["id"],
                        None,
                        "modality",
                        ", ".join(analysis.modalities),
                        analysis.summary,
                        paper.get("paper_url"),
                        0.90,
                    )
                if analysis.osi_layers:
                    self._insert_evidence(
                        stage_run_id,
                        paper["id"],
                        None,
                        "osi_layer",
                        ", ".join(analysis.osi_layers),
                        analysis.summary,
                        paper.get("paper_url"),
                        0.90,
                    )
                if analysis.review_needed:
                    insert_review_item(
                        self.conn,
                        stage_run_id,
                        _review(
                            "paper",
                            paper["title"],
                            None,
                            "Agentic Paper Analysis",
                            analysis.summary,
                            min(analysis.wireless_confidence, 0.80),
                            "Paper analysis confidence or evidence below threshold",
                            analysis.summary,
                            paper.get("paper_url"),
                        ),
                    )
                    review_count += 1
                for claim in analysis.dataset_claims:
                    decision = resolver.resolve(claim.dataset_name)
                    dataset_id = decision.canonical_dataset_id or self._create_dataset(claim.dataset_name, paper["id"])
                    review_needed = int(claim.confidence < self.settings.thresholds.dataset_use or not claim.modalities or not claim.osi_layers)
                    self.conn.execute(
                        """
                        INSERT OR IGNORE INTO paper_analysis_dataset_claims
                        (paper_id, dataset_id, run_id, dataset_name, relationship_type,
                         confidence, modalities_json, osi_layers_json, evidence_text,
                         source_url, availability_status, review_needed)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            paper["id"],
                            dataset_id,
                            stage_run_id,
                            claim.dataset_name,
                            claim.relationship_type,
                            claim.confidence,
                            json.dumps(claim.modalities, ensure_ascii=False),
                            json.dumps(claim.osi_layers, ensure_ascii=False),
                            claim.evidence_text,
                            claim.source_url,
                            claim.availability_status,
                            review_needed,
                        ),
                    )
                    self.conn.execute(
                        """
                        INSERT OR IGNORE INTO paper_dataset_links
                        (paper_id, dataset_id, run_id, relationship_type, confidence, evidence_text, evidence_url, review_needed)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (paper["id"], dataset_id, stage_run_id, claim.relationship_type, claim.confidence, claim.evidence_text, claim.source_url, review_needed),
                    )
                    self._insert_evidence(
                        stage_run_id,
                        paper["id"],
                        dataset_id,
                        "agentic_dataset_claim",
                        claim.dataset_name,
                        claim.evidence_text,
                        claim.source_url,
                        claim.confidence,
                        {"modalities": claim.modalities, "osi_layers": claim.osi_layers, "provider": analysis.provider_name},
                    )
                    dataset_claim_count += 1
                    if review_needed:
                        insert_review_item(
                            self.conn,
                            stage_run_id,
                            _review(
                                "dataset",
                                paper["title"],
                                claim.dataset_name,
                                "Dataset Claim",
                                claim.dataset_name,
                                claim.confidence,
                                "Dataset claim, modality, or OSI evidence below threshold",
                                claim.evidence_text,
                                claim.source_url,
                            ),
                        )
                        review_count += 1
                analysis_count += 1
            self._complete_run(
                stage_run_id,
                f"Analyzed {analysis_count} papers; {dataset_claim_count} dataset claims; {review_count} review items.",
            )
        EvidenceLogger(self.settings.evidence_dir, stage_run_id).event(
            "agentic_paper_analysis_completed",
            {"paper_count": analysis_count, "dataset_claim_count": dataset_claim_count, "review_count": review_count},
        )
        return stage_run_id

    def reflect_paper_analysis(self, run_id: int, analysis_run_id: int | None = None, paper_id: int | None = None) -> int:
        source_run = self._require_run(run_id)
        if analysis_run_id is None:
            analysis_run_id = self._latest_stage_run_id(source_run["conference_instance_id"], "agentic-paper-analysis")
        if analysis_run_id is None:
            raise ValueError("No completed agentic-paper-analysis run found to reflect")
        stage_run_id = self._create_run(source_run["conference_instance_id"], "reflect-paper-analysis", "run", str(analysis_run_id))
        reflector = DeterministicAnalysisReflector()
        params: tuple[object, ...] = (analysis_run_id,)
        query = """
            SELECT paa.*, p.title, p.authors, p.doi, p.abstract, p.paper_url
            FROM paper_agentic_analyses paa
            JOIN papers p ON p.id = paa.paper_id
            WHERE paa.run_id = ?
        """
        if paper_id is not None:
            query += " AND paa.paper_id = ?"
            params = (analysis_run_id, paper_id)
        rows = [dict(row) for row in self.conn.execute(query + " ORDER BY paa.paper_id", params)]
        reflected_count = 0
        review_count = 0
        with transaction(self.conn):
            for analysis in rows:
                claims = [
                    dict(row)
                    for row in self.conn.execute(
                        "SELECT * FROM paper_analysis_dataset_claims WHERE run_id = ? AND paper_id = ? ORDER BY id",
                        (analysis_run_id, analysis["paper_id"]),
                    )
                ]
                text = self._paper_text_for_analysis(analysis["paper_id"])
                reflection = reflector.reflect(analysis, analysis, claims, text)
                payload = reflection.to_dict()
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO paper_analysis_reflections
                    (paper_id, run_id, analysis_run_id, decision, confidence, issues_json, reflection_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        reflection.paper_id,
                        stage_run_id,
                        analysis_run_id,
                        reflection.decision,
                        reflection.confidence,
                        json.dumps([issue.to_dict() for issue in reflection.issues], ensure_ascii=False),
                        json.dumps(payload, ensure_ascii=False),
                    ),
                )
                self._insert_evidence(
                    stage_run_id,
                    reflection.paper_id,
                    None,
                    "analysis_reflection",
                    reflection.decision,
                    "; ".join(issue.reason for issue in reflection.issues) or "No reflection issues found",
                    analysis.get("paper_url"),
                    reflection.confidence,
                    {"provider": reflector.provider_name, "analysis_run_id": analysis_run_id},
                )
                if reflection.issues:
                    self.conn.execute(
                        "UPDATE paper_agentic_analyses SET review_needed = 1 WHERE run_id = ? AND paper_id = ?",
                        (analysis_run_id, reflection.paper_id),
                    )
                    dataset_names = {issue.dataset_name for issue in reflection.issues if issue.dataset_name}
                    for dataset_name in dataset_names:
                        self.conn.execute(
                            """
                            UPDATE paper_analysis_dataset_claims
                            SET review_needed = 1
                            WHERE run_id = ? AND paper_id = ? AND dataset_name = ?
                            """,
                            (analysis_run_id, reflection.paper_id, dataset_name),
                        )
                        self.conn.execute(
                            """
                            UPDATE paper_dataset_links
                            SET review_needed = 1
                            WHERE run_id = ? AND paper_id = ?
                              AND dataset_id IN (
                                SELECT dataset_id FROM paper_analysis_dataset_claims
                                WHERE run_id = ? AND paper_id = ? AND dataset_name = ?
                              )
                            """,
                            (analysis_run_id, reflection.paper_id, analysis_run_id, reflection.paper_id, dataset_name),
                        )
                    for issue in reflection.issues:
                        insert_review_item(
                            self.conn,
                            stage_run_id,
                            _review(
                                "dataset" if issue.dataset_name else "paper",
                                analysis["title"],
                                issue.dataset_name,
                                issue.field,
                                issue.suggested_value,
                                issue.confidence,
                                issue.reason,
                                issue.evidence,
                                issue.source_url or analysis.get("paper_url"),
                            ),
                        )
                        review_count += 1
                reflected_count += 1
            self._complete_run(stage_run_id, f"Reflected {reflected_count} paper analyses; {review_count} review items.")
        EvidenceLogger(self.settings.evidence_dir, stage_run_id).event(
            "paper_analysis_reflection_completed",
            {"paper_count": reflected_count, "review_count": review_count, "analysis_run_id": analysis_run_id},
        )
        return stage_run_id

    def extract_datasets(self, run_id: int) -> int:
        source_run = self._require_run(run_id)
        stage_run_id = self._create_run(source_run["conference_instance_id"], "extract-datasets", "run", str(run_id))
        fetcher = PaperTextFetcher(allow_remote=self.settings.enable_web_search)
        extractor = DatasetExtractor()
        resolver = DatasetResolver(self.conn)
        rows = self.conn.execute(
            "SELECT * FROM papers WHERE conference_instance_id = ?", (source_run["conference_instance_id"],)
        ).fetchall()
        claim_count = 0
        review_count = 0
        with transaction(self.conn):
            for paper in rows:
                text = self._paper_text_for_analysis(paper["id"])
                if not text:
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

    def export(self, run_id: int | None, out: str, fmt: str, scope: str = "related") -> Path:
        if fmt == "json":
            return JsonExporter(self.conn).export(run_id, out, scope)
        return SpreadsheetExporter(self.conn).export(run_id, out, fmt)

    def export_paper_set(
        self,
        run_id: int,
        out: str,
        fmt: str = "csv",
        wireless_only: bool = False,
        wireless_source: str = "classify",
    ) -> Path:
        return PaperSetExporter(self.conn).export(
            run_id, out, fmt, wireless_only=wireless_only, wireless_source=wireless_source
        )

    def jaccard(
        self,
        run_id: int,
        manual_csv: str,
        title_col: str | None = None,
        authors_col: str | None = None,
        conference_col: str | None = None,
        year_col: str | None = None,
        wireless_only: bool = True,
        wireless_source: str = "classify",
        conference_filter: bool = True,
        fuzzy: bool = True,
        out: str | None = None,
    ) -> JaccardReport:
        report = compute_paper_list_jaccard(
            self.conn,
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
        if out:
            write_jaccard_report(report, out)
        return report

    def jaccard_all(
        self,
        manual_csv: str,
        title_col: str | None = None,
        authors_col: str | None = None,
        conference_col: str | None = None,
        year_col: str | None = None,
        wireless_only: bool = True,
        wireless_source: str = "classify",
        fuzzy: bool = True,
        auto_classify: bool = True,
        out: str | None = None,
    ) -> JaccardAggregate:
        if wireless_only and auto_classify and wireless_source == "classify":
            for run_id, _venue, _year in list_conference_runs(self.conn):
                ci_id = self._require_run(run_id)["conference_instance_id"]
                if self._latest_stage_run_id(ci_id, "classify-wireless") is None:
                    self.classify_wireless(run_id)
        aggregate = compute_paper_list_jaccard_all(
            self.conn,
            manual_csv,
            title_col=title_col,
            authors_col=authors_col,
            conference_col=conference_col,
            year_col=year_col,
            wireless_only=wireless_only,
            wireless_source=wireless_source,
            fuzzy=fuzzy,
        )
        if out:
            write_jaccard_aggregate(aggregate, out)
        return aggregate

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

    def latest_scope_assessment(self, run_id: int | None = None) -> sqlite3.Row | None:
        params: tuple[object, ...] = ()
        query = "SELECT * FROM scope_assessments"
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

    def _latest_stage_run_id(self, conference_instance_id: int | None, stage: str) -> int | None:
        row = self.conn.execute(
            """
            SELECT id FROM pipeline_runs
            WHERE conference_instance_id = ? AND stage = ? AND status = 'completed'
            ORDER BY id DESC LIMIT 1
            """,
            (conference_instance_id, stage),
        ).fetchone()
        return int(row["id"]) if row else None

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

    def _insert_paper_text_artifact(self, run_id: int, artifact: PaperTextArtifact) -> int:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO paper_text_artifacts
            (paper_id, run_id, source_type, source_url, fetch_status, content_text,
             content_sha256, error_message, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.paper_id,
                run_id,
                artifact.source_type,
                artifact.source_url,
                artifact.fetch_status,
                artifact.content_text,
                artifact.content_sha256,
                artifact.error_message,
                artifact.fetched_at,
            ),
        )
        row = self.conn.execute(
            """
            SELECT id FROM paper_text_artifacts
            WHERE paper_id = ? AND run_id = ? AND source_type = ?
              AND (source_url = ? OR (source_url IS NULL AND ? IS NULL))
            ORDER BY id DESC LIMIT 1
            """,
            (artifact.paper_id, run_id, artifact.source_type, artifact.source_url, artifact.source_url),
        ).fetchone()
        return int(row["id"])

    def _artifact_id_for_source(self, artifact_ids: dict[tuple[str, str | None], int], source_url: str | None) -> int | None:
        for (_, artifact_url), artifact_id in artifact_ids.items():
            if artifact_url == source_url:
                return artifact_id
        return None

    def _persist_paper_text_enrichment(
        self,
        run_id: int,
        enrichment: PaperTextEnrichment,
        *,
        provider_name: str,
        artifact_claim_type: str,
        snippet_claim_type: str,
        artifact_success_confidence: float,
        artifact_failure_confidence: float,
        artifact_success_statuses: set[str] | None = None,
        default_snippets_to_first_artifact: bool = True,
    ) -> TextPersistenceResult:
        artifact_ids: dict[tuple[str, str | None], int] = {}
        artifact_count = 0
        link_count = 0
        snippet_count = 0
        failed_artifacts: list[PaperTextArtifact] = []
        fetched_artifacts: list[PaperTextArtifact] = []
        success_statuses = artifact_success_statuses or {"fetched"}

        for artifact in enrichment.artifacts:
            artifact_id = self._insert_paper_text_artifact(run_id, artifact)
            artifact_ids[(artifact.source_type, artifact.source_url)] = artifact_id
            artifact_count += 1
            if artifact.fetch_status == "fetched":
                fetched_artifacts.append(artifact)
            if artifact.fetch_status in {"error", "empty", "rejected"}:
                failed_artifacts.append(artifact)
            self._insert_evidence(
                run_id,
                artifact.paper_id,
                None,
                artifact_claim_type,
                f"{artifact.source_type}:{artifact.fetch_status}",
                artifact.content_text[:1000] if artifact.content_text else artifact.error_message,
                artifact.source_url,
                artifact_success_confidence if artifact.fetch_status in success_statuses else artifact_failure_confidence,
                {"content_sha256": artifact.content_sha256, "provider": provider_name},
            )

        default_artifact_id = next(iter(artifact_ids.values()), None)
        for link in enrichment.links:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO paper_text_links
                (paper_id, artifact_id, run_id, url, link_text, link_type, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (link.paper_id, default_artifact_id, run_id, link.url, link.link_text, link.link_type, link.confidence),
            )
            link_count += 1

        for snippet in enrichment.snippets:
            artifact_id = self._artifact_id_for_source(artifact_ids, snippet.source_url)
            if artifact_id is None and default_snippets_to_first_artifact:
                artifact_id = default_artifact_id
            self.conn.execute(
                """
                INSERT INTO paper_text_snippets
                (paper_id, artifact_id, run_id, snippet_type, snippet_text, source_url, start_char, end_char, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snippet.paper_id,
                    artifact_id,
                    run_id,
                    snippet.snippet_type,
                    snippet.snippet_text,
                    snippet.source_url,
                    snippet.start_char,
                    snippet.end_char,
                    snippet.confidence,
                ),
            )
            snippet_count += 1
            self._insert_evidence(
                run_id,
                snippet.paper_id,
                None,
                snippet_claim_type,
                snippet.snippet_type,
                snippet.snippet_text,
                snippet.source_url,
                snippet.confidence,
            )

        return TextPersistenceResult(
            artifact_count=artifact_count,
            link_count=link_count,
            snippet_count=snippet_count,
            failed_artifacts=failed_artifacts,
            fetched_artifacts=fetched_artifacts,
        )

    def _paper_text_for_analysis(self, paper_id: int) -> str:
        rows = self.conn.execute(
            """
            SELECT content_text AS text FROM paper_text_artifacts
            WHERE paper_id = ? AND fetch_status IN ('available', 'fetched', 'reference_only', 'remote_skipped')
            UNION ALL
            SELECT snippet_text AS text FROM paper_text_snippets
            WHERE paper_id = ?
            """,
            (paper_id, paper_id),
        ).fetchall()
        return "\n\n".join(row["text"] for row in rows if row["text"])

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


def _placeholders(values: list[object]) -> str:
    return ",".join("?" for _ in values) if values else "NULL"


def _full_text_failure_summary(artifacts) -> str:
    if not artifacts:
        return "No PDF fetch attempts were recorded."
    counts: dict[str, int] = {}
    examples: list[str] = []
    for artifact in artifacts:
        key = artifact.error_message or artifact.fetch_status
        counts[key] = counts.get(key, 0) + 1
        if len(examples) < 3:
            examples.append(f"{artifact.source_url}: {key}")
    count_text = "; ".join(f"{count}x {reason}" for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5])
    example_text = " | ".join(examples)
    return f"{count_text}. Examples: {example_text}"
