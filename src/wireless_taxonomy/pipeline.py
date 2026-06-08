from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from wireless_taxonomy.analyze.candidates import KeywordCandidateClassifier, LlmCandidateClassifier
from wireless_taxonomy.analyze.wireless import WirelessClassifier
from wireless_taxonomy.config import Settings
from wireless_taxonomy.db import connect, migrate, transaction
from wireless_taxonomy.evidence import EvidenceLogger
from wireless_taxonomy.export.paper_set import PaperSetExporter
from wireless_taxonomy.ingest.base import validate_paper_seeds
from wireless_taxonomy.ingest.bibtex import BibtexIngestAdapter
from wireless_taxonomy.ingest.csv import CsvIngestAdapter
from wireless_taxonomy.ingest.dblp import DblpIngestAdapter
from wireless_taxonomy.ingest.gold import GoldSheetReader
from wireless_taxonomy.ingest.url import UrlIngestAdapter
from wireless_taxonomy.models import EvidenceClaim, PaperSeed, new_id, utc_now
from wireless_taxonomy.eval import overlap
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

    def enrich_abstracts(
        self,
        run_id: int,
        overwrite: bool = False,
        enricher=None,
        resolve_dois: bool = True,
        doi_resolver=None,
    ) -> int:
        """Backfill missing paper abstracts (and optionally DOIs) from open APIs.

        Pulls abstracts from OpenAlex/Crossref/Semantic Scholar (metadata, not
        the paywalled PDF), so it sidesteps the ACM full-text block. When
        ``resolve_dois`` is set, papers with no DOI (e.g. USENIX/NSDI, which DBLP
        indexes without DOIs) first get one resolved from their title via
        Crossref/OpenAlex; the recovered DOI then drives a more reliable
        abstract lookup and makes downstream gold matching exact.
        """
        from wireless_taxonomy.analyze.abstracts import AbstractEnricher, DoiResolver

        source_run = self._require_run(run_id)
        conference_instance_id = source_run["conference_instance_id"]
        stage_run_id = self._create_run(conference_instance_id, "enrich-abstracts", "run", str(run_id))
        logger = EvidenceLogger(self.settings.evidence_dir, stage_run_id)
        enricher = enricher or AbstractEnricher()
        if resolve_dois:
            doi_resolver = doi_resolver or DoiResolver()
        rows = self.conn.execute(
            "SELECT * FROM papers WHERE conference_instance_id = ? ORDER BY id", (conference_instance_id,)
        ).fetchall()
        filled = 0
        attempted = 0
        dois_resolved = 0
        with transaction(self.conn):
            for paper in rows:
                doi = (paper["doi"] or "").strip()
                if resolve_dois and not doi and (paper["title"] or "").strip():
                    doi_result = doi_resolver.resolve(paper["title"])
                    if doi_result is not None:
                        doi = doi_result.doi
                        self.conn.execute("UPDATE papers SET doi = ? WHERE id = ?", (doi, paper["id"]))
                        self._insert_evidence(
                            stage_run_id,
                            paper["id"],
                            None,
                            "doi_backfill",
                            doi_result.provider,
                            doi,
                            doi_result.source_url,
                            0.8,
                            {"provider": doi_result.provider},
                        )
                        dois_resolved += 1
                existing = (paper["abstract"] or "").strip()
                if existing and not overwrite:
                    continue
                attempted += 1
                result = enricher.fetch(paper["title"], doi or None)
                if result is None:
                    continue
                self.conn.execute("UPDATE papers SET abstract = ? WHERE id = ?", (result.abstract, paper["id"]))
                self._insert_evidence(
                    stage_run_id,
                    paper["id"],
                    None,
                    "abstract_enrichment",
                    result.provider,
                    result.abstract[:1000],
                    result.source_url,
                    0.85,
                    {"provider": result.provider},
                )
                filled += 1
            self._complete_run(
                stage_run_id,
                f"Filled {filled}/{attempted} missing abstracts, resolved {dois_resolved} DOIs "
                f"({len(rows)} papers total).",
            )
        logger.event(
            "enrich_abstracts_completed",
            {
                "papers": len(rows),
                "attempted": attempted,
                "filled": filled,
                "dois_resolved": dois_resolved,
                "overwrite": overwrite,
            },
        )
        return stage_run_id

    def classify_candidates(self, run_id: int, use_llm: bool = False) -> int:
        """Wireless-candidate screening from title + abstract only.

        Stores per-paper label (yes/no/maybe) plus high-pass (yes) and low-pass
        (yes|maybe) filter flags for later Jaccard evaluation against a gold set.
        """
        source_run = self._require_run(run_id)
        conference_instance_id = source_run["conference_instance_id"]
        stage_run_id = self._create_run(conference_instance_id, "classify-candidates", "run", str(run_id))
        logger = EvidenceLogger(self.settings.evidence_dir, stage_run_id)
        classifier = LlmCandidateClassifier(self.settings.llm) if use_llm else KeywordCandidateClassifier()
        rows = self.conn.execute(
            "SELECT * FROM papers WHERE conference_instance_id = ? ORDER BY id", (conference_instance_id,)
        ).fetchall()
        counts = {"yes": 0, "no": 0, "maybe": 0}
        with transaction(self.conn):
            for paper in rows:
                prediction = classifier.classify(dict(paper))
                counts[prediction.label] = counts.get(prediction.label, 0) + 1
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO wireless_candidate_predictions
                    (paper_id, run_id, classifier, model_version, label, confidence,
                     evidence, high_pass, low_pass, used_abstract)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        prediction.paper_id,
                        stage_run_id,
                        prediction.classifier,
                        prediction.model_version,
                        prediction.label,
                        prediction.confidence,
                        prediction.evidence,
                        int(prediction.high_pass),
                        int(prediction.low_pass),
                        int(prediction.used_abstract),
                    ),
                )
            self._complete_run(
                stage_run_id,
                f"Classified {len(rows)} papers via {classifier.classifier}; "
                f"yes={counts['yes']} maybe={counts['maybe']} no={counts['no']}.",
            )
        logger.event(
            "classify_candidates_completed",
            {"classifier": classifier.classifier, "paper_count": len(rows), "labels": counts},
        )
        return stage_run_id

    def classify_conference(
        self,
        venue: str,
        year: int,
        use_llm: bool = True,
        pass_mode: str = "high",
        resolve_dois: bool = True,
        source_type: str = "dblp",
        source_value: str | None = None,
    ) -> dict:
        """Sheet-free classification loop for a single venue/year.

        Ingests the accepted-paper list (DBLP by default), backfills missing DOIs
        and abstracts from open APIs, classifies each paper as wireless from
        title+abstract, and returns the wireless-flagged papers. No gold sheet is
        involved, so this is the reusable unit the experiment harness can call
        per conference-year.
        """
        if pass_mode not in {"high", "low"}:
            raise ValueError("pass_mode must be 'high' or 'low'")
        ingest_run = self.ingest(venue, year, source_type, source_value or "")
        self.enrich_abstracts(ingest_run, resolve_dois=resolve_dois)
        classify_run = self.classify_candidates(ingest_run, use_llm=use_llm)
        classifier = "llm" if use_llm else "keyword"
        flag_column = "high_pass" if pass_mode == "high" else "low_pass"
        conference_instance_id = self._require_run(ingest_run)["conference_instance_id"]
        rows = self.conn.execute(
            f"""
            SELECT p.title, p.authors, p.doi, p.abstract, ci.year, v.name AS venue,
                   wcp.label, wcp.confidence, wcp.used_abstract
            FROM papers p
            JOIN conference_instances ci ON ci.id = p.conference_instance_id
            JOIN venues v ON v.id = ci.venue_id
            JOIN wireless_candidate_predictions wcp ON wcp.paper_id = p.id
            WHERE p.conference_instance_id = ? AND wcp.run_id = ? AND wcp.{flag_column} = 1
            ORDER BY wcp.confidence DESC, p.title
            """,
            (conference_instance_id, classify_run),
        ).fetchall()
        papers = [
            {
                "title": row["title"],
                "authors": row["authors"] or "",
                "doi": row["doi"] or "",
                "venue": row["venue"],
                "year": row["year"],
                "wireless_label": row["label"],
                "confidence": round(float(row["confidence"]), 4) if row["confidence"] is not None else "",
                "used_abstract": bool(row["used_abstract"]),
                "has_abstract": bool((row["abstract"] or "").strip()),
            }
            for row in rows
        ]
        total = int(
            self.conn.execute(
                "SELECT COUNT(*) AS n FROM papers WHERE conference_instance_id = ?",
                (conference_instance_id,),
            ).fetchone()["n"]
        )
        with_abstract = int(
            self.conn.execute(
                "SELECT COUNT(*) AS n FROM papers WHERE conference_instance_id = ? "
                "AND abstract IS NOT NULL AND TRIM(abstract) <> ''",
                (conference_instance_id,),
            ).fetchone()["n"]
        )
        return {
            "venue": venue,
            "year": year,
            "classifier": classifier,
            "pass_mode": pass_mode,
            "total_papers": total,
            "papers_with_abstract": with_abstract,
            "wireless_count": len(papers),
            "ingest_run_id": ingest_run,
            "classify_run_id": classify_run,
            "papers": papers,
        }

    def import_gold(
        self,
        path: str,
        venue: str | None = None,
        year: int | None = None,
        wireless_only: bool = False,
    ) -> int:
        """Load a manually curated gold sheet (csv/xlsx) of wireless papers."""
        fmt = "xlsx" if Path(path).suffix.lower() in {".xlsx", ".xls"} else "csv"
        stage_run_id = self._create_run(None, "import-gold", fmt, path)
        logger = EvidenceLogger(self.settings.evidence_dir, stage_run_id)
        records = GoldSheetReader(path, venue, year, wireless_only).read()
        instance_ids = {(record.venue, record.year): self._conference_instance_id(record.venue, record.year) for record in records}
        imported = 0
        instances: set[tuple[str, int]] = set()
        with transaction(self.conn):
            for record in records:
                conference_instance_id = instance_ids[(record.venue, record.year)]
                cur = self.conn.execute(
                    """
                    INSERT OR IGNORE INTO gold_papers
                    (conference_instance_id, run_id, title, normalized_title, doi, normalized_doi, raw_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        conference_instance_id,
                        stage_run_id,
                        record.title,
                        record.normalized_title,
                        record.doi,
                        record.normalized_doi or None,
                        json.dumps(record.raw, ensure_ascii=False),
                    ),
                )
                if cur.rowcount:
                    imported += 1
                instances.add((record.venue, record.year))
            self._complete_run(
                stage_run_id,
                f"Imported {imported}/{len(records)} gold papers across {len(instances)} conference instance(s).",
            )
        logger.event(
            "import_gold_completed",
            {"path": path, "rows": len(records), "imported": imported, "instances": sorted(instances)},
        )
        return stage_run_id

    def evaluate_overlap(
        self,
        classifier: str = "keyword",
        pass_mode: str = "high",
        fuzzy_threshold: float = 0.92,
        scope_to_universe: bool = False,
    ) -> dict:
        """Jaccard/IoU of the predicted wireless set vs the gold set, per venue/year.

        Only conference instances that have an imported gold set are scored. When
        ``scope_to_universe`` is set, gold papers absent from the ingested main
        proceedings (co-located workshop papers) are dropped from the denominator.
        """
        if pass_mode not in {"high", "low"}:
            raise ValueError("pass_mode must be 'high' or 'low'")
        flag_column = "high_pass" if pass_mode == "high" else "low_pass"
        gold_instances = self.conn.execute(
            """
            SELECT ci.id AS conference_instance_id, v.name AS venue, ci.year AS year
            FROM conference_instances ci
            JOIN venues v ON v.id = ci.venue_id
            WHERE ci.id IN (SELECT DISTINCT conference_instance_id FROM gold_papers)
            ORDER BY v.name, ci.year
            """
        ).fetchall()

        instance_rows: list[dict] = []
        mismatches: list[dict] = []
        for instance in gold_instances:
            conference_instance_id = instance["conference_instance_id"]
            gold_refs = [
                overlap.PaperRef.build(f"gold:{row['id']}", row["title"], row["doi"])
                for row in self.conn.execute(
                    "SELECT id, title, doi FROM gold_papers WHERE conference_instance_id = ?",
                    (conference_instance_id,),
                )
            ]
            latest = self.conn.execute(
                """
                SELECT MAX(wcp.run_id) AS run_id
                FROM wireless_candidate_predictions wcp
                JOIN papers p ON p.id = wcp.paper_id
                WHERE p.conference_instance_id = ? AND wcp.classifier = ?
                """,
                (conference_instance_id, classifier),
            ).fetchone()
            predict_run_id = latest["run_id"] if latest else None
            predicted_refs = []
            if predict_run_id is not None:
                predicted_refs = [
                    overlap.PaperRef.build(f"paper:{row['id']}", row["title"], row["doi"])
                    for row in self.conn.execute(
                        f"""
                        SELECT p.id, p.title, p.doi
                        FROM papers p
                        JOIN wireless_candidate_predictions wcp ON wcp.paper_id = p.id
                        WHERE p.conference_instance_id = ? AND wcp.classifier = ?
                          AND wcp.run_id = ? AND wcp.{flag_column} = 1
                        """,
                        (conference_instance_id, classifier, predict_run_id),
                    )
                ]
            universe_refs = [
                overlap.PaperRef.build(f"paper:{row['id']}", row["title"], row["doi"])
                for row in self.conn.execute(
                    "SELECT id, title, doi FROM papers WHERE conference_instance_id = ?",
                    (conference_instance_id,),
                )
            ]

            result = overlap.match(predicted_refs, gold_refs, fuzzy_threshold)
            # Split missed gold papers into classifier misses vs coverage gaps.
            in_universe = overlap.match(result.unmatched_b, universe_refs, fuzzy_threshold)
            fn_missed = len(in_universe.matched)
            fn_missing_from_universe = len(in_universe.unmatched_a)

            instance_rows.append(
                {
                    "venue": instance["venue"],
                    "year": instance["year"],
                    "tp": len(result.matched),
                    "fp": len(result.unmatched_a),
                    "fn": len(result.unmatched_b),
                    "fn_missed": fn_missed,
                    "fn_missing_from_universe": fn_missing_from_universe,
                }
            )
            mismatches.append(
                {
                    "venue": instance["venue"],
                    "year": instance["year"],
                    "predicted_run_id": predict_run_id,
                    "false_positives": [ref.title for ref in result.unmatched_a],
                    "false_negatives_classifier_miss": [b.title for _, b in in_universe.matched],
                    "false_negatives_missing_from_universe": [ref.title for ref in in_universe.unmatched_a],
                }
            )

        aggregates = overlap.aggregate(instance_rows, scope_to_universe=scope_to_universe)
        return {
            "classifier": classifier,
            "pass_mode": pass_mode,
            "fuzzy_threshold": fuzzy_threshold,
            "scope_to_universe": scope_to_universe,
            "instances": aggregates["per_conference_year"],
            "per_conference": aggregates["per_conference"],
            "overall": aggregates["overall"],
            "mismatches": mismatches,
        }


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


    def status(self, run_id: int | None = None) -> list[sqlite3.Row]:
        if run_id is None:
            return list(self.conn.execute("SELECT * FROM pipeline_runs ORDER BY id"))
        return list(self.conn.execute("SELECT * FROM pipeline_runs WHERE id = ?", (run_id,)))


    def _adapter(self, venue: str, year: int, source_type: str, source_value: str):
        if source_type == "url":
            return UrlIngestAdapter(venue, year, source_value, self.settings.llm)
        if source_type == "bibtex":
            return BibtexIngestAdapter(venue, year, source_value)
        if source_type == "csv":
            return CsvIngestAdapter(venue, year, source_value)
        if source_type == "dblp":
            return DblpIngestAdapter(venue, year)
        raise ValueError("source_type must be url, bibtex, csv, or dblp")

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


