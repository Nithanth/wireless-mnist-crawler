from __future__ import annotations

import json
import sqlite3

from wireless_taxonomy.analyze.candidates import KeywordCandidateClassifier, LlmCandidateClassifier
from wireless_taxonomy.config import Settings
from wireless_taxonomy.db import connect, migrate, transaction
from wireless_taxonomy.evidence import EvidenceLogger
from wireless_taxonomy.ingest.base import validate_paper_seeds
from wireless_taxonomy.ingest.bibtex import BibtexIngestAdapter
from wireless_taxonomy.ingest.csv import CsvIngestAdapter
from wireless_taxonomy.ingest.dblp import DblpIngestAdapter
from wireless_taxonomy.ingest.url import UrlIngestAdapter
from wireless_taxonomy.models import EvidenceClaim, PaperSeed, new_id, utc_now
from wireless_taxonomy.review.queue import insert_review_item


def _cache_has_abstract(cache, title: str | None, doi: str | None) -> bool:
    """True if the disk cache already holds a real abstract for this paper.

    A cached *miss* (provider == "miss" or empty abstract) returns False so the
    batch lookup still gets a chance to fill it; only a genuine cached abstract
    lets a warm re-run skip the network batch call.
    """
    entry = cache.get_abstract(title, doi)
    if not entry:
        return False
    return bool(entry.get("abstract")) and entry.get("provider") != "miss"


class Pipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        migrate(settings.db_path)
        self.conn = connect(settings.db_path)

    def close(self) -> None:
        self.conn.close()

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

    def enrich_abstracts(
        self,
        run_id: int,
        overwrite: bool = False,
        enricher=None,
        resolve_dois: bool = True,
        doi_resolver=None,
        cache=None,
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
        enricher = enricher or AbstractEnricher(cache=cache)
        if resolve_dois:
            doi_resolver = doi_resolver or DoiResolver(cache=cache)
        rows = self.conn.execute(
            "SELECT * FROM papers WHERE conference_instance_id = ? ORDER BY id", (conference_instance_id,)
        ).fetchall()
        source_urls = self._paper_source_urls(conference_instance_id)
        # Batch-fetch abstracts by DOI up front in a single Semantic Scholar
        # request. One batched call per conference is dramatically more reliable
        # than one GET per paper, which gets 429-throttled on a shared IP and
        # silently drops most abstracts (notably ACM venues like IMC/SIGCOMM).
        # Papers already served by the disk cache are excluded so a warm re-run
        # (fresh DB, but cached abstracts) skips the network batch call entirely
        # instead of re-paying its 429-throttled retries.
        batch_items = [
            (paper["title"], (paper["doi"] or "").strip())
            for paper in rows
            if (paper["doi"] or "").strip()
            and (overwrite or not (paper["abstract"] or "").strip())
            and not (cache is not None and not overwrite and _cache_has_abstract(cache, paper["title"], paper["doi"]))
        ]
        if batch_items and hasattr(enricher, "prefetch_semantic_scholar"):
            enricher.prefetch_semantic_scholar(batch_items)
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
                # Persist the cache periodically so a long, slow run never loses
                # already-resolved (or already-missed) abstracts/DOIs if it's
                # interrupted. Keyed on attempts so all-miss runs still save.
                if cache is not None and attempted % 20 == 0:
                    cache.save()
                result = enricher.fetch(paper["title"], doi or None, source_urls.get(paper["id"]))
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
        if cache is not None:
            cache.save()
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

    def classify_candidates(
        self,
        run_id: int,
        use_llm: bool = False,
        cache=None,
        refresh_llm: bool = False,
    ) -> int:
        """Wireless-candidate screening from title + abstract only.

        Stores per-paper label (yes/no/maybe) plus high-pass (yes) and low-pass
        (yes|maybe) filter flags for later Jaccard evaluation against a gold set.
        When ``cache`` is supplied, LLM labels are read from / written to it so a
        re-run reuses saved labels (unless ``refresh_llm`` forces fresh calls).
        """
        source_run = self._require_run(run_id)
        conference_instance_id = source_run["conference_instance_id"]
        stage_run_id = self._create_run(conference_instance_id, "classify-candidates", "run", str(run_id))
        logger = EvidenceLogger(self.settings.evidence_dir, stage_run_id)
        classifier = (
            LlmCandidateClassifier(self.settings.llm, cache=cache, refresh=refresh_llm)
            if use_llm
            else KeywordCandidateClassifier()
        )
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
        if cache is not None:
            cache.save()
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
        resolve_dois: bool = True,
        source_type: str = "dblp",
        source_value: str | None = None,
        cache=None,
        refresh_llm: bool = False,
    ) -> dict:
        """Sheet-free classification loop for a single venue/year.

        Ingests the accepted-paper list (DBLP by default), backfills missing DOIs
        and abstracts from open APIs, classifies each paper as wireless from
        title+abstract, and returns the **full** labelled set (every paper with
        its yes/maybe/no label). No gold sheet is involved, so this is the
        reusable unit the experiment harness can call per conference-year. The
        full set is what lets a downstream eval recover both the predicted
        positives (by label) and the proceedings universe (all rows).
        """
        ingest_run = self.ingest(venue, year, source_type, source_value or "")
        self.enrich_abstracts(ingest_run, resolve_dois=resolve_dois, cache=cache)
        classify_run = self.classify_candidates(
            ingest_run, use_llm=use_llm, cache=cache, refresh_llm=refresh_llm
        )
        classifier = "llm" if use_llm else "keyword"
        conference_instance_id = self._require_run(ingest_run)["conference_instance_id"]
        rows = self.conn.execute(
            """
            SELECT p.title, p.authors, p.doi, p.abstract, ci.year, v.name AS venue,
                   wcp.label, wcp.confidence, wcp.used_abstract
            FROM papers p
            JOIN conference_instances ci ON ci.id = p.conference_instance_id
            JOIN venues v ON v.id = ci.venue_id
            JOIN wireless_candidate_predictions wcp ON wcp.paper_id = p.id
            WHERE p.conference_instance_id = ? AND wcp.run_id = ?
            ORDER BY
                CASE wcp.label WHEN 'yes' THEN 0 WHEN 'maybe' THEN 1 ELSE 2 END,
                wcp.confidence DESC, p.title
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
                "label": row["label"],
                "confidence": round(float(row["confidence"]), 4) if row["confidence"] is not None else "",
                "used_abstract": bool(row["used_abstract"]),
                "has_abstract": bool((row["abstract"] or "").strip()),
            }
            for row in rows
        ]
        counts = {"yes": 0, "maybe": 0, "no": 0}
        for paper in papers:
            counts[paper["label"]] = counts.get(paper["label"], 0) + 1
        total = len(papers)
        with_abstract = sum(1 for paper in papers if paper["has_abstract"])
        return {
            "venue": venue,
            "year": year,
            "classifier": classifier,
            "total_papers": total,
            "papers_with_abstract": with_abstract,
            "counts": counts,
            "ingest_run_id": ingest_run,
            "classify_run_id": classify_run,
            "papers": papers,
        }

    def text_availability_conference(
        self,
        venue: str,
        year: int,
        source_type: str = "dblp",
        source_value: str | None = None,
        resolve_dois: bool = True,
        cache=None,
        resolver=None,
    ) -> dict:
        """Report which papers in a venue/year have a legally fetchable full text.

        Ingests the accepted-paper list, backfills missing DOIs (so the
        open-access lookups are reliable), then asks the open metadata APIs
        (Unpaywall/OpenAlex/Semantic Scholar/arXiv) whether each paper has a
        legally hosted OA copy. Returns the full per-paper set plus coverage
        counts. It reads OA *status* only — it never downloads or scrapes
        paywalled full text.
        """
        from wireless_taxonomy.analyze.oa_availability import OpenAccessResolver, summarize

        ingest_run = self.ingest(venue, year, source_type, source_value or "")
        if resolve_dois:
            self.enrich_abstracts(ingest_run, resolve_dois=True, cache=cache)
        conference_instance_id = self._require_run(ingest_run)["conference_instance_id"]
        source_urls = self._paper_source_urls(conference_instance_id)
        rows = self.conn.execute(
            "SELECT id, title, doi, paper_url FROM papers WHERE conference_instance_id = ? ORDER BY id",
            (conference_instance_id,),
        ).fetchall()
        resolver = resolver or OpenAccessResolver(cache=cache)
        papers: list[dict] = []
        for row in rows:
            title = row["title"]
            doi = (row["doi"] or "").strip()
            url = source_urls.get(row["id"]) or (row["paper_url"] or "").strip()
            result = resolver.resolve(title, doi or None, url or None)
            papers.append(
                {
                    "title": title,
                    "doi": doi,
                    "venue": venue,
                    "year": year,
                    "fetchable": result.fetchable,
                    "oa_status": result.oa_status,
                    "license": result.license,
                    "pdf_url": result.pdf_url,
                    "provider": result.provider,
                    "source_url": result.source_url,
                }
            )
        summary = summarize(papers)
        return {"venue": venue, "year": year, **summary, "papers": papers}

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

    def _paper_source_urls(self, conference_instance_id: int) -> dict[str, str]:
        """Map each paper to a source URL for page-scrape abstract fallbacks.

        Prefers a publisher landing page (e.g. the USENIX paper page DBLP links
        via ``ee``) over the generic DBLP TOC URL, so the USENIX abstract
        provider gets the per-paper page it needs.
        """
        rows = self.conn.execute(
            """
            SELECT ps.paper_id AS paper_id, ps.source_url AS source_url
            FROM paper_sources ps
            JOIN papers p ON p.id = ps.paper_id
            WHERE p.conference_instance_id = ?
            ORDER BY ps.id
            """,
            (conference_instance_id,),
        ).fetchall()
        urls: dict[str, str] = {}
        for row in rows:
            url = (row["source_url"] or "").strip()
            if not url:
                continue
            current = urls.get(row["paper_id"])
            if current is None or ("usenix.org" in url and "usenix.org" not in current):
                urls[row["paper_id"]] = url
        return urls


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


