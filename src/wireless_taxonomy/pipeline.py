
import json
import sqlite3
from typing import Any

from wireless_taxonomy.analyze.candidates import KeywordCandidateClassifier, LlmCandidateClassifier
from wireless_taxonomy.config import Settings
from wireless_taxonomy.llm import CreditExhaustedError
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
        # Skip network fetch if we already have papers for this conference instance.
        existing_count = self.conn.execute(
            "SELECT COUNT(*) AS n FROM papers WHERE conference_instance_id = ?", (ci_id,)
        ).fetchone()["n"]
        if existing_count > 0:
            # Return a synthetic run_id pointing to the existing data.
            last_run = self.conn.execute(
                "SELECT id FROM pipeline_runs WHERE conference_instance_id = ? AND stage = 'ingest' "
                "ORDER BY id DESC LIMIT 1", (ci_id,)
            ).fetchone()
            if last_run:
                return last_run["id"]
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
        workers: int = 1,
    ) -> int:
        """Wireless-candidate screening from title + abstract only.

        Stores per-paper label (yes/no/maybe) plus high-pass (yes) and low-pass
        (yes|maybe) filter flags for later Jaccard evaluation against a gold set.
        When ``cache`` is supplied, LLM labels are read from / written to it so a
        re-run reuses saved labels (unless ``refresh_llm`` forces fresh calls).
        ``workers`` bounds LLM-call parallelism; results and DB writes stay in
        input order, so the output is identical to a sequential run.
        """
        from wireless_taxonomy.parallel import parallel_map

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

        def _classify(paper_dict):
            return classifier.classify(paper_dict)

        try:
            with transaction(self.conn):
                for i, (paper, prediction, error) in enumerate(
                    parallel_map(_classify, (dict(r) for r in rows), workers), 1
                ):
                    if error is not None:
                        raise error
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
                    if cache is not None and i % 20 == 0:
                        cache.save()
                self._complete_run(
                    stage_run_id,
                    f"Classified {len(rows)} papers via {classifier.classifier}; "
                    f"yes={counts['yes']} maybe={counts['maybe']} no={counts['no']}.",
                )
        except Exception:
            if cache is not None:
                cache.save()
            raise
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
        workers: int = 1,
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
            ingest_run, use_llm=use_llm, cache=cache, refresh_llm=refresh_llm, workers=workers
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
                CASE wcp.confidence WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                p.title
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
                "confidence": row["confidence"] or "",
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
        workers: int = 1,
        web_search: bool = False,
    ) -> dict:
        """Report which papers in a venue/year have a legally fetchable full text.

        Ingests the accepted-paper list, backfills missing DOIs (so the
        open-access lookups are reliable), then asks the open metadata APIs
        (Unpaywall/OpenAlex/Semantic Scholar/arXiv) whether each paper has a
        legally hosted OA copy. Returns the full per-paper set plus coverage
        counts. It reads OA *status* only — it never downloads or scrapes
        paywalled full text.

        ``workers`` controls the number of concurrent OA resolver calls. Each
        paper is independent, so modest concurrency (4-6) reduces wall-clock
        time without changing the result. The shared cache is lock-protected.
        """
        import sys
        import time

        from wireless_taxonomy.analyze.oa_availability import OpenAccessResolver, summarize
        from wireless_taxonomy.parallel import parallel_map

        ingest_run = self.ingest(venue, year, source_type, source_value or "")
        if resolve_dois:
            self.enrich_abstracts(ingest_run, resolve_dois=True, cache=cache)
        conference_instance_id = self._require_run(ingest_run)["conference_instance_id"]
        source_urls = self._paper_source_urls(conference_instance_id)
        rows = self.conn.execute(
            "SELECT id, title, doi, paper_url FROM papers WHERE conference_instance_id = ? ORDER BY id",
            (conference_instance_id,),
        ).fetchall()
        resolver = resolver or OpenAccessResolver(cache=cache, web_search=web_search)
        n_total = len(rows)

        def _resolve_item(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
            i, row = item
            title = row["title"]
            doi = (row["doi"] or "").strip()
            url = source_urls.get(row["id"]) or (row["paper_url"] or "").strip()
            start = time.monotonic()
            result = resolver.resolve(title, doi or None, url or None)
            elapsed = time.monotonic() - start
            return {
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
                "_elapsed": elapsed,
                "_index": i,
            }

        papers: list[dict] = []
        items = ((i, row) for i, row in enumerate(rows, 1))
        for item, resolved, error in parallel_map(_resolve_item, items, workers):
            if error is not None:
                # Defensive: should not happen because resolve swallows exceptions,
                # but log and continue rather than crashing a long batch.
                print(
                    f"  [!] OA resolver error on paper {item[0]}/{n_total}: {error}",
                    file=sys.stderr,
                )
                continue
            i = resolved.pop("_index")
            elapsed = resolved.pop("_elapsed")
            if i % 10 == 0 or i == n_total:
                print(f"  [{i}/{n_total}] resolved OA coverage", file=sys.stderr)
            if elapsed > 30:
                print(
                    f"  [!] slow OA resolution ({elapsed:.1f}s): {resolved['title'][:60]}",
                    file=sys.stderr,
                )
            papers.append(resolved)
            if cache is not None and i % 50 == 0:
                cache.save()

        if cache is not None:
            cache.save()
        summary = summarize(papers)
        return {"venue": venue, "year": year, **summary, "papers": papers}

    def extract_datasets_conference(
        self,
        venue: str,
        year: int,
        source_type: str = "dblp",
        source_value: str | None = None,
        resolve_dois: bool = True,
        oa_pdf_urls: dict[str, str] | None = None,
        cache=None,
        extractor=None,
        fresh: bool = False,
        wireless_only: bool = True,
        workers: int = 1,
    ) -> dict:
        """Extract dataset records from every fetchable paper in a venue/year.

        Ingests the paper list (reusing any existing DB records), backfills
        DOIs, then for each paper fetches its PDF (if a non-ACM URL is known)
        or falls back to the abstract and asks the LLM to extract structured
        dataset records. Writes results to ``paper_analysis_dataset_claims`` and
        ``datasets``, and returns a summary dict suitable for JSON/CSV export.

        ``oa_pdf_urls`` is an optional {title: pdf_url} mapping produced by a
        prior ``text_availability_conference`` run so we skip re-fetching OA
        status.

        ``workers`` bounds thread parallelism for the PDF-fetch, classify, and
        extract stages. Each paper is independent (deterministic prompts,
        content-addressed caching), so results are identical to a sequential
        run; workers only perform network I/O while all SQLite writes stay on
        the calling thread.
        """
        import sys

        from wireless_taxonomy.analyze.dataset_extractor import DatasetExtractor
        from wireless_taxonomy.llm import LlmRouter

        ingest_run = self.ingest(venue, year, source_type, source_value or "")

        # Always resolve DOIs (needed for CrossRef BibTeX lookup).
        # Only fetch abstracts for papers that have no PDF URL and no abstract
        # already — papers with a fetchable PDF don't need it.
        if resolve_dois:
            pdf_url_map = oa_pdf_urls or {}
            self._enrich_for_extraction(ingest_run, pdf_url_map, cache=cache)

        conference_instance_id = self._require_run(ingest_run)["conference_instance_id"]
        rows = self.conn.execute(
            "SELECT id, title, authors, doi, abstract FROM papers WHERE conference_instance_id = ? ORDER BY id",
            (conference_instance_id,),
        ).fetchall()

        # Optionally filter to wireless-only papers using low_pass classifier
        # (yes + maybe) for maximum recall. Fetch PDFs for ALL papers first
        # (cached in SQLite) so the classifier sees full text.
        if wireless_only:
            from wireless_taxonomy.analyze.candidates import LlmCandidateClassifier
            from wireless_taxonomy.analyze.dataset_extractor import _fetch_pdf_bytes, load_cached_pdf, store_cached_pdf
            from wireless_taxonomy.llm import LlmRouter as _LlmRouter
            from wireless_taxonomy.parallel import parallel_map

            classifier = LlmCandidateClassifier(
                self.settings.llm, router=_LlmRouter(self.settings.llm), cache=cache
            )
            # Pre-fetch and cache all PDFs so both classification and extraction
            # can reuse them without re-downloading. Fetches run in worker
            # threads (pure network I/O); SQLite reads/writes stay on this
            # thread. Bytes are not retained — they're reloaded from the DB
            # cache per stage, keeping memory bounded at journal scale.
            n_total = len(rows)

            def _prefetch_items():
                for i, row in enumerate(rows, 1):
                    title = row["title"]
                    pdf_url = (oa_pdf_urls or {}).get(title)
                    cached = (
                        load_cached_pdf(self.conn, row["id"], pdf_url) is not None
                        if pdf_url else False
                    )
                    yield (i, row, pdf_url, cached)

            def _prefetch(item):
                _i, row, pdf_url, cached = item
                if not pdf_url or cached:
                    return None
                return _fetch_pdf_bytes(pdf_url, expected_title=row["title"])

            for item, fetched, error in parallel_map(_prefetch, _prefetch_items(), workers):
                i, row, pdf_url, cached = item
                title = row["title"]
                if not pdf_url:
                    print(f"  [{i}/{n_total}] No PDF URL: {title[:60]}", file=sys.stderr)
                elif cached:
                    print(f"  [{i}/{n_total}] PDF cached: {title[:60]}", file=sys.stderr)
                elif fetched:
                    store_cached_pdf(self.conn, row["id"], pdf_url, fetched)
                    print(f"  [{i}/{n_total}] PDF fetched: {title[:60]}", file=sys.stderr)
                else:
                    print(f"  [{i}/{n_total}] PDF failed: {title[:60]}", file=sys.stderr)

            # Classify (yes/maybe/no) with bounded parallelism. Workers only
            # call the LLM (the shared cache is lock-protected); PDF bytes are
            # loaded from the SQLite cache on this thread at submit time.
            def _classify_items():
                for i, row in enumerate(rows, 1):
                    pdf_url = (oa_pdf_urls or {}).get(row["title"])
                    pdf_bytes = load_cached_pdf(self.conn, row["id"], pdf_url) if pdf_url else None
                    yield (i, row, pdf_bytes)

            def _classify(item):
                _i, row, pdf_bytes = item
                return classifier.classify(dict(row), pdf_bytes=pdf_bytes)

            wireless_ids: set[int] = set()
            for item, pred, error in parallel_map(_classify, _classify_items(), workers):
                i, row, _pdf = item
                if isinstance(error, CreditExhaustedError):
                    if cache is not None:
                        cache.save()
                    print(
                        f"\n  Checkpoint saved after {i - 1}/{n_total} papers. "
                        "Re-run after reloading credits to resume.",
                        file=sys.stderr,
                    )
                    raise error
                if error is not None:
                    # On LLM failure, default to "maybe" for max recall rather
                    # than silently dropping the paper.
                    print(f"  [{i}/{n_total}] [!] error: {row['title'][:55]} — {error}", file=sys.stderr)
                    wireless_ids.add(row["id"])
                    continue
                label_icon = {"yes": "+", "maybe": "~", "no": "-"}.get(pred.label, "?")
                print(f"  [{i}/{n_total}] [{label_icon}] {pred.label}({pred.confidence}): {row['title'][:55]}", file=sys.stderr)
                if pred.low_pass:
                    wireless_ids.add(row["id"])
                # Save cache periodically so progress isn't lost on crash.
                if cache is not None and i % 20 == 0:
                    cache.save()
            if cache is not None:
                cache.save()
            total_before = len(rows)
            rows = [r for r in rows if r["id"] in wireless_ids]
            print(f"  Wireless filter: {len(rows)}/{total_before} papers pass low_pass (yes+maybe)", file=sys.stderr)

        effective_cache = None if fresh else cache
        extractor = extractor or DatasetExtractor(
            router=LlmRouter(self.settings.llm),
            cache=effective_cache,
            conn=self.conn,
        )

        run_id = self._create_run(conference_instance_id, "extract-datasets", source_type, source_value)
        n_extract = len(rows)
        results: list[dict] = []

        # Extraction runs with bounded parallelism: workers call the LLM only.
        # PDF bytes are loaded from the SQLite cache on this thread at submit
        # time (so workers never touch self.conn), and all DB writes below
        # happen on this thread as results arrive, in input order.
        from wireless_taxonomy.analyze.dataset_extractor import load_cached_pdf as _load_pdf
        from wireless_taxonomy.parallel import parallel_map as _pmap

        def _extract_items():
            for idx, row in enumerate(rows, 1):
                pdf_url = (oa_pdf_urls or {}).get(row["title"])
                pdf_bytes = _load_pdf(self.conn, row["id"], pdf_url) if pdf_url else None
                yield (idx, row, pdf_url, pdf_bytes)

        def _extract(item):
            _idx, row, pdf_url, pdf_bytes = item
            return extractor.extract(
                paper_id=row["id"],
                title=row["title"],
                authors=row["authors"] or "",
                venue=venue,
                year=year,
                doi=(row["doi"] or "").strip(),
                pdf_url=pdf_url,
                abstract=(row["abstract"] or "").strip() or None,
                pdf_bytes=pdf_bytes,
            )

        for item, result, error in _pmap(_extract, _extract_items(), workers):
            idx, row, pdf_url, _pdf = item
            title = row["title"]
            doi = (row["doi"] or "").strip()

            print(f"  [{idx}/{n_extract}] Extracting: {title[:60]}...", file=sys.stderr)
            if error is not None:
                if isinstance(error, CreditExhaustedError):
                    if cache is not None:
                        cache.save()
                    print(
                        f"\n  Checkpoint saved after {idx - 1}/{n_extract} papers. "
                        "Re-run after reloading credits to resume.",
                        file=sys.stderr,
                    )
                raise error

            with transaction(self.conn):
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO bibtex_entries(paper_id, citation_key, doi, bibtex)
                    VALUES (?, ?, ?, ?)
                    """,
                    (row["id"], result.bibtex_key, doi or None, result.bibtex),
                )
                for ds in result.datasets:
                    self.conn.execute(
                        """
                        INSERT OR IGNORE INTO datasets
                        (canonical_name, normalized_name, source_paper_id,
                         availability_status, modalities_json, osi_layers_json,
                         collection_environment, known_users_json, availability_notes, availability_url)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            ds.name,
                            ds.name.lower().strip(),
                            row["id"] if ds.relationship_type == "introduced" else None,
                            "open" if ds.availability else ("closed" if ds.availability is False else "unknown"),
                            json.dumps(ds.modalities),
                            json.dumps(ds.osi_layers),
                            ds.collection_environment,
                            json.dumps(ds.known_users),
                            ds.availability_notes or None,
                            ds.availability_url or None,
                        ),
                    )
                    dataset_row = self.conn.execute(
                        "SELECT id FROM datasets WHERE canonical_name = ?", (ds.name,)
                    ).fetchone()
                    dataset_id = dataset_row["id"] if dataset_row else None
                    self.conn.execute(
                        """
                        INSERT OR IGNORE INTO paper_analysis_dataset_claims
                        (paper_id, dataset_id, run_id, dataset_name, relationship_type,
                         confidence, modalities_json, osi_layers_json, evidence_text,
                         availability_status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row["id"], dataset_id, run_id, ds.name, ds.relationship_type,
                            ds.confidence, json.dumps(ds.modalities), json.dumps(ds.osi_layers),
                            ds.evidence_text,
                            "open" if ds.availability else ("closed" if ds.availability is False else "unknown"),
                        ),
                    )

            results.append({
                "paper_id": result.paper_id,
                "title": result.title,
                "authors": result.authors,
                "venue": venue,
                "year": year,
                "doi": result.doi,
                "bibtex_key": result.bibtex_key,
                "bibtex": result.bibtex,
                "extraction_source": result.extraction_source,
                "error": result.error,
                "datasets": [
                    {
                        "name": ds.name,
                        "relationship_type": ds.relationship_type,
                        "modalities": ds.modalities,
                        "osi_layers": ds.osi_layers,
                        "availability": ds.availability,
                        "availability_notes": ds.availability_notes,
                        "collection_environment": ds.collection_environment,
                        "known_users": ds.known_users,
                        "confidence": ds.confidence,
                        "evidence_text": ds.evidence_text,
                    }
                    for ds in result.datasets
                ],
            })

        # -- in-corpus usage counts (cross-paper within this venue/year) ------
        corpus_counts: dict[str, int] = {}
        for r in results:
            for d in r["datasets"]:
                corpus_counts[d["name"]] = corpus_counts.get(d["name"], 0) + 1
        for r in results:
            for d in r["datasets"]:
                d["usage_count"] = corpus_counts.get(d["name"], 1)
                d["usage_sources"] = {"corpus": corpus_counts.get(d["name"], 1)}

        self._complete_run(run_id, f"Extracted datasets for {len(results)} papers in {venue} {year}.")
        total_datasets = sum(len(r["datasets"]) for r in results)
        return {
            "venue": venue,
            "year": year,
            "total_papers": len(results),
            "papers_with_datasets": sum(1 for r in results if r["datasets"]),
            "total_dataset_records": total_datasets,
            "run_id": run_id,
            "papers": results,
        }

    def _enrich_for_extraction(self, ingest_run: int, pdf_url_map: dict[str, str], cache=None) -> None:
        """DOI-resolve all papers; only fetch abstracts for those without a PDF URL.

        Papers that already have a fetchable PDF don't need an abstract — the
        extractor will use the full text directly. This avoids wasting Semantic
        Scholar / OpenAlex quota on papers we'll never use the abstract for.
        """
        from wireless_taxonomy.analyze.abstracts import DoiResolver

        source_run = self._require_run(ingest_run)
        conference_instance_id = source_run["conference_instance_id"]
        stage_run_id = self._create_run(conference_instance_id, "enrich-for-extraction", "run", str(ingest_run))
        doi_resolver = DoiResolver(cache=cache)
        rows = self.conn.execute(
            "SELECT * FROM papers WHERE conference_instance_id = ? ORDER BY id",
            (conference_instance_id,),
        ).fetchall()
        from wireless_taxonomy.analyze.abstracts import AbstractEnricher

        source_urls = self._paper_source_urls(conference_instance_id)
        enricher = AbstractEnricher(cache=cache)
        with transaction(self.conn):
            for paper in rows:
                doi = (paper["doi"] or "").strip()
                if not doi and (paper["title"] or "").strip():
                    doi_result = doi_resolver.resolve(paper["title"])
                    if doi_result is not None:
                        doi = doi_result.doi
                        self.conn.execute("UPDATE papers SET doi = ? WHERE id = ?", (doi, paper["id"]))
                        self._insert_evidence(
                            stage_run_id, paper["id"], None, "doi_backfill",
                            doi_result.provider, doi, doi_result.source_url, 0.8,
                            {"provider": doi_result.provider},
                        )
                has_pdf = bool(pdf_url_map.get(paper["title"]))
                has_abstract = bool((paper["abstract"] or "").strip())
                if has_pdf or has_abstract:
                    continue
                result = enricher.fetch(paper["title"], doi or None, source_urls.get(paper["id"]))
                if result is not None:
                    self.conn.execute("UPDATE papers SET abstract = ? WHERE id = ?", (result.abstract, paper["id"]))
            self._complete_run(stage_run_id, f"DOI resolution + selective abstract enrichment for {len(rows)} papers.")

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


