"""Admin commands: llm-config, corpus-status, prune, fill-availability."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from typing import Optional

import typer

from wireless_taxonomy.config import load_settings


def register(app: typer.Typer, advanced: typer.Typer | None = None) -> None:
    # Internal commands go on the advanced subapp if provided, else fall back
    # to the main app (for backwards compatibility in tests).
    _adv = advanced if advanced is not None else app

    @_adv.command("llm-config")
    def llm_config(db: str = typer.Option("taxonomy.sqlite", "--db")) -> None:
        """Show configured LLM providers, models, and API key status."""
        settings = load_settings(db)
        typer.echo(f"Primary provider: {settings.llm.primary_provider}")
        fallbacks = ", ".join(settings.llm.fallback_providers) if settings.llm.fallback_providers else "(none)"
        typer.echo(f"Fallback providers: {fallbacks}")
        for provider in settings.llm.ordered_providers():
            key_status = "configured" if provider.api_key_configured else f"missing {provider.api_key_env}"
            typer.echo(f"- {provider.provider}: model={provider.model}, key={key_status}")

    @_adv.command("corpus-status")
    def corpus_status(db: str = typer.Option("taxonomy.sqlite", "--db")) -> None:
        """Show what's in the corpus: venues, years, paper counts, extraction status."""
        from wireless_taxonomy.db import connect

        conn = connect(db)
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT v.name AS venue, ci.year, COUNT(p.id) AS papers,
                   SUM(CASE WHEN wcp.label IS NOT NULL THEN 1 ELSE 0 END) AS classified,
                   SUM(CASE WHEN be.id IS NOT NULL THEN 1 ELSE 0 END) AS extracted,
                   SUM(CASE WHEN pta.content_text != '' THEN 1 ELSE 0 END) AS has_pdf
            FROM papers p
            JOIN conference_instances ci ON ci.id = p.conference_instance_id
            JOIN venues v ON v.id = ci.venue_id
            LEFT JOIN wireless_candidate_predictions wcp ON wcp.paper_id = p.id
            LEFT JOIN bibtex_entries be ON be.paper_id = p.id
            LEFT JOIN paper_text_artifacts pta ON pta.paper_id = p.id AND pta.fetch_status = 'success'
            GROUP BY v.name, ci.year
            ORDER BY v.name, ci.year
        """).fetchall()

        if not rows:
            typer.echo("Corpus is empty.")
            conn.close()
            raise typer.Exit()

        total_papers = sum(r["papers"] for r in rows)
        total_extracted = sum(r["extracted"] for r in rows)
        total_classified = sum(r["classified"] for r in rows)
        n_datasets = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        n_claims = conn.execute("SELECT COUNT(*) FROM paper_analysis_dataset_claims").fetchone()[0]

        typer.echo(f"Corpus: {total_papers} papers across {len(rows)} venue-years")
        typer.echo(f"  Classified: {total_classified}  Extracted: {total_extracted}  Datasets: {n_datasets}  Claims: {n_claims}")
        typer.echo("")
        typer.echo(f"  {'Venue':<12} {'Year':<6} {'Papers':<8} {'Classified':<12} {'Extracted':<11} {'Has PDF'}")
        typer.echo(f"  {'─'*12} {'─'*6} {'─'*8} {'─'*12} {'─'*11} {'─'*8}")
        for r in rows:
            typer.echo(
                f"  {r['venue']:<12} {r['year']:<6} {r['papers']:<8} "
                f"{r['classified']:<12} {r['extracted']:<11} {r['has_pdf']}"
            )

        typer.echo("")
        runs = conn.execute("""
            SELECT pr.id, v.name AS venue, ci.year, pr.stage, pr.status, pr.message,
                   pr.started_at
            FROM pipeline_runs pr
            JOIN conference_instances ci ON ci.id = pr.conference_instance_id
            JOIN venues v ON v.id = ci.venue_id
            ORDER BY pr.id DESC LIMIT 20
        """).fetchall()
        if runs:
            typer.echo("  Recent runs (newest first):")
            for r in runs:
                status_icon = {"completed": "+", "running": "~", "failed": "x"}.get(r["status"], "?")
                msg = f" — {r['message'][:60]}" if r["message"] else ""
                typer.echo(f"    [{status_icon}] run {r['id']:>3}: {r['venue']} {r['year']} / {r['stage']} ({r['status']}){msg}")

        conn.close()

    @_adv.command("prune")
    def prune(
        venue: Optional[str] = typer.Option(None, "--venue", help="Venue to prune (e.g. SIGCOMM). Required unless --run-id is given."),
        year: Optional[int] = typer.Option(None, "--year", help="Year to prune. Required unless --run-id is given."),
        run_id: Optional[int] = typer.Option(None, "--run-id", help="Specific pipeline_run ID to prune."),
        stage: Optional[str] = typer.Option(None, "--stage", help="Only prune a specific stage (extract-datasets, classify-candidates, etc.)"),
        keep_pdfs: bool = typer.Option(True, "--keep-pdfs/--drop-pdfs", help="Keep cached PDF text artifacts (default: keep)."),
        db: str = typer.Option("taxonomy.sqlite", "--db"),
    ) -> None:
        """Prune extraction/classification results by venue/year or run_id.

        \b
        Examples:
          prune --venue SIGCOMM --year 2023               # all stages for SIGCOMM 2023
          prune --venue IMC --year 2024 --stage extract-datasets  # only extraction
          prune --run-id 42                               # a specific pipeline run
          prune --venue NSDI --year 2022 --drop-pdfs      # also clear cached PDFs

        Cached abstracts, DOIs, and OA lookups (.wt_cache.json) are NOT touched.
        Re-run extract-datasets to regenerate pruned data (LLM cache was already
        cleared so fresh extractions will happen).
        """
        from wireless_taxonomy.db import connect, transaction

        conn = connect(db)
        conn.row_factory = sqlite3.Row

        if run_id is not None:
            run_row = conn.execute("SELECT * FROM pipeline_runs WHERE id = ?", (run_id,)).fetchone()
            if not run_row:
                typer.echo(f"Run ID {run_id} not found.", err=True)
                conn.close()
                raise typer.Exit(1)
            target_runs = [run_row]
            ci_ids = {run_row["conference_instance_id"]}
        elif venue and year:
            ci_row = conn.execute("""
                SELECT ci.id FROM conference_instances ci
                JOIN venues v ON v.id = ci.venue_id
                WHERE LOWER(v.name) = LOWER(?) AND ci.year = ?
            """, (venue, year)).fetchone()
            if not ci_row:
                typer.echo(f"No data for {venue} {year} in the database.", err=True)
                conn.close()
                raise typer.Exit(1)
            ci_ids = {ci_row["id"]}
            query = "SELECT * FROM pipeline_runs WHERE conference_instance_id = ?"
            params: list = [ci_row["id"]]
            if stage:
                query += " AND stage = ?"
                params.append(stage)
            target_runs = conn.execute(query, params).fetchall()
        else:
            typer.echo("Provide --venue + --year or --run-id.", err=True)
            conn.close()
            raise typer.Exit(1)

        if not target_runs:
            typer.echo("No matching pipeline runs found.")
            conn.close()
            raise typer.Exit()

        typer.echo(f"Will prune {len(target_runs)} pipeline run(s):")
        for r in target_runs:
            typer.echo(f"  run {r['id']}: {r['stage']} ({r['status']}) — {r['message'] or '(no message)'}")

        run_ids = [r["id"] for r in target_runs]
        stages = {r["stage"] for r in target_runs}

        typer.confirm("Proceed?", abort=True)

        with transaction(conn):
            deleted = {}

            if "classify-candidates" in stages or stage is None:
                cur = conn.execute(
                    f"DELETE FROM wireless_candidate_predictions WHERE run_id IN ({','.join('?' * len(run_ids))})",
                    run_ids,
                )
                deleted["wireless_candidate_predictions"] = cur.rowcount

            if "extract-datasets" in stages or stage is None:
                cur = conn.execute(
                    f"DELETE FROM paper_analysis_dataset_claims WHERE run_id IN ({','.join('?' * len(run_ids))})",
                    run_ids,
                )
                deleted["paper_analysis_dataset_claims"] = cur.rowcount

                for ci_id in ci_ids:
                    cur = conn.execute("""
                        DELETE FROM bibtex_entries WHERE paper_id IN (
                            SELECT id FROM papers WHERE conference_instance_id = ?
                        )
                    """, (ci_id,))
                    deleted["bibtex_entries"] = deleted.get("bibtex_entries", 0) + cur.rowcount

            cur = conn.execute("""
                DELETE FROM datasets WHERE id NOT IN (
                    SELECT DISTINCT dataset_id FROM paper_analysis_dataset_claims WHERE dataset_id IS NOT NULL
                )
            """)
            deleted["datasets (orphaned)"] = cur.rowcount

            if not keep_pdfs:
                for ci_id in ci_ids:
                    cur = conn.execute("""
                        DELETE FROM paper_text_artifacts WHERE paper_id IN (
                            SELECT id FROM papers WHERE conference_instance_id = ?
                        )
                    """, (ci_id,))
                    deleted["paper_text_artifacts"] = deleted.get("paper_text_artifacts", 0) + cur.rowcount

            conn.execute(
                f"DELETE FROM pipeline_runs WHERE id IN ({','.join('?' * len(run_ids))})",
                run_ids,
            )
            deleted["pipeline_runs"] = len(run_ids)

        typer.echo("\nPruned:")
        for table, count in sorted(deleted.items()):
            if count > 0:
                typer.echo(f"  {table}: {count} rows")
        typer.echo("\nDone. Re-run extract-datasets to regenerate.")
        conn.close()

    @_adv.command("purge-cache")
    def purge_cache(
        dataset_name: list[str] = typer.Option(
            ..., "--dataset", "-d",
            help="Dataset name substring to purge (case-insensitive). "
                 "Repeat to purge multiple. All LLM cache entries whose "
                 "extracted datasets contain any of these substrings are removed.",
        ),
        cache_file: str = typer.Option(".wt_cache.json", "--cache-path", "--cache"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Print what would be removed without writing."),
    ) -> None:
        """Surgically purge LLM extraction cache entries by dataset name.

        \b
        Use this after a prompt change that affects a known set of dataset
        names — cheaper than re-extracting every paper.

        \b
        Examples:
          purge-cache -d "MovieLens" -d "OpenWeatherMap"
          purge-cache -d "census" --dry-run
        """
        import pathlib

        from wireless_taxonomy.analyze.cache import MetadataCache

        path = pathlib.Path(cache_file)
        if not path.exists():
            typer.echo(f"Cache file not found: {cache_file}", err=True)
            raise typer.Exit(1)

        c = MetadataCache(path)
        needles = [n.lower() for n in dataset_name]

        doomed: list[tuple[str, str]] = []  # (cache_key, matched dataset name)
        for key in list(c.llm.keys()):
            if not key.startswith("de:"):
                continue
            val = c.get_llm(key)
            datasets = val.get("datasets", []) if isinstance(val, dict) else []
            hit = next(
                (ds.get("name", "") for ds in datasets
                 if any(needle in (ds.get("name") or "").lower() for needle in needles)),
                None,
            )
            if hit:
                doomed.append((key, hit))

        if dry_run:
            typer.echo(f"[dry-run] Would remove {len(doomed)} cache entries:")
            for _, n in doomed:
                typer.echo(f"  {n}")
        else:
            for key, _ in doomed:
                c.delete_llm(key)
            c.save()  # atomic write via tempfile + os.replace
            typer.echo(f"Purged {len(doomed)} cache entries containing: {', '.join(dataset_name)}")
            typer.echo("Re-run extract-datasets for affected venues to get fresh extractions.")

    @_adv.command("fill-availability")
    def fill_availability(
        db: str = typer.Option("taxonomy.sqlite", "--db"),
        cache_file: str = typer.Option(".wt_cache.json", "--cache-path", "--cache"),
        limit: int = typer.Option(0, "--limit", help="Max datasets to process (0 = all)."),
        dry_run: bool = typer.Option(False, "--dry-run", help="Print candidates without making LLM calls."),
        relationship: str = typer.Option(
            "all", "--relationship",
            help="Which relationship types to process: 'introduced', 'reused', or 'all'.",
        ),
    ) -> None:
        """Fill unknown availability for datasets using a targeted second-pass LLM call.

        \b
        For each dataset whose availability is unknown AND whose source paper has
        a PDF in the cache, sends a short focused prompt to the LLM asking only
        about availability.  Much cheaper than a full re-extraction.

        \b
        Examples:
          fill-availability                         # all unknown, all types
          fill-availability --relationship introduced
          fill-availability --limit 20 --dry-run
        """
        from wireless_taxonomy.analyze.cache import MetadataCache
        from wireless_taxonomy.analyze.dataset_extractor import _check_url_live
        from wireless_taxonomy.db import connect
        from wireless_taxonomy.llm import LlmRequest, LlmRouter

        settings = load_settings(db)
        conn = connect(db)
        conn.row_factory = sqlite3.Row
        cache = MetadataCache(cache_file)
        router = LlmRouter(settings.llm)

        # ── Candidate query ───────────────────────────────────────────────────
        # We need: dataset id/name, the paper that introduced/claimed it, and
        # the paper's PDF text artifact.  We only attempt papers with a
        # successfully fetched PDF — abstract-only papers already had their best
        # shot at extraction.
        rel_filter = ""
        if relationship == "introduced":
            rel_filter = "AND c.relationship_type = 'introduced'"
        elif relationship == "reused":
            rel_filter = "AND c.relationship_type = 'reused'"

        rows = conn.execute(f"""
            SELECT DISTINCT
                d.id        AS dataset_id,
                d.canonical_name,
                p.id        AS paper_id,
                p.title,
                p.abstract,
                pta.content_text AS pdf_text
            FROM datasets d
            JOIN paper_analysis_dataset_claims c ON c.dataset_id = d.id
            JOIN papers p ON p.id = c.paper_id
            LEFT JOIN paper_text_artifacts pta
                   ON pta.paper_id = p.id AND pta.fetch_status = 'ok'
            WHERE d.availability_status = 'unknown'
              {rel_filter}
            ORDER BY d.canonical_name
        """).fetchall()

        if not rows:
            typer.echo("No unknown-availability datasets with PDF text found.")
            conn.close()
            return

        candidates = list(rows)
        if limit and limit > 0:
            candidates = candidates[:limit]

        typer.echo(f"Found {len(candidates)} unknown-availability datasets to process.")
        if dry_run:
            for r in candidates:
                src = "PDF" if r["pdf_text"] else "abstract"
                typer.echo(f"  [{src}] {r['canonical_name']} (paper: {r['title'][:50]})")
            conn.close()
            return

        # ── Focused availability prompt ───────────────────────────────────────
        _AVAIL_PROMPT = (
            "You are a research data curator. Given the text of a research paper, "
            "determine whether the dataset named below is publicly available.\n\n"
            "Dataset name: {dataset_name}\n"
            "Paper title: {title}\n\n"
            "Paper text:\n---\n{text}\n---\n\n"
            "Answer in JSON only:\n"
            '{{"available": true|false|null, '
            '"url": "<exact URL from paper or empty string>", '
            '"notes": "<one sentence from the paper about availability or empty string>"}}\n\n'
            "Rules:\n"
            "- available=true only if the paper explicitly states the dataset is publicly downloadable.\n"
            "- available=false if the paper says restricted, proprietary, or available only on request.\n"
            "- available=null if the paper says nothing about availability.\n"
            "- url must be copied verbatim from the paper — do NOT guess or construct URLs.\n"
            "Return ONLY the JSON object, no markdown."
        )

        updated = 0
        skipped = 0
        errors = 0

        for r in candidates:
            dataset_id = r["dataset_id"]
            dataset_name = r["canonical_name"]
            paper_id = r["paper_id"]
            title = r["title"] or ""
            text = r["pdf_text"] or r["abstract"] or ""

            if not text:
                skipped += 1
                continue

            # Cache key: scoped to dataset name + paper content so prompt
            # tweaks don't silently serve stale answers.
            text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
            name_hash = hashlib.sha256(dataset_name.encode()).hexdigest()[:8]
            cache_key = f"fa:v1:{name_hash}:{paper_id}:{text_hash}"

            cached = cache.get_llm(cache_key)
            if cached is None:
                prompt = _AVAIL_PROMPT.format(
                    dataset_name=dataset_name,
                    title=title,
                    text=text[:80_000],
                )
                try:
                    response = router.complete(
                        LlmRequest(
                            task="fill_availability",
                            schema_name="AvailabilityCheck",
                            prompt=prompt,
                            metadata={"dataset_id": dataset_id, "paper_id": paper_id},
                        )
                    )
                    parsed = response.parsed
                    if not isinstance(parsed, dict):
                        raise ValueError(f"Non-dict response: {response.content[:200]}")
                except Exception as exc:
                    typer.echo(f"  ERROR {dataset_name[:40]}: {exc}", err=True)
                    errors += 1
                    continue
                cache.set_llm(cache_key, parsed)
            else:
                parsed = cached

            raw_avail = parsed.get("available")
            url = str(parsed.get("url") or "").strip()
            notes = str(parsed.get("notes") or "").strip()

            # Live-check any URL the LLM found
            if url:
                is_live = _check_url_live(url)
                avail_status = "open" if is_live else (
                    "open" if raw_avail is True else "unknown"
                )
            elif raw_avail is True:
                avail_status = "open"
            elif raw_avail is False:
                avail_status = "closed"
            else:
                skipped += 1
                continue  # Still unknown — don't write anything

            # Update datasets table
            conn.execute(
                """UPDATE datasets
                   SET availability_status = ?,
                       availability_url    = COALESCE(NULLIF(?, ''), availability_url),
                       availability_notes  = COALESCE(NULLIF(?, ''), availability_notes)
                   WHERE id = ?""",
                (avail_status, url or None, notes or None, dataset_id),
            )
            # Update all claims for this dataset
            conn.execute(
                "UPDATE paper_analysis_dataset_claims SET availability_status = ? WHERE dataset_id = ?",
                (avail_status, dataset_id),
            )
            conn.commit()

            src = "PDF" if r["pdf_text"] else "abstract"
            url_disp = f" → {url[:50]}" if url else ""
            typer.echo(f"  [{avail_status.upper():7}] [{src}] {dataset_name[:45]}{url_disp}")
            updated += 1

        typer.echo(
            f"\nDone: {updated} updated, {skipped} still unknown, {errors} errors."
        )
