"""Admin commands: llm-config, corpus-status, prune."""

import sqlite3
from typing import Optional

import typer

from wireless_taxonomy.config import load_settings


def register(app: typer.Typer) -> None:
    @app.command("llm-config")
    def llm_config(db: str = typer.Option("taxonomy.sqlite", "--db")) -> None:
        settings = load_settings(db)
        typer.echo(f"Primary provider: {settings.llm.primary_provider}")
        fallbacks = ", ".join(settings.llm.fallback_providers) if settings.llm.fallback_providers else "(none)"
        typer.echo(f"Fallback providers: {fallbacks}")
        for provider in settings.llm.ordered_providers():
            key_status = "configured" if provider.api_key_configured else f"missing {provider.api_key_env}"
            typer.echo(f"- {provider.provider}: model={provider.model}, key={key_status}")

    @app.command("corpus-status")
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

    @app.command("prune")
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
