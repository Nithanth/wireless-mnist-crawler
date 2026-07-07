"""CLI commands: init, status, rollback (top-level) + corpus management."""

from __future__ import annotations

from typing import Optional

import typer

from wireless_taxonomy.corpus import (
    Corpus,
    active_corpus,
    list_corpora,
    next_auto_name,
    resolve_corpus,
    set_active,
)


def register(app: typer.Typer) -> None:
    # ── Top-level commands ─────────────────────────────────────────────────────

    @app.command("init")
    def init(
        name: Optional[str] = typer.Argument(
            None,
            help="Corpus name. Auto-named (corpus_v1, corpus_v2, ...) if omitted.",
        ),
        adopt: bool = typer.Option(
            False, "--adopt",
            help="Migrate legacy repo-root layout (taxonomy.sqlite + src/results/) into the corpus.",
        ),
    ) -> None:
        """Create a new corpus (workspace) or adopt existing data.

        \b
        A corpus bundles one DB + results dir + metadata under corpora/<name>/.

        \b
        Examples:
          wt init                      # auto-named corpus_v1
          wt init wireless_v1          # explicit name
          wt init wireless_v1 --adopt  # migrate existing taxonomy.sqlite + src/results/
        """
        if adopt:
            from wireless_taxonomy.corpus import adopt_legacy

            adopt_name = name or next_auto_name()
            model_identity = ""
            try:
                from wireless_taxonomy.config import load_settings
                from wireless_taxonomy.llm import LlmRouter

                p = LlmRouter(load_settings("taxonomy.sqlite").llm).select_provider()
                model_identity = f"{p.provider}/{p.model}"
            except Exception:
                pass
            typer.echo(
                f"Migrating taxonomy.sqlite + src/results/ into corpora/{adopt_name}/"
            )
            try:
                c = adopt_legacy(adopt_name, model_identity=model_identity)
            except (FileExistsError, FileNotFoundError, ValueError) as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(1)
            typer.echo(f"Done. Active corpus: {c.name}")
            typer.echo(f"  DB:      {c.db_path}")
            typer.echo(f"  Results: {c.results_dir}")
        else:
            try:
                c = resolve_corpus(name or next_auto_name(), create=True)
            except ValueError as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(1)
            set_active(c.name)
            typer.echo(f"Created corpus: {c.name}")
            typer.echo(f"  DB:      {c.db_path}")
            typer.echo(f"  Results: {c.results_dir}")

    @app.command("status")
    def status() -> None:
        """Show what's in the active corpus: venues, papers, datasets.
        """
        import sqlite3

        c = active_corpus()
        if c is None:
            typer.echo(
                "No active corpus. Create one with: wt init"
            )
            raise typer.Exit()

        meta = c.read_meta()
        typer.echo(f"Corpus: {c.name}")
        typer.echo(f"  Model:   {meta.get('model_identity') or '(not yet recorded)'}")
        typer.echo(f"  Created: {meta.get('created_at', '?')}")

        # Show per-stage models if used
        runs = meta.get("runs", [])
        stage_models: set[str] = set()
        for r in runs:
            if r.get("classify_model"):
                stage_models.add(f"classify={r['classify_model']}")
            if r.get("extract_model"):
                stage_models.add(f"extract={r['extract_model']}")
        if stage_models:
            typer.echo(f"  Stage models: {', '.join(sorted(stage_models))}")

        if not c.db_path.exists():
            typer.echo("\n  (empty — run `wt add` to extract data)")
            raise typer.Exit()

        from wireless_taxonomy.db import connect
        conn = connect(str(c.db_path))
        conn.row_factory = sqlite3.Row

        # Per-venue/year breakdown with coverage + extraction stats
        # Use paper_text_artifacts for real PDF coverage (pdf_url is only
        # set by the newer fetch-coverage pipeline; legacy runs fetched PDFs
        # without populating that column).
        rows = conn.execute("""
            SELECT v.name AS venue, ci.year,
                   COUNT(p.id) AS papers,
                   SUM(CASE WHEN pta.fetch_status = 'ok' OR p.pdf_url IS NOT NULL THEN 1 ELSE 0 END) AS has_pdf,
                   SUM(CASE WHEN (pta.fetch_status IS NULL OR pta.fetch_status != 'ok')
                            AND p.pdf_url IS NULL
                            AND p.abstract IS NOT NULL THEN 1 ELSE 0 END) AS abstract_only
            FROM papers p
            JOIN conference_instances ci ON ci.id = p.conference_instance_id
            JOIN venues v ON v.id = ci.venue_id
            LEFT JOIN paper_text_artifacts pta ON pta.paper_id = p.id
            GROUP BY v.name, ci.year
            ORDER BY v.name, ci.year
        """).fetchall()

        if not rows:
            typer.echo("\n  (empty — run `wt add` to extract data)")
            conn.close()
            raise typer.Exit()

        # Dataset counts per venue/year
        ds_counts = {}
        ds_rows = conn.execute("""
            SELECT v.name AS venue, ci.year, COUNT(DISTINCT c.dataset_id) AS datasets
            FROM paper_analysis_dataset_claims c
            JOIN papers p ON p.id = c.paper_id
            JOIN conference_instances ci ON ci.id = p.conference_instance_id
            JOIN venues v ON v.id = ci.venue_id
            GROUP BY v.name, ci.year
        """).fetchall()
        for dr in ds_rows:
            ds_counts[(dr["venue"], dr["year"])] = dr["datasets"]

        n_datasets = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        n_claims = conn.execute("SELECT COUNT(*) FROM paper_analysis_dataset_claims").fetchone()[0]
        total_papers = sum(r["papers"] for r in rows)
        total_pdfs = sum(r["has_pdf"] for r in rows)

        typer.echo(f"\n  {total_papers} papers | {total_pdfs} PDFs ({100*total_pdfs//total_papers}%) | {n_datasets} datasets | {n_claims} claims")
        typer.echo(f"\n  {'Venue':<12} {'Year':<6} {'Papers':<8} {'PDFs':<8} {'Abstract':<10} {'Datasets'}")
        typer.echo(f"  {'─'*12} {'─'*6} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")
        for r in rows:
            pct = f"({100*r['has_pdf']//r['papers']}%)" if r['papers'] > 0 else ""
            ds = ds_counts.get((r["venue"], r["year"]), 0)
            typer.echo(
                f"  {r['venue']:<12} {r['year']:<6} {r['papers']:<8} "
                f"{r['has_pdf']:<4}{pct:<4} {r['abstract_only']:<10} {ds}"
            )

        snaps = c.list_snapshots()
        if snaps:
            typer.echo(f"\n  Snapshots: {len(snaps)} (latest: {snaps[-1].stem})")

        if runs:
            typer.echo(f"\n  Last run: {runs[-1].get('at', '?')} — {runs[-1].get('venues_years', '?')}")
        conn.close()

    @app.command("use")
    def use(
        name: Optional[str] = typer.Argument(
            None,
            help="Corpus name to switch to. Omit to list available corpora.",
        ),
    ) -> None:
        """Switch the active corpus (or list all corpora).

        \b
        Examples:
          wt use                  # list available corpora
          wt use wireless_v1     # switch to wireless_v1
        """
        if name is None:
            corpora = list_corpora()
            if not corpora:
                typer.echo("No corpora. Create one with: wt init")
                raise typer.Exit()
            active = active_corpus()
            for c in corpora:
                meta = c.read_meta()
                marker = "*" if active and c.name == active.name else " "
                model = meta.get("model_identity") or "-"
                typer.echo(f"{marker} {c.name:<20} model={model}  created={meta.get('created_at', '?')}")
            raise typer.Exit()

        try:
            c = Corpus(name)
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1)
        if not c.exists():
            typer.echo(f"Corpus '{name}' not found. See: wt use", err=True)
            raise typer.Exit(1)
        set_active(name)
        typer.echo(f"Switched to: {name}")

    @app.command("rollback")
    def rollback(
        snapshot: Optional[str] = typer.Argument(
            None,
            help="Snapshot timestamp (from `wt status`). Omit to see available snapshots.",
        ),
    ) -> None:
        """Undo the last run by restoring a DB snapshot.

        \b
        Examples:
          wt rollback                  # list available snapshots
          wt rollback 20250601_120000  # restore specific snapshot
        """
        c = active_corpus()
        if c is None:
            typer.echo("No active corpus.", err=True)
            raise typer.Exit(1)

        if snapshot is None:
            snaps = c.list_snapshots()
            if not snaps:
                typer.echo(f"No snapshots for {c.name}.")
                raise typer.Exit()
            typer.echo(f"Available snapshots for {c.name}:")
            for s in snaps:
                size_mb = s.stat().st_size / 1_048_576
                typer.echo(f"  {s.stem}  ({size_mb:.1f} MB)")
            typer.echo(f"\nUsage: wt rollback {snaps[-1].stem}")
            raise typer.Exit()

        try:
            restored = c.rollback(snapshot)
        except FileNotFoundError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1)
        typer.echo(f"Rolled back to: {restored.stem}")
        typer.echo("(Pre-rollback state was snapshotted — rollback is reversible.)")

    # ── Legacy `corpus` subcommand (kept for backward compat) ─────────────────

    @app.command("corpus", hidden=True)
    def corpus_cmd(
        action: str = typer.Argument(
            "status",
            help="Action: status | list | new | use | adopt | snapshots | rollback",
        ),
        name: Optional[str] = typer.Argument(
            None,
            help="Corpus name or snapshot timestamp.",
        ),
    ) -> None:
        """[Legacy] Manage corpora. Use `wt init`, `wt status`, `wt rollback` instead."""
        if action == "status":
            c = active_corpus()
            if c is None:
                typer.echo(
                    "No active corpus (legacy mode: taxonomy.sqlite + src/results/).\n"
                    f"Create one with: corpus new  (next auto-name: {next_auto_name()})"
                )
                return
            meta = c.read_meta()
            typer.echo(f"Active corpus: {c.name}")
            typer.echo(f"  DB:       {c.db_path}")
            typer.echo(f"  Results:  {c.results_dir}")
            typer.echo(f"  Created:  {meta.get('created_at', '?')}")
            typer.echo(f"  Model:    {meta.get('model_identity') or '(not yet recorded)'}")
            runs = meta.get("runs", [])
            if runs:
                typer.echo(f"  Runs ({len(runs)}):")
                for r in runs[-5:]:
                    typer.echo(f"    {r.get('at', '?')}  {r.get('venues_years', '?')}")
            snaps = c.list_snapshots()
            typer.echo(f"  Snapshots: {len(snaps)}")

        elif action == "list":
            corpora = list_corpora()
            if not corpora:
                typer.echo("No corpora yet. Create one with: corpus new")
                return
            active = active_corpus()
            for c in corpora:
                meta = c.read_meta()
                marker = "*" if active and c.name == active.name else " "
                model = meta.get("model_identity") or "-"
                typer.echo(f"{marker} {c.name:<20} model={model}  created={meta.get('created_at', '?')}")

        elif action == "new":
            try:
                c = resolve_corpus(name or next_auto_name(), create=True)
            except ValueError as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(1)
            set_active(c.name)
            typer.echo(f"Created and switched to corpus: {c.name}")
            typer.echo(f"  DB:      {c.db_path}")
            typer.echo(f"  Results: {c.results_dir}")

        elif action == "use":
            if not name:
                typer.echo("Usage: corpus use <name>", err=True)
                raise typer.Exit(1)
            try:
                c = Corpus(name)
            except ValueError as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(1)
            if not c.exists():
                typer.echo(f"Corpus '{name}' does not exist. See: corpus list", err=True)
                raise typer.Exit(1)
            set_active(name)
            typer.echo(f"Active corpus: {name}")

        elif action == "adopt":
            from wireless_taxonomy.corpus import adopt_legacy
            from wireless_taxonomy.corpus import next_auto_name as _next

            adopt_name = name or _next()
            # Record the current model so the model-change guard works from day one.
            model_identity = ""
            try:
                from wireless_taxonomy.config import load_settings
                from wireless_taxonomy.llm import LlmRouter

                p = LlmRouter(load_settings("taxonomy.sqlite").llm).select_provider()
                model_identity = f"{p.provider}/{p.model}"
            except Exception:
                pass
            typer.echo("This will MOVE taxonomy.sqlite and src/results/* into "
                       f"corpora/{adopt_name}/ (one-time migration).")
            typer.confirm("Proceed?", abort=True)
            try:
                c = adopt_legacy(adopt_name, model_identity=model_identity)
            except (FileExistsError, FileNotFoundError, ValueError) as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(1)
            typer.echo(f"Adopted legacy data into corpus: {c.name}")
            typer.echo(f"  DB:      {c.db_path}")
            typer.echo(f"  Results: {c.results_dir}")
            typer.echo(f"  Model:   {model_identity or '(unknown)'}")
            typer.echo(f"\n'{c.name}' is now the active corpus. Future runs of "
                       "./run_loop.sh automatically use it.")

        elif action == "snapshots":
            c = active_corpus()
            if c is None:
                typer.echo("No active corpus.", err=True)
                raise typer.Exit(1)
            snaps = c.list_snapshots()
            if not snaps:
                typer.echo(f"No snapshots for {c.name}. One is taken before each batch run.")
                return
            typer.echo(f"Snapshots for {c.name}:")
            for s in snaps:
                size_mb = s.stat().st_size / 1_048_576
                typer.echo(f"  {s.stem}  ({size_mb:.1f} MB)")

        elif action == "rollback":
            if not name:
                typer.echo("Usage: corpus rollback <timestamp>  (see: corpus snapshots)", err=True)
                raise typer.Exit(1)
            c = active_corpus()
            if c is None:
                typer.echo("No active corpus.", err=True)
                raise typer.Exit(1)
            try:
                restored = c.rollback(name)
            except FileNotFoundError as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(1)
            typer.echo(f"Rolled back {c.name} DB to snapshot {restored.stem}.")
            typer.echo("(The pre-rollback state was itself snapshotted — rollback is reversible.)")

        else:
            typer.echo(
                f"Unknown action '{action}'. Use: status | list | new | use | adopt | snapshots | rollback",
                err=True,
            )
            raise typer.Exit(1)
