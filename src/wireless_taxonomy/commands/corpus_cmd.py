"""CLI command: corpus — list, status, switch, and rollback corpora."""

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
    @app.command("corpus")
    def corpus_cmd(
        action: str = typer.Argument(
            "status",
            help="Action: status | list | new | use | snapshots | rollback",
        ),
        name: Optional[str] = typer.Argument(
            None,
            help="Corpus name (for 'use', 'new') or snapshot timestamp (for 'rollback').",
        ),
    ) -> None:
        """Manage corpora — versioned, isolated corpus directories.

        \b
        A corpus bundles one DB + results dir + metadata under corpora/<name>/.
        Corpora are auto-named corpus_v1, corpus_v2, ... unless you name them.

        \b
        Actions:
          status              Show the active corpus and its metadata.
          list                List all corpora.
          new [name]          Create (and switch to) a new corpus.
          use <name>          Switch the active corpus.
          snapshots           List rollback snapshots for the active corpus.
          rollback <stamp>    Restore the active corpus DB from a snapshot.

        \b
        Examples:
          corpus status
          corpus new                  # creates corpus_v2 (auto-named)
          corpus new mobicom_2025     # explicit name
          corpus use corpus_v1
          corpus rollback 20250601
        """
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
            c = resolve_corpus(name or next_auto_name(), create=True)
            set_active(c.name)
            typer.echo(f"Created and switched to corpus: {c.name}")
            typer.echo(f"  DB:      {c.db_path}")
            typer.echo(f"  Results: {c.results_dir}")

        elif action == "use":
            if not name:
                typer.echo("Usage: corpus use <name>", err=True)
                raise typer.Exit(1)
            c = Corpus(name)
            if not c.exists():
                typer.echo(f"Corpus '{name}' does not exist. See: corpus list", err=True)
                raise typer.Exit(1)
            set_active(name)
            typer.echo(f"Active corpus: {name}")

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
                f"Unknown action '{action}'. Use: status | list | new | use | snapshots | rollback",
                err=True,
            )
            raise typer.Exit(1)
