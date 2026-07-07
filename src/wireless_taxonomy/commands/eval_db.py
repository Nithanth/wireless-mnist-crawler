"""The `eval-db` command: eval corpus DB against curated gold sheets.

For each venue/year in the gold sheet, pull papers from the corpus DB and
compare wireless labels. Reports mismatches and precision/recall/F1.
"""

from pathlib import Path
from typing import Optional

import typer


def register(app: typer.Typer) -> None:
    @app.command("eval-db")
    def eval_db(
        gold: list[str] = typer.Option(..., "--gold", help="Curated gold sheet (csv/xlsx); repeatable."),
        pass_mode: str = typer.Option("high", "--pass", help="high = label 'yes' only; low = 'yes' or 'maybe'."),
        fuzzy_threshold: float = typer.Option(0.92, "--fuzzy-threshold", help="Title fuzzy-match ratio; 1.0 disables fuzzy."),
    ) -> None:
        """Eval corpus DB against curated gold sheets.

        For each venue/year in the gold sheet, queries the corpus DB for papers
        and compares wireless labels. Reports mismatches and precision/recall/F1.

        \b
        Examples:
          wt advanced eval-db --gold gold_sheets/SIGCOMM_2023.csv
          wt advanced eval-db --gold gold_sheets/*.csv --pass low
        """
        if pass_mode not in {"high", "low"}:
            raise typer.BadParameter("--pass must be 'high' or 'low'.")

        for path in gold:
            if not Path(path).exists():
                raise typer.BadParameter(f"--gold file not found: {path}")

        from wireless_taxonomy.corpus import active_corpus
        from wireless_taxonomy.db import connect
        from wireless_taxonomy.ingest.gold import GoldSheetReader
        from wireless_taxonomy.eval.overlap import PaperRef, aggregate, match, to_markdown

        corpus_obj = active_corpus()
        if corpus_obj is None:
            typer.echo("No active corpus. Create one with: wt init", err=True)
            raise typer.Exit(1)

        db = str(corpus_obj.db_path)
        conn = connect(db)
        import sqlite3
        conn.row_factory = sqlite3.Row

        # Load gold sheets
        gold_papers: list[PaperRef] = []
        for gpath in gold:
            reader = GoldSheetReader(Path(gpath))
            for row in reader.rows():
                gold_papers.append(PaperRef(
                    title=row.title,
                    doi=row.doi or "",
                    venue=row.venue,
                    year=str(row.year),
                    label="yes",  # gold sheets are curated as wireless
                ))

        if not gold_papers:
            typer.echo("No gold papers loaded.", err=True)
            raise typer.Exit(1)

        # Group gold by (venue, year)
        from collections import defaultdict
        gold_by_vy: dict[tuple[str, str], list[PaperRef]] = defaultdict(list)
        for p in gold_papers:
            gold_by_vy[(p.venue, p.year)].append(p)

        # For each venue/year, query DB and match
        all_matches = []
        mismatched = []

        for (venue, year), gold_list in sorted(gold_by_vy.items()):
            # Query DB for this venue/year
            db_rows = conn.execute("""
                SELECT title, doi, wireless_label
                FROM papers p
                JOIN conference_instances ci ON ci.id = p.conference_instance_id
                JOIN venues v ON v.id = ci.venue_id
                WHERE v.name = ? AND ci.year = ?
            """, (venue, year)).fetchall()

            db_papers = [
                PaperRef(
                    title=row["title"] or "",
                    doi=row["doi"] or "",
                    venue=venue,
                    year=year,
                    label=row["wireless_label"] or "no",
                )
                for row in db_rows
            ]

            # Match gold vs DB
            matches = match(
                gold_list,
                db_papers,
                fuzzy_threshold=fuzzy_threshold,
            )

            for m in matches:
                all_matches.append(m)
                # Track mismatches
                if m.gold.label != m.pred.label:
                    mismatched.append(m)

        if not all_matches:
            typer.echo("No matches found. Check venue/year names match between gold and DB.", err=True)
            conn.close()
            raise typer.Exit(1)

        # Compute metrics
        pass_labels = {"high": {"yes"}, "low": {"yes", "maybe"}}[pass_mode]
        metrics = aggregate(all_matches, pass_labels=pass_labels)

        # Report
        typer.echo(f"Eval against DB: pass={pass_mode} fuzzy={fuzzy_threshold}")
        typer.echo(f"Gold papers: {len(gold_papers)}")
        typer.echo(f"Matched: {len(all_matches)}")
        typer.echo(f"Mismatches: {len(mismatched)}")
        typer.echo("")

        if mismatched:
            typer.echo("Mismatches:")
            for m in mismatched[:20]:  # show first 20
                typer.echo(
                    f"  [{m.venue} {m.year}] {m.gold.title[:60]}... "
                    f"gold={m.gold.label} db={m.pred.label}"
                )
            if len(mismatched) > 20:
                typer.echo(f"  ... and {len(mismatched) - 20} more")
            typer.echo("")

        # Per-venue/year metrics
        vy_metrics: dict[tuple[str, str], list] = defaultdict(list)
        for m in all_matches:
            vy_metrics[(m.venue, m.year)].append(m)

        typer.echo("Per venue/year:")
        for (venue, year), vy_matches in sorted(vy_metrics.items()):
            vy_agg = aggregate(vy_matches, pass_labels=pass_labels)
            overall = vy_agg["overall"]
            typer.echo(
                f"  {venue} {year}: jaccard={overall['jaccard']:.3f} "
                f"precision={overall['precision']:.3f} recall={overall['recall']:.3f} "
                f"f1={overall['f1']:.3f} (tp={overall['tp']} fp={overall['fp']} fn={overall['fn']})"
            )

        # Overall metrics
        overall = metrics["overall"]
        typer.echo("")
        typer.echo(
            f"OVERALL: jaccard={overall['jaccard']:.3f} precision={overall['precision']:.3f} "
            f"recall={overall['recall']:.3f} f1={overall['f1']:.3f} "
            f"(TP {overall['tp']} / FP {overall['fp']} / FN {overall['fn']})"
        )

        conn.close()
