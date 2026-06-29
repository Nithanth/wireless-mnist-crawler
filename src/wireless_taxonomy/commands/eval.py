
from pathlib import Path
from typing import Optional

import typer

from wireless_taxonomy.commands._shared import parse_venue_years


def register(app: typer.Typer) -> None:
    @app.command()
    def eval(
        classified: list[str] = typer.Option(..., "--classified", help="Full labelled CSV from `classify --csv`; repeatable."),
        gold: list[str] = typer.Option(..., "--gold", help="Curated gold sheet (csv/xlsx); repeatable."),
        pass_mode: str = typer.Option("high", "--pass", help="high = label 'yes' only; low = 'yes' or 'maybe'."),
        drop_workshops: bool = typer.Option(
            False,
            "--drop-workshops/--keep-workshops",
            help="Drop curated papers absent from the classified universe (co-located workshops) from the calculation.",
        ),
        fuzzy_threshold: float = typer.Option(0.92, "--fuzzy-threshold", help="Title fuzzy-match ratio; 1.0 disables fuzzy."),
        exclude: list[str] = typer.Option(
            [], "--exclude", help="Venue-year to drop from the headline (VENUE:YEAR, e.g. IMC:2025); repeatable."
        ),
        min_gold: int = typer.Option(
            0, "--min-gold", help="Report venue-years with fewer than N curated gold papers separately (0 = off)."
        ),
        out: Optional[str] = typer.Option(None, "--out", help="Write the JSON report here."),
        md: Optional[str] = typer.Option(None, "--md", help="Write the Markdown report here."),
    ) -> None:
        """DB-free snapshot eval: score classified CSV(s) against gold sheet(s).

        No database or network — pure file-in, metrics-out. Matches DOI → exact
        title → fuzzy title per (venue, year), and scores only the conferences
        present in the classified CSV(s). ``--drop-workshops`` excludes curated
        papers absent from the classified universe (the full set written by
        ``classify --csv``) instead of counting them as misses.

        ``--exclude VENUE:YEAR`` and ``--min-gold N`` pull thinly- or stale-curated
        venue-years out of the overall metrics (reported separately) so they don't
        drag the headline — useful when a conference was curated before its papers
        were released.
        """
        if pass_mode not in {"high", "low"}:
            raise typer.BadParameter("--pass must be 'high' or 'low'.")
        if min_gold < 0:
            raise typer.BadParameter("--min-gold must be >= 0.")
        for label, paths in (("--classified", classified), ("--gold", gold)):
            for path in paths:
                if not Path(path).exists():
                    raise typer.BadParameter(f"{label} file not found: {path}")
        exclude_pairs = parse_venue_years(list(exclude))
        from wireless_taxonomy.eval.standalone import eval_files_to_outputs

        try:
            report = eval_files_to_outputs(
                list(classified),
                list(gold),
                json_out=out,
                md_out=md,
                pass_mode=pass_mode,
                drop_workshops=drop_workshops,
                fuzzy_threshold=fuzzy_threshold,
                exclude=exclude_pairs,
                min_gold=min_gold,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

        scope_note = "main-proceedings only" if drop_workshops else "workshops included"
        typer.echo(f"snapshot eval: pass={pass_mode} fuzzy={fuzzy_threshold} scope={scope_note}")
        if not report["instances"] and not (report.get("under_curated_instances")):
            typer.echo(
                "no venue-years scored — the classified CSV(s) and gold sheet share no "
                "(venue, year). Check the venue/year columns match.",
                err=True,
            )
        for row in report["instances"]:
            typer.echo(
                f"- {row['venue']} {row['year']}: jaccard={row['jaccard']} "
                f"precision={row['precision']} recall={row['recall']} f1={row['f1']} "
                f"(tp={row['tp']} fp={row['fp']} fn={row['fn']})"
            )
        overall = report["overall"]
        typer.echo(
            f"OVERALL: jaccard={overall['jaccard']} precision={overall['precision']} "
            f"recall={overall['recall']} f1={overall['f1']} "
            f"(TP {overall['tp']} / FP {overall['fp']} / FN {overall['fn']})"
        )
        under = report.get("under_curated_instances") or []
        if under:
            typer.echo("under-curated / excluded (not in headline):")
            for r in under:
                typer.echo(
                    f"- {r['venue']} {r['year']}: gold={r.get('gold_papers')} [{r.get('reason')}] "
                    f"would be precision={r['precision']} recall={r['recall']} f1={r['f1']} "
                    f"(tp={r['tp']} fp={r['fp']} fn={r['fn']})"
                )
        ignored = report.get("ignored_gold_instances") or []
        if ignored:
            pairs = ", ".join(f"{r['venue']}:{r['year']}({r['gold_papers']})" for r in ignored)
            typer.echo(f"ignored gold venue-years not in classified set: {pairs}", err=True)
        if out:
            typer.echo(f"Wrote JSON: {out}")
        if md:
            typer.echo(f"Wrote Markdown: {md}")
