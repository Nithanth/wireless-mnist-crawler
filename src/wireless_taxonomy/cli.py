from __future__ import annotations

import csv as _csv
import inspect
import json
from pathlib import Path
from typing import List, Optional

import click
import typer
from typer.core import TyperArgument, TyperOption

from wireless_taxonomy.config import load_settings
from wireless_taxonomy.pipeline import Pipeline

_OPTION_MAKE_METAVAR = TyperOption.make_metavar
_ARGUMENT_MAKE_METAVAR = TyperArgument.make_metavar
_CLICK_PARAMETER_MAKE_METAVAR = click.core.Parameter.make_metavar
_CLICK_OPTION_MAKE_METAVAR = click.core.Option.make_metavar
_CLICK_ARGUMENT_MAKE_METAVAR = click.core.Argument.make_metavar


def _patch_typer_click_compat() -> None:
    """Typer 0.15.x rich help calls make_metavar without Click 8.2's ctx."""

    for cls, original in [
        (click.core.Parameter, _CLICK_PARAMETER_MAKE_METAVAR),
        (click.core.Option, _CLICK_OPTION_MAKE_METAVAR),
        (click.core.Argument, _CLICK_ARGUMENT_MAKE_METAVAR),
    ]:
        params = inspect.signature(cls.make_metavar).parameters
        if params.get("ctx") is not None and params["ctx"].default is inspect.Parameter.empty:

            def make_metavar(self, ctx=None, _original=original):
                return _original(self, ctx)

            cls.make_metavar = make_metavar  # type: ignore[method-assign]

    option_params = inspect.signature(TyperOption.make_metavar).parameters
    if option_params.get("ctx") is not None and option_params["ctx"].default is inspect.Parameter.empty:

        def option_make_metavar(self, ctx=None):
            return _OPTION_MAKE_METAVAR(self, ctx)

        TyperOption.make_metavar = option_make_metavar  # type: ignore[method-assign]

    argument_params = inspect.signature(TyperArgument.make_metavar).parameters
    if argument_params.get("ctx") is None:

        def argument_make_metavar(self, ctx=None):
            if self.metavar is not None:
                return self.metavar
            var = (self.name or "").upper()
            if not self.required:
                var = f"[{var}]"
            type_var = self.type.get_metavar(param=self, ctx=ctx)
            if type_var:
                var += f":{type_var}"
            if self.nargs != 1:
                var += "..."
            return var

        TyperArgument.make_metavar = argument_make_metavar  # type: ignore[method-assign]


_patch_typer_click_compat()

app = typer.Typer(
    help=(
        "Measure automated wireless-paper classification coverage per conference-year.\n\n"
        "Three commands:\n"
        "  classify   loop a venue over a year (or year range), fetch the paper "
        "list, backfill abstracts, label each paper, and print/export the "
        "yes/maybe/no breakdown.\n"
        "  eval       DB-free snapshot scoring of a labelled CSV vs a gold sheet.\n"
        "  llm-config show which LLM providers are configured."
    )
)

_CSV_FIELDS = ["title", "authors", "doi", "venue", "year", "label", "confidence", "used_abstract", "has_abstract"]


def _pipeline(db: str) -> Pipeline:
    return Pipeline(load_settings(db))


def _parse_years(years: str) -> list[int]:
    """Parse ``2024`` or an inclusive range ``2023:2025`` into a list of years."""
    text = years.strip()
    if ":" in text:
        start_s, _, end_s = text.partition(":")
        try:
            start, end = int(start_s), int(end_s)
        except ValueError as exc:
            raise typer.BadParameter("--years range must look like 2023:2025.") from exc
        if start > end:
            raise typer.BadParameter("--years range start must be <= end.")
        return list(range(start, end + 1))
    try:
        return [int(text)]
    except ValueError as exc:
        raise typer.BadParameter("--years must be a year (2024) or range (2023:2025).") from exc


def _pct(count: int, total: int) -> float:
    return round(100.0 * count / total, 1) if total else 0.0


def _echo_breakdown(result: dict) -> None:
    counts = result["counts"]
    total = result["total_papers"]
    with_abs = result["papers_with_abstract"]
    abs_pct = _pct(with_abs, total)
    typer.echo(
        f"{result['venue']} {result['year']} — {total} papers "
        f"(abstracts: {with_abs}/{total}, {abs_pct:.0f}%)"
    )
    for label in ("yes", "maybe", "no"):
        n = counts.get(label, 0)
        typer.echo(f"  {label:<5} {n:>4}  ({_pct(n, total):>5.1f}%)")


@app.command()
def classify(
    venue: str = typer.Option(..., "--venue", help="Conference venue, e.g. NSDI, SIGCOMM, IMC."),
    years: str = typer.Option(..., "--years", help="A year (2024) or inclusive range (2023:2025)."),
    llm: bool = typer.Option(True, "--llm/--no-llm", help="Use the LLM classifier (default) or the keyword baseline."),
    resolve_dois: bool = typer.Option(
        True, "--resolve-dois/--no-resolve-dois", help="Backfill missing DOIs from title before classifying."
    ),
    source: str = typer.Option("dblp", "--source", help="Paper-list source: dblp (default), bibtex, csv, or url."),
    source_value: Optional[str] = typer.Option(None, "--source-value", help="Path or URL when --source is not dblp."),
    json_out: Optional[str] = typer.Option(None, "--json", help="Write the full labelled set (all years) as JSON here."),
    csv_out: Optional[str] = typer.Option(None, "--csv", help="Write the full labelled set (all years) as CSV here."),
    cache: bool = typer.Option(
        True, "--cache/--no-cache", help="Read/write resolved abstracts+DOIs to a disk index for fast, deterministic re-runs."
    ),
    cache_path: str = typer.Option(
        ".wt_cache.json", "--cache-path", help="Where the abstract/DOI/LLM cache lives (used unless --no-cache)."
    ),
    refresh_llm: bool = typer.Option(
        False, "--refresh-llm", help="Ignore cached LLM labels and re-call the model (fresh classification)."
    ),
    db: str = typer.Option("taxonomy.sqlite", "--db", help="SQLite work DB (created/reused)."),
) -> None:
    """Loop a venue over a year (or range): fetch list, backfill abstracts, label.

    Runs the full per-year pipeline (DBLP list -> DOI/abstract backfill ->
    yes/maybe/no classification).
    Prints a yes/maybe/no breakdown (counts + % of the conference set) per year
    and, for a range, an aggregate. ``--csv``/``--json`` export the **full**
    labelled set (every paper, not just the wireless ones), which is exactly what
    ``eval`` consumes.
    """
    if source != "dblp" and not source_value:
        raise typer.BadParameter("--source-value is required when --source is not 'dblp'.")
    year_list = _parse_years(years)

    metadata_cache = None
    if cache:
        from wireless_taxonomy.analyze.cache import MetadataCache

        metadata_cache = MetadataCache(cache_path)

    results: list[dict] = []
    pipeline = _pipeline(db)
    try:
        for year in year_list:
            results.append(
                pipeline.classify_conference(
                    venue,
                    year,
                    use_llm=llm,
                    resolve_dois=resolve_dois,
                    source_type=source,
                    source_value=source_value,
                    cache=metadata_cache,
                    refresh_llm=refresh_llm,
                )
            )
    finally:
        pipeline.close()
        if metadata_cache is not None:
            metadata_cache.save()
            stats = metadata_cache.stats()
            typer.echo(
                f"Cache: {stats['abstracts']} abstracts, {stats['dois']} DOIs, "
                f"{stats.get('llm', 0)} LLM labels at {cache_path}"
            )

    all_papers = [paper for result in results for paper in result["papers"]]

    for result in results:
        _echo_breakdown(result)
        typer.echo("")

    if len(results) > 1:
        agg_counts = {"yes": 0, "maybe": 0, "no": 0}
        agg_total = 0
        agg_abs = 0
        for result in results:
            for label, n in result["counts"].items():
                agg_counts[label] = agg_counts.get(label, 0) + n
            agg_total += result["total_papers"]
            agg_abs += result["papers_with_abstract"]
        span = f"{year_list[0]}–{year_list[-1]}"
        typer.echo(
            f"{venue} {span} (aggregate) — {agg_total} papers "
            f"(abstracts: {agg_abs}/{agg_total}, {_pct(agg_abs, agg_total):.0f}%)"
        )
        for label in ("yes", "maybe", "no"):
            n = agg_counts.get(label, 0)
            typer.echo(f"  {label:<5} {n:>4}  ({_pct(n, agg_total):>5.1f}%)")

    if json_out:
        out_path = Path(json_out)
        if out_path.suffix.lower() != ".json":
            out_path = out_path.with_suffix(".json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"venue": venue, "years": year_list, "runs": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        typer.echo(f"Wrote JSON: {out_path}")
    if csv_out:
        csv_path = Path(csv_out)
        if csv_path.suffix.lower() != ".csv":
            csv_path = csv_path.with_suffix(".csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(all_papers)
        typer.echo(f"Wrote CSV: {csv_path}")


@app.command()
def eval(
    classified: List[str] = typer.Option(..., "--classified", help="Full labelled CSV from `classify --csv`; repeatable."),
    gold: List[str] = typer.Option(..., "--gold", help="Curated gold sheet (csv/xlsx); repeatable."),
    pass_mode: str = typer.Option("high", "--pass", help="high = label 'yes' only; low = 'yes' or 'maybe'."),
    drop_workshops: bool = typer.Option(
        False,
        "--drop-workshops/--keep-workshops",
        help="Drop curated papers absent from the classified universe (co-located workshops) from the calculation.",
    ),
    fuzzy_threshold: float = typer.Option(0.92, "--fuzzy-threshold", help="Title fuzzy-match ratio; 1.0 disables fuzzy."),
    out: Optional[str] = typer.Option(None, "--out", help="Write the JSON report here."),
    md: Optional[str] = typer.Option(None, "--md", help="Write the Markdown report here."),
) -> None:
    """DB-free snapshot eval: score classified CSV(s) against gold sheet(s).

    No database or network — pure file-in, metrics-out. Matches DOI → exact
    title → fuzzy title per (venue, year), and scores only the conferences
    present in the classified CSV(s). ``--drop-workshops`` excludes curated
    papers absent from the classified universe (the full set written by
    ``classify --csv``) instead of counting them as misses.
    """
    if pass_mode not in {"high", "low"}:
        raise typer.BadParameter("--pass must be 'high' or 'low'.")
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
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    scope_note = "main-proceedings only" if drop_workshops else "workshops included"
    typer.echo(f"snapshot eval: pass={pass_mode} fuzzy={fuzzy_threshold} scope={scope_note}")
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
    ignored = report.get("ignored_gold_instances") or []
    if ignored:
        pairs = ", ".join(f"{r['venue']}:{r['year']}({r['gold_papers']})" for r in ignored)
        typer.echo(f"ignored gold venue-years not in classified set: {pairs}", err=True)
    if out:
        typer.echo(f"Wrote JSON: {out}")
    if md:
        typer.echo(f"Wrote Markdown: {md}")


@app.command("llm-config")
def llm_config(db: str = typer.Option("taxonomy.sqlite", "--db")) -> None:
    settings = load_settings(db)
    typer.echo(f"Primary provider: {settings.llm.primary_provider}")
    fallbacks = ", ".join(settings.llm.fallback_providers) if settings.llm.fallback_providers else "(none)"
    typer.echo(f"Fallback providers: {fallbacks}")
    for provider in settings.llm.ordered_providers():
        key_status = "configured" if provider.api_key_configured else f"missing {provider.api_key_env}"
        typer.echo(f"- {provider.provider}: model={provider.model}, key={key_status}")


if __name__ == "__main__":
    app()
