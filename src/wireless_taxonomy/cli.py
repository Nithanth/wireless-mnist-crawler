from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import List, Optional

import click
import typer
from typer.core import TyperArgument, TyperOption

from wireless_taxonomy.config import load_settings
from wireless_taxonomy.evaluate.run_diff import (
    diff_paper_sets,
    format_diff_summary,
    load_paper_set,
    write_diff_csv,
    write_diff_report,
)
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

app = typer.Typer(help="Build accuracy-first wireless paper and dataset taxonomy records.")


def _pipeline(db: str) -> Pipeline:
    return Pipeline(load_settings(db))


@app.command()
def init(db: str = typer.Option("taxonomy.sqlite", "--db", help="SQLite database path.")) -> None:
    pipeline = _pipeline(db)
    try:
        pipeline.init_db()
        typer.echo(f"Initialized database: {Path(db)}")
    finally:
        pipeline.close()


@app.command()
def ingest(
    venue: str = typer.Option(..., "--venue"),
    year: int = typer.Option(..., "--year"),
    url: Optional[str] = typer.Option(None, "--url"),
    bibtex: Optional[str] = typer.Option(None, "--bibtex"),
    csv_path: Optional[str] = typer.Option(None, "--csv"),
    dblp: bool = typer.Option(False, "--dblp", help="Fetch the main-track paper list from DBLP for --venue/--year."),
    db: str = typer.Option("taxonomy.sqlite", "--db"),
) -> None:
    source_type, source_value = _source(url, bibtex, csv_path, dblp)
    pipeline = _pipeline(db)
    try:
        run_id = pipeline.ingest(venue, year, source_type, source_value)
        typer.echo(f"Ingest completed. run_id={run_id}")
    finally:
        pipeline.close()




@app.command("classify-wireless")
def classify_wireless(run_id: int = typer.Option(..., "--run-id"), db: str = typer.Option("taxonomy.sqlite", "--db")) -> None:
    pipeline = _pipeline(db)
    try:
        stage_run = pipeline.classify_wireless(run_id)
        typer.echo(f"Wireless classification completed. run_id={stage_run}")
    finally:
        pipeline.close()


@app.command("enrich-abstracts")
def enrich_abstracts(
    run_id: int = typer.Option(..., "--run-id"),
    overwrite: bool = typer.Option(False, "--overwrite/--missing-only", help="Refetch even papers that already have an abstract."),
    resolve_dois: bool = typer.Option(
        True,
        "--resolve-dois/--no-resolve-dois",
        help="Backfill missing DOIs from title via Crossref/OpenAlex before fetching abstracts.",
    ),
    db: str = typer.Option("taxonomy.sqlite", "--db"),
) -> None:
    pipeline = _pipeline(db)
    try:
        stage_run = pipeline.enrich_abstracts(run_id, overwrite=overwrite, resolve_dois=resolve_dois)
        typer.echo(f"Abstract enrichment completed. run_id={stage_run}")
    finally:
        pipeline.close()


@app.command("classify-candidates")
def classify_candidates(
    run_id: int = typer.Option(..., "--run-id"),
    llm: bool = typer.Option(False, "--llm/--no-llm", help="Use the configured LLM instead of the keyword baseline."),
    db: str = typer.Option("taxonomy.sqlite", "--db"),
) -> None:
    pipeline = _pipeline(db)
    try:
        stage_run = pipeline.classify_candidates(run_id, use_llm=llm)
        typer.echo(f"Candidate classification completed. run_id={stage_run}")
    finally:
        pipeline.close()


@app.command("classify-conference")
def classify_conference(
    venue: str = typer.Option(..., "--venue"),
    year: int = typer.Option(..., "--year"),
    llm: bool = typer.Option(True, "--llm/--no-llm", help="Use the LLM classifier (default) or the keyword baseline."),
    pass_mode: str = typer.Option("high", "--pass", help="high = 'yes' only; low = 'yes'|'maybe'."),
    resolve_dois: bool = typer.Option(True, "--resolve-dois/--no-resolve-dois", help="Backfill missing DOIs before classifying."),
    source: str = typer.Option("dblp", "--source", help="Paper-list source: dblp (default), bibtex, csv, or url."),
    source_value: Optional[str] = typer.Option(None, "--source-value", help="Path or URL when --source is not dblp."),
    out: Optional[str] = typer.Option(None, "--out", help="Write the classified list as JSON here."),
    csv_out: Optional[str] = typer.Option(None, "--csv", help="Write the classified list as CSV here."),
    db: str = typer.Option("taxonomy.sqlite", "--db"),
) -> None:
    """Sheet-free: loop a venue+year through ingest -> DOI/abstract backfill -> classify, and emit the wireless list."""
    if source != "dblp" and not source_value:
        raise typer.BadParameter("--source-value is required when --source is not 'dblp'.")
    pipeline = _pipeline(db)
    try:
        result = pipeline.classify_conference(
            venue,
            year,
            use_llm=llm,
            pass_mode=pass_mode,
            resolve_dois=resolve_dois,
            source_type=source,
            source_value=source_value,
        )
    finally:
        pipeline.close()

    if out:
        out_path = Path(out)
        if out_path.suffix.lower() != ".json":
            out_path = out_path.with_suffix(".json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if csv_out:
        import csv as _csv

        csv_path = Path(csv_out)
        if csv_path.suffix.lower() != ".csv":
            csv_path = csv_path.with_suffix(".csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["title", "authors", "doi", "venue", "year", "wireless_label", "confidence", "used_abstract", "has_abstract"]
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(result["papers"])

    typer.echo(
        f"{result['venue']} {result['year']}: {result['wireless_count']} wireless / "
        f"{result['total_papers']} papers ({result['papers_with_abstract']} with abstracts) "
        f"via {result['classifier']} classifier (--pass {result['pass_mode']})."
    )
    for paper in result["papers"]:
        conf = paper["confidence"]
        typer.echo(f"  [{conf}] {paper['title']}")


@app.command("import-gold")
def import_gold(
    path: str = typer.Option(..., "--path", help="Manual gold sheet (csv or xlsx) of wireless papers."),
    venue: Optional[str] = typer.Option(None, "--venue", help="Default venue if the sheet has no conference column."),
    year: Optional[int] = typer.Option(None, "--year", help="Default year if the sheet has no year column."),
    wireless_only: bool = typer.Option(
        False, "--wireless-only/--all-rows",
        help="If the sheet lists ALL papers with a wireless flag column, keep only flagged-wireless rows.",
    ),
    db: str = typer.Option("taxonomy.sqlite", "--db"),
) -> None:
    pipeline = _pipeline(db)
    try:
        stage_run = pipeline.import_gold(path, venue=venue, year=year, wireless_only=wireless_only)
        typer.echo(f"Gold import completed. run_id={stage_run}")
    finally:
        pipeline.close()


@app.command("gold-venues")
def gold_venues(
    path: List[str] = typer.Option(..., "--path", help="Gold sheet(s) (csv/xlsx); repeat for multiple."),
    venue: Optional[str] = typer.Option(None, "--venue", help="Default venue if a sheet has no conference column."),
    year: Optional[int] = typer.Option(None, "--year", help="Default year if a sheet has no year column."),
    ingestable_only: bool = typer.Option(
        True, "--ingestable-only/--all",
        help="Only list venues that resolve to a known DBLP stream (skip e.g. journals).",
    ),
) -> None:
    """List the distinct VENUE:YEAR conferences detected across the gold sheet(s).

    Lets the eval harness (or you) drive the classify loop off whatever a
    dropped-in sheet contains, instead of a hardcoded venue list. Resolvable
    conferences print to stdout as ``VENUE:YEAR``; skipped ones go to stderr.
    """
    from wireless_taxonomy.ingest.dblp import resolve_stream
    from wireless_taxonomy.ingest.gold import distinct_venue_years

    pairs = distinct_venue_years(list(path), default_venue=venue, default_year=year)
    skipped: list[str] = []
    for v, y in pairs:
        if ingestable_only and resolve_stream(v) is None:
            skipped.append(f"{v}:{y}")
            continue
        typer.echo(f"{v}:{y}")
    if skipped:
        typer.echo(
            "skipped (no DBLP stream mapping): " + ", ".join(skipped),
            err=True,
        )


@app.command("eval-overlap")
def eval_overlap(
    classifier: str = typer.Option("keyword", "--classifier", help="Which prediction set to score: keyword or llm."),
    pass_mode: str = typer.Option("high", "--pass", help="high = label yes only; low = label yes or maybe."),
    fuzzy_threshold: float = typer.Option(0.92, "--fuzzy-threshold", help="Title fuzzy-match ratio; 1.0 disables fuzzy."),
    drop_workshops: bool = typer.Option(
        False,
        "--drop-workshops/--keep-workshops",
        help="Drop curated papers absent from the main proceedings (co-located workshops) from the calculation.",
    ),
    out: Optional[str] = typer.Option(None, "--out", help="Optional path to write the full JSON report."),
    md_out: Optional[str] = typer.Option(None, "--md", help="Optional path to write a readable Markdown report."),
    db: str = typer.Option("taxonomy.sqlite", "--db"),
) -> None:
    pipeline = _pipeline(db)
    try:
        report = pipeline.evaluate_overlap(
            classifier=classifier,
            pass_mode=pass_mode,
            fuzzy_threshold=fuzzy_threshold,
            scope_to_universe=drop_workshops,
        )
        overall = report["overall"]
        scope_note = "main-proceedings only" if drop_workshops else "workshops included"
        typer.echo(
            f"Overlap eval: classifier={report['classifier']} pass={report['pass_mode']} "
            f"fuzzy={report['fuzzy_threshold']} scope={scope_note}"
        )
        if not report["instances"]:
            typer.echo("No gold-backed conference instances found. Run import-gold first.")
        for row in report["instances"]:
            typer.echo(
                f"- {row['venue']} {row['year']}: jaccard={row['jaccard']} "
                f"precision={row['precision']} recall={row['recall']} f1={row['f1']} "
                f"(tp={row['tp']} fp={row['fp']} fn={row['fn']}; "
                f"fn_miss={row['fn_missed']} fn_not_ingested={row['fn_missing_from_universe']})"
            )
        for row in report["per_conference"]:
            typer.echo(
                f"= {row['venue']} (all years): jaccard={row['jaccard']} "
                f"precision={row['precision']} recall={row['recall']} f1={row['f1']}"
            )
        typer.echo(
            f"OVERALL: jaccard={overall['jaccard']} precision={overall['precision']} "
            f"recall={overall['recall']} f1={overall['f1']} "
            f"(tp={overall['tp']} fp={overall['fp']} fn={overall['fn']})"
        )
        if out:
            Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            typer.echo(f"Wrote report: {out}")
        if md_out:
            from wireless_taxonomy.eval.overlap import to_markdown

            Path(md_out).write_text(to_markdown(report), encoding="utf-8")
            typer.echo(f"Wrote Markdown report: {md_out}")
    finally:
        pipeline.close()






























@app.command("paper-set")
def paper_set(
    run_id: int = typer.Option(..., "--run-id"),
    out: str = typer.Option(..., "--out"),
    fmt: str = typer.Option("csv", "--format", help="csv or json."),
    wireless_only: bool = typer.Option(
        False, "--wireless-only/--all-papers", help="Restrict to papers the pipeline classified as wireless."
    ),
    wireless_source: str = typer.Option(
        "classify", "--wireless-source", help="Wireless decision source: classify (keyword) or agentic (analysis)."
    ),
    db: str = typer.Option("taxonomy.sqlite", "--db"),
) -> None:
    """Export the conference-scoped set of fetched papers (match_key, title, abstract, ...)."""
    pipeline = _pipeline(db)
    try:
        path = pipeline.export_paper_set(run_id, out, fmt, wireless_only=wireless_only, wireless_source=wireless_source)
        typer.echo(f"Exported paper set ({fmt}): {path}")
    finally:
        pipeline.close()


@app.command("diff-sets")
def diff_sets(
    a: str = typer.Option(..., "--a", help="First paper-set export (csv or json) — e.g. the URL+LLM run."),
    b: str = typer.Option(..., "--b", help="Second paper-set export (csv or json) — e.g. the DBLP+OpenAlex run."),
    label_a: str = typer.Option("A", "--label-a", help="Display label for the first set."),
    label_b: str = typer.Option("B", "--label-b", help="Display label for the second set."),
    fuzzy: bool = typer.Option(
        True,
        "--fuzzy/--exact",
        help="Match near-duplicate titles (difflib + author overlap) vs exact normalized title only.",
    ),
    reference: Optional[str] = typer.Option(
        None,
        "--reference",
        help="Treat one side as ground truth ('a' or 'b') to also report precision/recall/F1.",
    ),
    out: Optional[str] = typer.Option(None, "--out", help="Write the full diff report JSON to this path."),
    csv_out: Optional[str] = typer.Option(
        None, "--csv", help="Write a per-paper diff CSV (status / match type / abstract flags) to this path."
    ),
) -> None:
    """Diff two `paper-set` exports to measure how reliably two automated sources agree.

    Compares two automated paper sets (e.g. a URL+LLM ingest vs a DBLP+OpenAlex
    ingest) and reports their Jaccard overlap, the papers unique to each side, and
    abstract coverage per side. Matching is DOI-first, then exact title, then fuzzy.
    Pass `--reference a|b` to also get precision/recall/F1 against that ground-truth
    side. No database needed — it operates on the exported files.
    """
    if reference is not None and reference not in ("a", "b"):
        raise typer.BadParameter("--reference must be 'a' or 'b'")
    rows_a = load_paper_set(a)
    rows_b = load_paper_set(b)
    summary, diff_rows = diff_paper_sets(
        rows_a, rows_b, fuzzy=fuzzy, label_a=label_a, label_b=label_b, reference=reference
    )
    typer.echo(format_diff_summary(summary))
    if out:
        path = write_diff_report(summary, diff_rows, out)
        typer.echo(f"Wrote diff report: {path}")
    if csv_out:
        path = write_diff_csv(diff_rows, csv_out)
        typer.echo(f"Wrote per-paper diff CSV: {path}")






@app.command()
def status(run_id: Optional[int] = typer.Option(None, "--run-id"), db: str = typer.Option("taxonomy.sqlite", "--db")) -> None:
    pipeline = _pipeline(db)
    try:
        rows = pipeline.status(run_id)
        if not rows:
            typer.echo("No runs found.")
            return
        for row in rows:
            typer.echo(f"#{row['id']} {row['stage']} {row['status']} - {row['message'] or ''}")
    finally:
        pipeline.close()


@app.command("llm-config")
def llm_config(db: str = typer.Option("taxonomy.sqlite", "--db")) -> None:
    settings = load_settings(db)
    typer.echo(f"Primary provider: {settings.llm.primary_provider}")
    fallbacks = ", ".join(settings.llm.fallback_providers) if settings.llm.fallback_providers else "(none)"
    typer.echo(f"Fallback providers: {fallbacks}")
    for provider in settings.llm.ordered_providers():
        key_status = "configured" if provider.api_key_configured else f"missing {provider.api_key_env}"
        typer.echo(f"- {provider.provider}: model={provider.model}, key={key_status}")


def _source(url: str | None, bibtex: str | None, csv_path: str | None, dblp: bool = False) -> tuple[str, str]:
    provided = [(name, value) for name, value in [("url", url), ("bibtex", bibtex), ("csv", csv_path)] if value]
    if dblp:
        provided.append(("dblp", ""))
    if len(provided) != 1:
        raise typer.BadParameter("Provide exactly one of --dblp, --url, --bibtex, or --csv.")
    return provided[0][0], provided[0][1] or ""




if __name__ == "__main__":
    app()
