from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Optional

import click
import typer
from typer.core import TyperArgument, TyperOption

from wireless_taxonomy.config import load_settings
from wireless_taxonomy.pipeline import Pipeline
from wireless_taxonomy.review.interactive import review_summary

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
    db: str = typer.Option("taxonomy.sqlite", "--db"),
) -> None:
    source_type, source_value = _source(url, bibtex, csv_path)
    pipeline = _pipeline(db)
    try:
        run_id = pipeline.ingest(venue, year, source_type, source_value)
        typer.echo(f"Ingest completed. run_id={run_id}")
    finally:
        pipeline.close()


@app.command()
def run(
    venue: str = typer.Option(..., "--venue"),
    year: int = typer.Option(..., "--year"),
    url: Optional[str] = typer.Option(None, "--url"),
    bibtex: Optional[str] = typer.Option(None, "--bibtex"),
    csv_path: Optional[str] = typer.Option(None, "--csv"),
    out: Optional[str] = typer.Option(None, "--out"),
    fmt: str = typer.Option("csv", "--format"),
    yes: bool = typer.Option(False, "--yes", help="Proceed past scope warnings without prompting."),
    llm: bool = typer.Option(False, "--llm/--no-llm", help="Use configured LLM for paper analysis."),
    db: str = typer.Option("taxonomy.sqlite", "--db"),
) -> None:
    source_type, source_value = _source(url, bibtex, csv_path)
    pipeline = _pipeline(db)
    try:
        run_id = pipeline.ingest(venue, year, source_type, source_value)
        scope_run_id = pipeline.assess_scope(run_id)
        assessment = pipeline.latest_scope_assessment(scope_run_id)
        if assessment and not assessment["should_proceed"]:
            typer.echo(
                "Scope warning: "
                f"decision={assessment['decision']} "
                f"networking={assessment['networking_like_ratio']} "
                f"wireless={assessment['wireless_like_ratio']} "
                f"malformed={assessment['malformed_count']}"
            )
            if not yes and not typer.confirm("Proceed with verification and enrichment anyway?"):
                typer.echo(f"Stopped after scope assessment. root_run_id={run_id} scope_run_id={scope_run_id}")
                return
        pipeline.verify_paper_list(run_id, run_external=False, run_llm=False)
        pipeline.enrich_paper_text(run_id)
        pipeline.discover_full_text(run_id)
        pipeline.assess_paper_inputs(run_id)
        analysis_run_id = pipeline.agentic_paper_analysis(run_id, use_llm=llm)
        pipeline.reflect_paper_analysis(run_id, analysis_run_id=analysis_run_id)
        pipeline.classify_wireless(run_id)
        pipeline.extract_datasets(run_id)
        pipeline.check_availability(run_id)
        pipeline.resolve_reuse(run_id)
        if out:
            pipeline.export(run_id, out, fmt)
        typer.echo(f"Run completed. root_run_id={run_id}")
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


@app.command("verify-paper-list")
def verify_paper_list(
    run_id: int = typer.Option(..., "--run-id"),
    external: bool = typer.Option(False, "--external/--no-external", help="Cross-check DOI/title metadata with external services."),
    llm: bool = typer.Option(False, "--llm/--no-llm", help="Ask the configured LLM to verify extraction against the source page."),
    db: str = typer.Option("taxonomy.sqlite", "--db"),
) -> None:
    pipeline = _pipeline(db)
    try:
        stage_run = pipeline.verify_paper_list(run_id, run_external=external, run_llm=llm)
        report = pipeline.latest_paper_list_report(stage_run)
        if report:
            issues = _verification_issue_count(report)
            typer.echo(
                "Paper-list verification completed. "
                f"run_id={stage_run} papers={report['paper_count']} "
                f"issues={issues} "
                f"confidence={report['final_confidence']}"
            )
        else:
            typer.echo(f"Paper-list verification completed. run_id={stage_run}")
    finally:
        pipeline.close()


@app.command("assess-scope")
def assess_scope(run_id: int = typer.Option(..., "--run-id"), db: str = typer.Option("taxonomy.sqlite", "--db")) -> None:
    pipeline = _pipeline(db)
    try:
        stage_run = pipeline.assess_scope(run_id)
        assessment = pipeline.latest_scope_assessment(stage_run)
        if assessment:
            typer.echo(
                "Scope assessment completed. "
                f"run_id={stage_run} decision={assessment['decision']} "
                f"papers={assessment['paper_count']} "
                f"networking={assessment['networking_like_ratio']} "
                f"wireless={assessment['wireless_like_ratio']} "
                f"malformed={assessment['malformed_count']} "
                f"should_proceed={bool(assessment['should_proceed'])}"
            )
        else:
            typer.echo(f"Scope assessment completed. run_id={stage_run}")
    finally:
        pipeline.close()


@app.command("enrich-paper-text")
def enrich_paper_text(run_id: int = typer.Option(..., "--run-id"), db: str = typer.Option("taxonomy.sqlite", "--db")) -> None:
    pipeline = _pipeline(db)
    try:
        stage_run = pipeline.enrich_paper_text(run_id)
        typer.echo(f"Paper text enrichment completed. run_id={stage_run}")
    finally:
        pipeline.close()


@app.command("assess-paper-inputs")
def assess_paper_inputs(run_id: int = typer.Option(..., "--run-id"), db: str = typer.Option("taxonomy.sqlite", "--db")) -> None:
    pipeline = _pipeline(db)
    try:
        stage_run = pipeline.assess_paper_inputs(run_id)
        typer.echo(f"Paper input readiness assessment completed. run_id={stage_run}")
    finally:
        pipeline.close()


@app.command("discover-full-text")
def discover_full_text(
    run_id: int = typer.Option(..., "--run-id"),
    paper_id: Optional[int] = typer.Option(None, "--paper-id", help="Discover full text for one paper from the run instead of all papers."),
    db: str = typer.Option("taxonomy.sqlite", "--db"),
) -> None:
    pipeline = _pipeline(db)
    try:
        stage_run = pipeline.discover_full_text(run_id, paper_id=paper_id)
        typer.echo(f"Full-text discovery completed. run_id={stage_run}")
    finally:
        pipeline.close()


@app.command("add-pdfs")
def add_pdfs(
    run_id: int = typer.Option(..., "--run-id"),
    directory: str = typer.Option(..., "--dir", help="Directory containing downloaded PDFs to match to papers in the run."),
    db: str = typer.Option("taxonomy.sqlite", "--db"),
) -> None:
    pipeline = _pipeline(db)
    try:
        stage_run = pipeline.add_pdfs(run_id, directory)
        typer.echo(f"Local PDF import completed. run_id={stage_run}")
    finally:
        pipeline.close()


@app.command("fetch-acm-browser")
def fetch_acm_browser(
    run_id: int = typer.Option(..., "--run-id"),
    profile_dir: str = typer.Option(".browser/acm", "--profile-dir", help="Persistent browser profile directory for ACM/institutional login cookies."),
    paper_id: Optional[int] = typer.Option(None, "--paper-id", help="Fetch one paper from the run instead of all ACM DOI papers."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Maximum number of ACM DOI papers to try."),
    login: bool = typer.Option(False, "--login", help="Open ACM in a persistent browser so you can log in through your institution."),
    headless: bool = typer.Option(False, "--headless/--no-headless", help="Run the browser without a visible window after login."),
    browser_channel: Optional[str] = typer.Option(None, "--browser-channel", help="Playwright browser channel, e.g. chrome, msedge, or chrome-beta."),
    cdp_url: Optional[str] = typer.Option(None, "--cdp-url", help="Connect to a manually launched Chrome remote debugging URL, e.g. http://127.0.0.1:9222."),
    delay_seconds: Optional[float] = typer.Option(None, "--delay-seconds", help="Delay between ACM PDF requests. Defaults to env or 8 seconds."),
    db: str = typer.Option("taxonomy.sqlite", "--db"),
) -> None:
    pipeline = _pipeline(db)
    try:
        stage_run = pipeline.fetch_acm_browser(
            run_id,
            profile_dir=profile_dir,
            paper_id=paper_id,
            limit=limit,
            headless=headless,
            browser_channel=browser_channel,
            cdp_url=cdp_url,
            delay_seconds=delay_seconds,
            login_only=login,
        )
        if login:
            typer.echo(f"ACM browser login session saved in: {profile_dir}")
        else:
            typer.echo(f"ACM browser fetch completed. run_id={stage_run}")
    finally:
        pipeline.close()


@app.command("extract-datasets")
def extract_datasets(run_id: int = typer.Option(..., "--run-id"), db: str = typer.Option("taxonomy.sqlite", "--db")) -> None:
    pipeline = _pipeline(db)
    try:
        stage_run = pipeline.extract_datasets(run_id)
        typer.echo(f"Dataset extraction completed. run_id={stage_run}")
    finally:
        pipeline.close()


@app.command("agentic-paper-analysis")
def agentic_paper_analysis(
    run_id: int = typer.Option(..., "--run-id"),
    paper_id: Optional[int] = typer.Option(None, "--paper-id", help="Analyze one paper from the run instead of all papers."),
    llm: bool = typer.Option(False, "--llm/--no-llm", help="Use configured LLM instead of deterministic local analyzer."),
    db: str = typer.Option("taxonomy.sqlite", "--db"),
) -> None:
    pipeline = _pipeline(db)
    try:
        stage_run = pipeline.agentic_paper_analysis(run_id, paper_id=paper_id, use_llm=llm)
        typer.echo(f"Agentic paper analysis completed. run_id={stage_run}")
    finally:
        pipeline.close()


@app.command("reflect-paper-analysis")
def reflect_paper_analysis(
    run_id: int = typer.Option(..., "--run-id"),
    analysis_run_id: Optional[int] = typer.Option(None, "--analysis-run-id", help="Specific agentic-paper-analysis run to reflect."),
    paper_id: Optional[int] = typer.Option(None, "--paper-id", help="Reflect one paper analysis instead of all papers."),
    db: str = typer.Option("taxonomy.sqlite", "--db"),
) -> None:
    pipeline = _pipeline(db)
    try:
        stage_run = pipeline.reflect_paper_analysis(run_id, analysis_run_id=analysis_run_id, paper_id=paper_id)
        typer.echo(f"Paper analysis reflection completed. run_id={stage_run}")
    finally:
        pipeline.close()


@app.command("check-availability")
def check_availability(run_id: int = typer.Option(..., "--run-id"), db: str = typer.Option("taxonomy.sqlite", "--db")) -> None:
    pipeline = _pipeline(db)
    try:
        stage_run = pipeline.check_availability(run_id)
        typer.echo(f"Availability check completed. run_id={stage_run}")
    finally:
        pipeline.close()


@app.command("resolve-reuse")
def resolve_reuse(run_id: int = typer.Option(..., "--run-id"), db: str = typer.Option("taxonomy.sqlite", "--db")) -> None:
    pipeline = _pipeline(db)
    try:
        stage_run = pipeline.resolve_reuse(run_id)
        typer.echo(f"Reuse resolution completed. run_id={stage_run}")
    finally:
        pipeline.close()


@app.command()
def review(run_id: Optional[int] = typer.Option(None, "--run-id"), db: str = typer.Option("taxonomy.sqlite", "--db")) -> None:
    pipeline = _pipeline(db)
    try:
        typer.echo(review_summary(pipeline.conn, run_id))
    finally:
        pipeline.close()


@app.command()
def export(
    run_id: Optional[int] = typer.Option(None, "--run-id"),
    fmt: str = typer.Option("xlsx", "--format"),
    out: str = typer.Option(..., "--out"),
    scope: str = typer.Option("related", "--scope", help="For JSON exports: related conference runs or exact run only."),
    db: str = typer.Option("taxonomy.sqlite", "--db"),
) -> None:
    pipeline = _pipeline(db)
    try:
        path = pipeline.export(run_id, out, fmt, scope)
        typer.echo(f"Exported {fmt}: {path}")
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


@app.command()
def jaccard(
    run_id: int = typer.Option(..., "--run-id"),
    manual: str = typer.Option(..., "--manual", help="CSV of the manually curated paper list."),
    title_col: Optional[str] = typer.Option(
        None, "--title-col", help="Column in the manual CSV holding paper titles. Auto-detected when omitted."
    ),
    conference_col: Optional[str] = typer.Option(
        None, "--conference-col", help="Manual CSV conference/venue column. Auto-detected when omitted."
    ),
    year_col: Optional[str] = typer.Option(
        None, "--year-col", help="Manual CSV year column. Auto-detected when omitted."
    ),
    wireless_only: bool = typer.Option(
        True,
        "--wireless-only/--all-papers",
        help="Compare the pipeline's wireless-classified papers (default) vs the full ingested list.",
    ),
    wireless_source: str = typer.Option(
        "classify", "--wireless-source", help="Wireless decision source: classify (keyword) or agentic (analysis)."
    ),
    conference_filter: bool = typer.Option(
        True,
        "--conference-filter/--no-conference-filter",
        help="Filter the manual CSV to the run's conference+year when those columns exist.",
    ),
    out: Optional[str] = typer.Option(None, "--out", help="Write the full diff report JSON to this path."),
    db: str = typer.Option("taxonomy.sqlite", "--db"),
) -> None:
    """Jaccard (IoU) of the pipeline's papers vs a manually curated list, by normalized title."""
    pipeline = _pipeline(db)
    try:
        report = pipeline.jaccard(
            run_id,
            manual,
            title_col=title_col,
            conference_col=conference_col,
            year_col=year_col,
            wireless_only=wireless_only,
            wireless_source=wireless_source,
            conference_filter=conference_filter,
            out=out,
        )
        typer.echo(
            "Paper-list coverage (Jaccard/IoU). "
            f"venue={report.venue} year={report.year} "
            f"wireless_only={report.wireless_only} conference_filtered={report.conference_filtered} "
            f"index={report.jaccard_index:.4f} "
            f"intersection={report.intersection_count} union={report.union_count} "
            f"automated={report.automated_count} manual={report.manual_count} "
            f"missed_by_cli={len(report.missed_by_cli)} extra_from_cli={len(report.extra_from_cli)} "
            f"title_column={report.title_column!r}"
        )
        if out:
            typer.echo(f"Wrote diff report: {out}")
    finally:
        pipeline.close()


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


def _source(url: str | None, bibtex: str | None, csv_path: str | None) -> tuple[str, str]:
    provided = [(name, value) for name, value in [("url", url), ("bibtex", bibtex), ("csv", csv_path)] if value]
    if len(provided) != 1:
        raise typer.BadParameter("Provide exactly one of --url, --bibtex, or --csv.")
    return provided[0][0], provided[0][1] or ""


def _verification_issue_count(report) -> int:
    try:
        payload = json.loads(report["report_json"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return (
            report["missing_authors_count"]
            + report["missing_abstract_count"]
            + report["missing_doi_count"]
            + report["duplicate_title_count"]
            + report["low_confidence_count"]
            + report["external_mismatch_count"]
            + report["llm_correction_count"]
        )
    issues = payload.get("issues")
    return len(issues) if isinstance(issues, list) else 0


if __name__ == "__main__":
    app()
