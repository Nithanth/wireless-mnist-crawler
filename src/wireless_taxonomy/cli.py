from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from wireless_taxonomy.config import load_settings
from wireless_taxonomy.pipeline import Pipeline
from wireless_taxonomy.review.interactive import review_summary

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
    db: str = typer.Option("taxonomy.sqlite", "--db"),
) -> None:
    source_type, source_value = _source(url, bibtex, csv_path)
    pipeline = _pipeline(db)
    try:
        run_id = pipeline.run(venue, year, source_type, source_value, out)
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
            typer.echo(
                "Paper-list verification completed. "
                f"run_id={stage_run} papers={report['paper_count']} "
                f"issues={report['missing_authors_count'] + report['missing_abstract_count'] + report['missing_doi_count'] + report['duplicate_title_count'] + report['low_confidence_count'] + report['external_mismatch_count'] + report['llm_correction_count']} "
                f"confidence={report['final_confidence']}"
            )
        else:
            typer.echo(f"Paper-list verification completed. run_id={stage_run}")
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
    db: str = typer.Option("taxonomy.sqlite", "--db"),
) -> None:
    pipeline = _pipeline(db)
    try:
        path = pipeline.export(run_id, out, fmt)
        typer.echo(f"Exported {fmt}: {path}")
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
    typer.echo(f"Embedding provider: {settings.llm.embedding_provider}")
    typer.echo(f"Embedding model: {settings.llm.embedding_model}")
    for provider in settings.llm.ordered_providers():
        key_status = "configured" if provider.api_key_configured else f"missing {provider.api_key_env}"
        typer.echo(f"- {provider.provider}: model={provider.model}, key={key_status}")


def _source(url: str | None, bibtex: str | None, csv_path: str | None) -> tuple[str, str]:
    provided = [(name, value) for name, value in [("url", url), ("bibtex", bibtex), ("csv", csv_path)] if value]
    if len(provided) != 1:
        raise typer.BadParameter("Provide exactly one of --url, --bibtex, or --csv.")
    return provided[0][0], provided[0][1] or ""


if __name__ == "__main__":
    app()
