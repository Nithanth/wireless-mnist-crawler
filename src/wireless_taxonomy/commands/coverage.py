
import json
from pathlib import Path
from typing import Optional

import typer

from wireless_taxonomy.commands._shared import make_pipeline, parse_years, pct


def _echo_coverage(result: dict) -> None:
    total = result["total_papers"]
    fetchable = result["fetchable"]
    typer.echo(
        f"{result['venue']} {result['year']} — {fetchable}/{total} legally fetchable "
        f"({result['fetchable_pct']:.1f}%)"
    )
    by_status = result.get("by_oa_status") or {}
    by_source = result.get("by_source") or {}
    if by_status:
        typer.echo("  by OA status: " + ", ".join(f"{k} {v}" for k, v in by_status.items()))
    if by_source:
        typer.echo("  by source: " + ", ".join(f"{k} {v}" for k, v in by_source.items()))


def register(app: typer.Typer) -> None:
    @app.command("fetch-coverage")
    def fetch_coverage(
        venue: str = typer.Option(..., "--venue", help="Conference venue, e.g. NSDI, SIGCOMM, GLOBECOM."),
        years: str = typer.Option(..., "--years", help="A year (2024) or inclusive range (2023:2025)."),
        source: str = typer.Option("dblp", "--source", help="Paper-list source: dblp (default), bibtex, csv, or url."),
        source_value: Optional[str] = typer.Option(None, "--source-value", help="Path or URL when --source is not dblp."),
        resolve_dois: bool = typer.Option(
            True, "--resolve-dois/--no-resolve-dois", help="Backfill missing DOIs from title before the OA lookup."
        ),
        json_out: Optional[str] = typer.Option(None, "--json", help="Write the full per-paper availability set (all years) as JSON here."),
        cache: bool = typer.Option(
            True, "--cache/--no-cache", help="Read/write OA lookups to a disk index for fast, deterministic re-runs."
        ),
        cache_path: str = typer.Option(
            ".wt_cache.json", "--cache-path", help="Where the abstract/DOI/OA cache lives (used unless --no-cache)."
        ),
        db: str = typer.Option("taxonomy.sqlite", "--db", help="SQLite work DB (created/reused)."),
    ) -> None:
        """Report the %/list of papers we can LEGALLY fetch full text for.

        For each venue-year, ingests the paper list, backfills DOIs, then asks the
        open metadata APIs (Unpaywall → OpenAlex → Semantic Scholar → arXiv) whether
        a legally hosted open-access copy exists. It reads OA *status* only and never
        downloads or scrapes paywalled full text. Prints the fetchable percentage
        (with a gold/green/etc. breakdown) per year; ``--json`` writes the full
        per-paper set (DOI, OA status, license, PDF URL, source) for every year.

        Unpaywall is the canonical legal-OA source but needs a contact email — set
        ``WIRELESS_TAXONOMY_CONTACT_EMAIL`` to enable it (the other sources still run
        without it).
        """
        valid_sources = {"dblp", "bibtex", "csv", "url"}
        if source not in valid_sources:
            raise typer.BadParameter(f"--source must be one of {', '.join(sorted(valid_sources))} (got {source!r}).")
        if source != "dblp":
            if not source_value:
                raise typer.BadParameter("--source-value is required when --source is not 'dblp'.")
            if source in {"bibtex", "csv"} and not Path(source_value).exists():
                raise typer.BadParameter(f"--source-value file not found: {source_value}")
        year_list = parse_years(years)

        metadata_cache = None
        if cache:
            from wireless_taxonomy.analyze.cache import MetadataCache

            metadata_cache = MetadataCache(cache_path)

        import os as _os

        from wireless_taxonomy.config import load_dotenv

        load_dotenv()  # so a .env-provided contact email enables Unpaywall (and the note below is accurate)
        if not (_os.getenv("WIRELESS_TAXONOMY_CONTACT_EMAIL") or "").strip():
            typer.echo(
                "note: WIRELESS_TAXONOMY_CONTACT_EMAIL not set — skipping Unpaywall "
                "(the canonical OA source); using OpenAlex/Semantic Scholar/arXiv only.",
                err=True,
            )

        results: list[dict] = []
        pipeline = make_pipeline(db)
        try:
            for year in year_list:
                results.append(
                    pipeline.text_availability_conference(
                        venue,
                        year,
                        source_type=source,
                        source_value=source_value,
                        resolve_dois=resolve_dois,
                        cache=metadata_cache,
                    )
                )
        finally:
            pipeline.close()
            if metadata_cache is not None:
                metadata_cache.save()

        for result in results:
            _echo_coverage(result)
            typer.echo("")

        if len(results) > 1:
            agg_total = sum(r["total_papers"] for r in results)
            agg_fetch = sum(r["fetchable"] for r in results)
            span = f"{year_list[0]}–{year_list[-1]}"
            typer.echo(
                f"{venue} {span} (aggregate) — {agg_fetch}/{agg_total} legally fetchable "
                f"({pct(agg_fetch, agg_total):.1f}%)"
            )

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
