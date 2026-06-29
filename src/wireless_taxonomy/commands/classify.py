
import csv as _csv
import json
from pathlib import Path
from typing import Optional

import typer

from wireless_taxonomy.commands._shared import CSV_FIELDS, echo_breakdown, make_pipeline, parse_years, pct


def register(app: typer.Typer) -> None:
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

        results: list[dict] = []
        pipeline = make_pipeline(db)
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
            echo_breakdown(result)
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
                f"(abstracts: {agg_abs}/{agg_total}, {pct(agg_abs, agg_total):.0f}%)"
            )
            for label in ("yes", "maybe", "no"):
                n = agg_counts.get(label, 0)
                typer.echo(f"  {label:<5} {n:>4}  ({pct(n, agg_total):>5.1f}%)")

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
                writer = _csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(all_papers)
            typer.echo(f"Wrote CSV: {csv_path}")
