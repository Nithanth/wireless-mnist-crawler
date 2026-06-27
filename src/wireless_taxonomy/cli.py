from __future__ import annotations

import csv as _csv
import inspect
import json
import sqlite3
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
        "Wireless paper classification + dataset extraction CLI.\n\n"
        "Commands:\n"
        "  classify          Classify papers as wireless (yes/maybe/no) for a venue/year.\n"
        "  eval              DB-free snapshot eval of classified CSV vs gold sheet.\n"
        "  fetch-coverage    Report OA full-text availability per venue/year.\n"
        "  extract-datasets  Full pipeline: classify → fetch PDF → extract datasets → CSV.\n"
        "  merge-results     Combine per-venue/year CSVs into master files.\n"
        "  cache             Inspect or clear the LLM/API cache.\n"
        "  llm-config        Show configured LLM providers and models."
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


def _parse_venue_years(entries: list[str]) -> list[tuple[str, str]]:
    """Parse ``VENUE:YEAR`` exclude entries into ``(venue, year)`` pairs."""
    parsed: list[tuple[str, str]] = []
    for raw in entries:
        venue, sep, year = raw.partition(":")
        if not sep or not venue.strip() or not year.strip():
            raise typer.BadParameter(f"--exclude must look like VENUE:YEAR (got {raw!r}).")
        parsed.append((venue.strip(), year.strip()))
    return parsed


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
    valid_sources = {"dblp", "bibtex", "csv", "url"}
    if source not in valid_sources:
        raise typer.BadParameter(f"--source must be one of {', '.join(sorted(valid_sources))} (got {source!r}).")
    if source != "dblp":
        if not source_value:
            raise typer.BadParameter("--source-value is required when --source is not 'dblp'.")
        if source in {"bibtex", "csv"} and not Path(source_value).exists():
            raise typer.BadParameter(f"--source-value file not found: {source_value}")
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
    exclude: List[str] = typer.Option(
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
    exclude_pairs = _parse_venue_years(list(exclude))
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
    year_list = _parse_years(years)

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
    pipeline = _pipeline(db)
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
            f"({_pct(agg_fetch, agg_total):.1f}%)"
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


@app.command("extract-datasets")
def extract_datasets(
    venue: str = typer.Option(..., "--venue", help="Conference venue, e.g. NSDI, SIGCOMM, IMC."),
    years: str = typer.Option(..., "--years", help="A year (2024) or inclusive range (2022:2025)."),
    out: str = typer.Option(".", "--out", help="Output directory for the 3 CSV sheets + raw JSON."),
    oa_json: Optional[str] = typer.Option(None, "--oa-json", help="Glob path to cov_*.json files from fetch-coverage to reuse known PDF URLs."),
    fresh: bool = typer.Option(False, "--fresh", help="Ignore LLM cache and re-extract all papers. PDF text cache in the DB is still reused."),
    wireless_only: bool = typer.Option(True, "--wireless-only/--all-papers", help="Only extract datasets from papers classified as wireless (yes+maybe for max recall). Use --all-papers to process every paper."),
    db: str = typer.Option("taxonomy.sqlite", "--db"),
) -> None:
    """Run the full dataset-extraction loop for a venue and year range.

    For each paper in the corpus:
      1. Fetch the PDF (or fall back to abstract) and send to the LLM natively.
      2. Extract datasets: name, relationship, modalities, OSI layers,
         availability (verified via live URL check), collection environment.
      3. Search Semantic Scholar + GitHub + web for other papers using each dataset.

    Writes three CSV sheets to --out matching the manual spreadsheet schema:
      <venue>_<years>_papers.csv   — Paper Title, Authors, Conference, Year, Datasets, BibTeX Key
      <venue>_<years>_bibtex.csv   — BibTeX Key, DOI Key, Full BibTeX
      <venue>_<years>_datasets.csv — Dataset Name, OSI Layers, Modalities, Availability, ...

    All LLM and search results are cached in .wt_cache.json for fast re-runs.
    """
    import glob as _glob

    year_list = _parse_years(years)
    year_tag = years.replace(":", "-")

    from wireless_taxonomy.analyze.cache import MetadataCache
    metadata_cache = MetadataCache(".wt_cache.json")

    oa_pdf_urls: dict[str, str] = {}
    if oa_json:
        oa_paths = sorted(_glob.glob(oa_json)) if "*" in oa_json else [oa_json]
        for p in oa_paths:
            try:
                data = json.loads(Path(p).read_text(encoding="utf-8"))
                for run in data.get("runs", []):
                    for paper in run.get("papers", []):
                        url = paper.get("pdf_url") or ""
                        if url and "dl.acm.org" not in url:
                            oa_pdf_urls[paper["title"]] = url
            except Exception as exc:
                typer.echo(f"Warning: could not load OA JSON {p}: {exc}", err=True)
    if oa_pdf_urls:
        typer.echo(f"Loaded {len(oa_pdf_urls)} PDF URLs from fetch-coverage output.")

    all_results: list[dict] = []
    pipeline = _pipeline(db)
    try:
        for year in year_list:
            typer.echo(f"\n[{venue} {year}] Extracting datasets...")
            result = pipeline.extract_datasets_conference(
                venue=venue,
                year=year,
                source_type="dblp",
                resolve_dois=True,
                oa_pdf_urls=oa_pdf_urls,
                cache=metadata_cache,
                fresh=fresh,
                wireless_only=wireless_only,
            )
            all_results.append(result)
            typer.echo(
                f"  {result['papers_with_datasets']}/{result['total_papers']} papers "
                f"with datasets — {result['total_dataset_records']} records"
            )
    finally:
        pipeline.close()
        metadata_cache.save()

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{venue.lower()}_{year_tag}"

    # Raw JSON for debugging / re-processing
    json_path = out_dir / f"{slug}_raw.json"
    json_path.write_text(
        json.dumps({"venue": venue, "years": year_list, "runs": all_results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    all_papers = [p for r in all_results for p in r["papers"]]

    # Sheet 1: Papers  — matches manual "List of Papers" sheet
    papers_path = out_dir / f"{slug}_papers.csv"
    with papers_path.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=[
            "Paper Title", "Authors", "Conference", "Year",
            "Datasets", "Bibtex Citation Key",
        ])
        writer.writeheader()
        for p in all_papers:
            writer.writerow({
                "Paper Title": p["title"],
                "Authors": p["authors"],
                "Conference": p["venue"],
                "Year": p["year"],
                "Datasets": "; ".join(d["name"] for d in p["datasets"]),
                "Bibtex Citation Key": p["bibtex_key"],
            })

    # Sheet 2: BibTeX — matches manual "BibTeX" sheet
    bibtex_path = out_dir / f"{slug}_bibtex.csv"
    with bibtex_path.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=[
            "Bibtex Citation Key", "DOI Version of Key", "Bibtex Citation",
        ])
        writer.writeheader()
        for p in all_papers:
            doi_key = f"doi:{p['doi']}" if p["doi"] else ""
            writer.writerow({
                "Bibtex Citation Key": p["bibtex_key"],
                "DOI Version of Key": doi_key,
                "Bibtex Citation": p["bibtex"],
            })

    # Sheet 3: Datasets — matches manual "List of Datasets" sheet
    seen_datasets: dict[str, dict] = {}
    for p in all_papers:
        for d in p["datasets"]:
            name = d["name"]
            if name not in seen_datasets:
                seen_datasets[name] = d.copy()
                seen_datasets[name]["_paper_count"] = 1
                seen_datasets[name]["_first_key"] = p["bibtex_key"]
                seen_datasets[name]["_introducing_key"] = p["bibtex_key"] if d.get("relationship_type") == "introduced" else ""
            else:
                seen_datasets[name]["_paper_count"] += 1
                if d.get("relationship_type") == "introduced" and not seen_datasets[name]["_introducing_key"]:
                    seen_datasets[name]["_introducing_key"] = p["bibtex_key"]

    datasets_path = out_dir / f"{slug}_datasets.csv"
    with datasets_path.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=[
            "Dataset Name", "Bibtex Citation Key",
            "OSI Layer (L1-L7)", "Modality(ies)",
            "Availability (Open? Y/N)", "Availability URL", "Annotations on Availability",
            "Collection Environment", "Number of Papers using Dataset",
        ])
        writer.writeheader()
        for name, d in sorted(seen_datasets.items()):
            avail = "Y" if d["availability"] else ("N" if d["availability"] is False else "")
            writer.writerow({
                "Dataset Name": name,
                "Bibtex Citation Key": d.get("_introducing_key") or d.get("_first_key", ""),
                "OSI Layer (L1-L7)": "; ".join(d["osi_layers"]),
                "Modality(ies)": "; ".join(d["modalities"]),
                "Availability (Open? Y/N)": avail,
                "Availability URL": d.get("availability_url", ""),
                "Annotations on Availability": d.get("availability_notes") or "",
                "Collection Environment": d.get("collection_environment") or "",
                "Number of Papers using Dataset": d.get("usage_count") or d["_paper_count"],
            })

    typer.echo(f"\nOutput written to {out_dir}/")
    typer.echo(f"  {papers_path.name}")
    typer.echo(f"  {bibtex_path.name}")
    typer.echo(f"  {datasets_path.name}")
    typer.echo(f"  {json_path.name} (raw)")


@app.command("merge-results")
def merge_results(
    results_dir: str = typer.Option("./src/results", "--dir", help="Directory containing per-venue/year CSV + JSON files."),
    out: str = typer.Option("./src/results", "--out", help="Output directory for merged master files."),
    min_corpus_reuse: int = typer.Option(
        2, "--min-corpus-reuse",
        help="Only include datasets mentioned in at least this many papers across the entire corpus. "
             "Set to 1 to keep all datasets.",
    ),
) -> None:
    """Merge all per-venue/year CSVs and JSONs into master files.

    Reads all *_papers.csv, *_bibtex.csv, *_datasets.csv, and *_raw.json files
    from --dir and produces:
      master_papers.csv, master_bibtex.csv, master_datasets.csv, master_raw.json

    With --min-corpus-reuse=2 (the default), only datasets referenced by at
    least 2 papers in the merged corpus survive into master_datasets.csv.
    """
    import glob as _glob

    src = Path(results_dir)
    dst = Path(out)
    dst.mkdir(parents=True, exist_ok=True)

    # --- Papers ---
    papers_files = sorted(_glob.glob(str(src / "*_papers.csv")))
    all_paper_rows: list[dict] = []
    paper_fields: list[str] = []
    for f in papers_files:
        with open(f, newline="", encoding="utf-8") as fh:
            reader = _csv.DictReader(fh)
            if not paper_fields and reader.fieldnames:
                paper_fields = list(reader.fieldnames)
            all_paper_rows.extend(reader)
    if all_paper_rows:
        p = dst / "master_papers.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=paper_fields)
            writer.writeheader()
            writer.writerows(all_paper_rows)
        typer.echo(f"  {p.name}: {len(all_paper_rows)} papers from {len(papers_files)} files")

    # --- BibTeX ---
    bibtex_files = sorted(_glob.glob(str(src / "*_bibtex.csv")))
    all_bib_rows: list[dict] = []
    seen_bib_keys: set[str] = set()
    bib_fields: list[str] = []
    for f in bibtex_files:
        with open(f, newline="", encoding="utf-8") as fh:
            reader = _csv.DictReader(fh)
            if not bib_fields and reader.fieldnames:
                bib_fields = list(reader.fieldnames)
            for row in reader:
                key = row.get("Bibtex Citation Key", "")
                if key not in seen_bib_keys:
                    seen_bib_keys.add(key)
                    all_bib_rows.append(row)
    if all_bib_rows:
        p = dst / "master_bibtex.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=bib_fields)
            writer.writeheader()
            writer.writerows(all_bib_rows)
        typer.echo(f"  {p.name}: {len(all_bib_rows)} entries from {len(bibtex_files)} files")

    # --- Datasets (deduplicated, cross-corpus paper counts, reuse filter) ---
    datasets_files = sorted(_glob.glob(str(src / "*_datasets.csv")))
    merged_ds: dict[str, dict] = {}
    ds_fields: list[str] = []
    for f in datasets_files:
        with open(f, newline="", encoding="utf-8") as fh:
            reader = _csv.DictReader(fh)
            if not ds_fields and reader.fieldnames:
                ds_fields = list(reader.fieldnames)
            for row in reader:
                name = row.get("Dataset Name", "")
                if name not in merged_ds:
                    merged_ds[name] = row
                else:
                    existing = merged_ds[name]
                    try:
                        old_count = int(existing.get("Number of Papers using Dataset") or 0)
                        new_count = int(row.get("Number of Papers using Dataset") or 0)
                        existing["Number of Papers using Dataset"] = str(old_count + new_count)
                    except ValueError:
                        pass
                    if not existing.get("Bibtex Citation Key") and row.get("Bibtex Citation Key"):
                        existing["Bibtex Citation Key"] = row["Bibtex Citation Key"]

    # Cross-corpus reuse: count distinct papers mentioning each dataset
    # across ALL raw JSON runs for an accurate corpus-wide count.
    corpus_paper_counts: dict[str, int] = {}
    for f in sorted(_glob.glob(str(src / "*_raw.json"))):
        try:
            run_data = json.loads(Path(f).read_text(encoding="utf-8"))
            for paper in run_data.get("papers") or []:
                seen_in_paper: set[str] = set()
                for ds in paper.get("datasets") or []:
                    ds_name = ds.get("name", "")
                    if ds_name and ds_name not in seen_in_paper:
                        seen_in_paper.add(ds_name)
                        corpus_paper_counts[ds_name] = corpus_paper_counts.get(ds_name, 0) + 1
        except Exception:
            pass

    # Update paper counts from corpus-wide scan and apply reuse filter
    total_before_filter = len(merged_ds)
    for name in list(merged_ds):
        corpus_count = corpus_paper_counts.get(name, 1)
        merged_ds[name]["Number of Papers using Dataset"] = str(corpus_count)
        if corpus_count < min_corpus_reuse:
            del merged_ds[name]

    if merged_ds:
        p = dst / "master_datasets.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=ds_fields)
            writer.writeheader()
            for name in sorted(merged_ds):
                writer.writerow(merged_ds[name])
        typer.echo(
            f"  {p.name}: {len(merged_ds)} datasets from {len(datasets_files)} files"
            f" (filtered from {total_before_filter} with --min-corpus-reuse={min_corpus_reuse})"
        )

    # --- Raw JSON ---
    json_files = sorted(_glob.glob(str(src / "*_raw.json")))
    all_runs: list[dict] = []
    for f in json_files:
        try:
            data = json.loads(Path(f).read_text(encoding="utf-8"))
            all_runs.append(data)
        except Exception:
            pass
    if all_runs:
        p = dst / "master_raw.json"
        p.write_text(json.dumps(all_runs, indent=2, ensure_ascii=False), encoding="utf-8")
        typer.echo(f"  {p.name}: {len(all_runs)} venue/year runs from {len(json_files)} files")

    typer.echo(f"\nMerged output in {dst}/")


@app.command("cache")
def cache_cmd(
    action: str = typer.Argument("status", help="Action: status | clear | clear-section"),
    section: Optional[str] = typer.Argument(None, help="Section name for clear-section (abstracts, dois, llm, oa, dataset_usage)"),
    cache_path: str = typer.Option(".wt_cache.json", "--cache-path"),
) -> None:
    """Inspect or manage the .wt_cache.json LLM/API response cache.

    \b
    Actions:
      status         Show entry counts per section and file size
      clear          Wipe the entire cache (prompts for confirmation)
      clear-section  Clear one section: abstracts | dois | llm | oa | dataset_usage
    """
    from wireless_taxonomy.analyze.cache import MetadataCache

    p = Path(cache_path)
    if not p.exists():
        typer.echo(f"Cache file not found: {p}")
        raise typer.Exit()

    c = MetadataCache(p)

    if action == "status":
        stats = c.stats()
        size_kb = p.stat().st_size / 1024
        typer.echo(f"Cache: {p}  ({size_kb:.1f} KB)")
        for section_name, count in stats.items():
            typer.echo(f"  {section_name:<20} {count} entries")

    elif action == "clear":
        typer.confirm(f"Wipe ALL entries in {p}?", abort=True)
        c.clear()
        c.save()
        typer.echo("Cache cleared.")

    elif action == "clear-section":
        if not section:
            typer.echo("Provide a section name: abstracts | dois | llm | oa | dataset_usage", err=True)
            raise typer.Exit(1)
        try:
            removed = c.clear_section(section)
            c.save()
            typer.echo(f"Cleared {removed} entries from '{section}'.")
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1)

    else:
        typer.echo(f"Unknown action '{action}'. Use: status | clear | clear-section", err=True)
        raise typer.Exit(1)


@app.command("llm-config")
def llm_config(db: str = typer.Option("taxonomy.sqlite", "--db")) -> None:
    settings = load_settings(db)
    typer.echo(f"Primary provider: {settings.llm.primary_provider}")
    fallbacks = ", ".join(settings.llm.fallback_providers) if settings.llm.fallback_providers else "(none)"
    typer.echo(f"Fallback providers: {fallbacks}")
    for provider in settings.llm.ordered_providers():
        key_status = "configured" if provider.api_key_configured else f"missing {provider.api_key_env}"
        typer.echo(f"- {provider.provider}: model={provider.model}, key={key_status}")


@app.command("corpus-status")
def corpus_status(db: str = typer.Option("taxonomy.sqlite", "--db")) -> None:
    """Show what's in the corpus: venues, years, paper counts, extraction status."""
    from wireless_taxonomy.db import connect

    conn = connect(db)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT v.name AS venue, ci.year, COUNT(p.id) AS papers,
               SUM(CASE WHEN wcp.label IS NOT NULL THEN 1 ELSE 0 END) AS classified,
               SUM(CASE WHEN be.id IS NOT NULL THEN 1 ELSE 0 END) AS extracted,
               SUM(CASE WHEN pta.content_text != '' THEN 1 ELSE 0 END) AS has_pdf
        FROM papers p
        JOIN conference_instances ci ON ci.id = p.conference_instance_id
        JOIN venues v ON v.id = ci.venue_id
        LEFT JOIN wireless_candidate_predictions wcp ON wcp.paper_id = p.id
        LEFT JOIN bibtex_entries be ON be.paper_id = p.id
        LEFT JOIN paper_text_artifacts pta ON pta.paper_id = p.id AND pta.fetch_status = 'success'
        GROUP BY v.name, ci.year
        ORDER BY v.name, ci.year
    """).fetchall()

    if not rows:
        typer.echo("Corpus is empty.")
        conn.close()
        raise typer.Exit()

    # Summary
    total_papers = sum(r["papers"] for r in rows)
    total_extracted = sum(r["extracted"] for r in rows)
    total_classified = sum(r["classified"] for r in rows)
    n_datasets = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
    n_claims = conn.execute("SELECT COUNT(*) FROM paper_analysis_dataset_claims").fetchone()[0]

    typer.echo(f"Corpus: {total_papers} papers across {len(rows)} venue-years")
    typer.echo(f"  Classified: {total_classified}  Extracted: {total_extracted}  Datasets: {n_datasets}  Claims: {n_claims}")
    typer.echo("")
    typer.echo(f"  {'Venue':<12} {'Year':<6} {'Papers':<8} {'Classified':<12} {'Extracted':<11} {'Has PDF'}")
    typer.echo(f"  {'─'*12} {'─'*6} {'─'*8} {'─'*12} {'─'*11} {'─'*8}")
    for r in rows:
        typer.echo(
            f"  {r['venue']:<12} {r['year']:<6} {r['papers']:<8} "
            f"{r['classified']:<12} {r['extracted']:<11} {r['has_pdf']}"
        )

    # Show pipeline runs
    typer.echo("")
    runs = conn.execute("""
        SELECT pr.id, v.name AS venue, ci.year, pr.stage, pr.status, pr.message,
               pr.started_at
        FROM pipeline_runs pr
        JOIN conference_instances ci ON ci.id = pr.conference_instance_id
        JOIN venues v ON v.id = ci.venue_id
        ORDER BY pr.id DESC LIMIT 20
    """).fetchall()
    if runs:
        typer.echo("  Recent runs (newest first):")
        for r in runs:
            status_icon = {"completed": "+", "running": "~", "failed": "x"}.get(r["status"], "?")
            msg = f" — {r['message'][:60]}" if r["message"] else ""
            typer.echo(f"    [{status_icon}] run {r['id']:>3}: {r['venue']} {r['year']} / {r['stage']} ({r['status']}){msg}")

    conn.close()


@app.command("prune")
def prune(
    venue: Optional[str] = typer.Option(None, "--venue", help="Venue to prune (e.g. SIGCOMM). Required unless --run-id is given."),
    year: Optional[int] = typer.Option(None, "--year", help="Year to prune. Required unless --run-id is given."),
    run_id: Optional[int] = typer.Option(None, "--run-id", help="Specific pipeline_run ID to prune."),
    stage: Optional[str] = typer.Option(None, "--stage", help="Only prune a specific stage (extract-datasets, classify-candidates, etc.)"),
    keep_pdfs: bool = typer.Option(True, "--keep-pdfs/--drop-pdfs", help="Keep cached PDF text artifacts (default: keep)."),
    db: str = typer.Option("taxonomy.sqlite", "--db"),
) -> None:
    """Prune extraction/classification results by venue/year or run_id.

    \b
    Examples:
      prune --venue SIGCOMM --year 2023               # all stages for SIGCOMM 2023
      prune --venue IMC --year 2024 --stage extract-datasets  # only extraction
      prune --run-id 42                               # a specific pipeline run
      prune --venue NSDI --year 2022 --drop-pdfs      # also clear cached PDFs

    Cached abstracts, DOIs, and OA lookups (.wt_cache.json) are NOT touched.
    Re-run extract-datasets to regenerate pruned data (LLM cache was already
    cleared so fresh extractions will happen).
    """
    from wireless_taxonomy.db import connect, transaction

    conn = connect(db)
    conn.row_factory = sqlite3.Row

    # Determine which conference_instance_id(s) and run(s) to target
    if run_id is not None:
        run_row = conn.execute("SELECT * FROM pipeline_runs WHERE id = ?", (run_id,)).fetchone()
        if not run_row:
            typer.echo(f"Run ID {run_id} not found.", err=True)
            conn.close()
            raise typer.Exit(1)
        target_runs = [run_row]
        ci_ids = {run_row["conference_instance_id"]}
    elif venue and year:
        ci_row = conn.execute("""
            SELECT ci.id FROM conference_instances ci
            JOIN venues v ON v.id = ci.venue_id
            WHERE LOWER(v.name) = LOWER(?) AND ci.year = ?
        """, (venue, year)).fetchone()
        if not ci_row:
            typer.echo(f"No data for {venue} {year} in the database.", err=True)
            conn.close()
            raise typer.Exit(1)
        ci_ids = {ci_row["id"]}
        query = "SELECT * FROM pipeline_runs WHERE conference_instance_id = ?"
        params: list = [ci_row["id"]]
        if stage:
            query += " AND stage = ?"
            params.append(stage)
        target_runs = conn.execute(query, params).fetchall()
    else:
        typer.echo("Provide --venue + --year or --run-id.", err=True)
        conn.close()
        raise typer.Exit(1)

    if not target_runs:
        typer.echo("No matching pipeline runs found.")
        conn.close()
        raise typer.Exit()

    # Show what will be pruned
    typer.echo(f"Will prune {len(target_runs)} pipeline run(s):")
    for r in target_runs:
        typer.echo(f"  run {r['id']}: {r['stage']} ({r['status']}) — {r['message'] or '(no message)'}")

    run_ids = [r["id"] for r in target_runs]
    stages = {r["stage"] for r in target_runs}

    typer.confirm("Proceed?", abort=True)

    with transaction(conn):
        deleted = {}

        # Prune classification results
        if "classify-candidates" in stages or stage is None:
            cur = conn.execute(
                f"DELETE FROM wireless_candidate_predictions WHERE run_id IN ({','.join('?' * len(run_ids))})",
                run_ids,
            )
            deleted["wireless_candidate_predictions"] = cur.rowcount

        # Prune extraction results
        if "extract-datasets" in stages or stage is None:
            cur = conn.execute(
                f"DELETE FROM paper_analysis_dataset_claims WHERE run_id IN ({','.join('?' * len(run_ids))})",
                run_ids,
            )
            deleted["paper_analysis_dataset_claims"] = cur.rowcount

            # Remove bibtex entries for papers in these conference instances
            for ci_id in ci_ids:
                cur = conn.execute("""
                    DELETE FROM bibtex_entries WHERE paper_id IN (
                        SELECT id FROM papers WHERE conference_instance_id = ?
                    )
                """, (ci_id,))
                deleted["bibtex_entries"] = deleted.get("bibtex_entries", 0) + cur.rowcount

        # Prune orphaned datasets (no remaining claims reference them)
        cur = conn.execute("""
            DELETE FROM datasets WHERE id NOT IN (
                SELECT DISTINCT dataset_id FROM paper_analysis_dataset_claims WHERE dataset_id IS NOT NULL
            )
        """)
        deleted["datasets (orphaned)"] = cur.rowcount

        # Optionally prune PDF text cache
        if not keep_pdfs:
            for ci_id in ci_ids:
                cur = conn.execute("""
                    DELETE FROM paper_text_artifacts WHERE paper_id IN (
                        SELECT id FROM papers WHERE conference_instance_id = ?
                    )
                """, (ci_id,))
                deleted["paper_text_artifacts"] = deleted.get("paper_text_artifacts", 0) + cur.rowcount

        # Mark runs as pruned
        conn.execute(
            f"DELETE FROM pipeline_runs WHERE id IN ({','.join('?' * len(run_ids))})",
            run_ids,
        )
        deleted["pipeline_runs"] = len(run_ids)

    typer.echo("\nPruned:")
    for table, count in sorted(deleted.items()):
        if count > 0:
            typer.echo(f"  {table}: {count} rows")
    typer.echo("\nDone. Re-run extract-datasets to regenerate.")
    conn.close()


if __name__ == "__main__":
    app()
