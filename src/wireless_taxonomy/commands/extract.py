
import csv as _csv
import glob as _glob
import json
from pathlib import Path
from typing import Optional

import typer

from wireless_taxonomy.commands._shared import make_pipeline, parse_years


def register(app: typer.Typer) -> None:
    @app.command("extract-datasets")
    def extract_datasets(
        venue: str = typer.Option(..., "--venue", help="Conference venue, e.g. NSDI, SIGCOMM, IMC."),
        years: str = typer.Option(..., "--years", help="A year (2024) or inclusive range (2022:2025)."),
        out: str = typer.Option(".", "--out", help="Output directory for the 3 CSV sheets + raw JSON."),
        oa_json: Optional[str] = typer.Option(None, "--oa-json", help="Glob path to cov_*.json files from fetch-coverage to reuse known PDF URLs."),
        fresh: bool = typer.Option(False, "--fresh", help="Ignore LLM cache and re-extract all papers. PDF text cache in the DB is still reused."),
        retry_failed: bool = typer.Option(False, "--retry-failed", help="Retry PDF downloads that previously failed (e.g. after fixing a download bug). Only affects papers with a recorded failure; all other papers are unaffected."),
        refresh_paper: list[str] = typer.Option(
            [], "--refresh-paper",
            help="Force re-classify + re-extract just this paper (exact title, case/punctuation-insensitive); repeatable. Cache is refreshed for these papers only.",
        ),
        wireless_only: bool = typer.Option(True, "--wireless-only/--all-papers", help="Only extract datasets from papers classified as wireless (yes+maybe for max recall). Use --all-papers to process every paper."),
        workers: int = typer.Option(1, "--workers", min=1, max=16, help="Thread parallelism for PDF fetch + LLM stages. Results are identical to a sequential run; 4-8 is a good default for paid API tiers."),
        verbose: bool = typer.Option(False, "--verbose", help="Print one line per paper during classification (default: rolling count only)."),
        classify_model: Optional[str] = typer.Option(
            None, "--classify-model",
            help="Override the LLM model for the classification stage only (e.g. google/gemini-2.0-flash). "
                 "Format: provider/model. Cheaper models work well here.",
        ),
        extract_model: Optional[str] = typer.Option(
            None, "--extract-model",
            help="Override the LLM model for the dataset extraction stage only (e.g. google/gemini-2.5-flash). "
                 "Format: provider/model. More capable models produce better results here.",
        ),
        db: str = typer.Option("taxonomy.sqlite", "--db"),
        corpus: str = typer.Option(
            None, "--corpus",
            help="Corpus name (corpora/<name>/). Overrides --db and --out with the "
                 "corpus's own DB and results dir, checks model compatibility, and "
                 "records the run in corpus metadata.",
        ),
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive confirmations (e.g. model-change warning)."),
    ) -> None:
        """Run the full dataset-extraction loop for a venue and year range.

        For each paper in the corpus:
          1. Fetch the PDF (or fall back to abstract) and send to the LLM natively.
          2. Extract datasets: name, relationship, modalities, OSI layers,
             availability (verified via live URL check), collection environment.

        Writes three CSV sheets to --out matching the manual spreadsheet schema:
          <venue>_<years>_papers.csv   — Paper Title, Authors, Conference, Year, Datasets, BibTeX Key
          <venue>_<years>_bibtex.csv   — BibTeX Key, DOI Key, Full BibTeX
          <venue>_<years>_datasets.csv — Dataset Name, OSI Layers, Modalities, Availability, ...

        All LLM and search results are cached in .wt_cache.json for fast re-runs.
        """
        year_list = parse_years(years)
        year_tag = years.replace(":", "-")

        # ── Corpus resolution + model-change guard ───────────────────────────
        corpus_obj = None
        if corpus:
            from wireless_taxonomy.config import load_settings as _load_settings
            from wireless_taxonomy.corpus import check_model_compatibility, resolve_corpus
            from wireless_taxonomy.llm import LlmRouter as _Router

            try:
                corpus_obj = resolve_corpus(corpus, create=True)
            except ValueError as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(1)
            db = str(corpus_obj.db_path)
            out = str(corpus_obj.results_dir)

            try:
                _provider = _Router(_load_settings(db).llm).select_provider()
                current_model = f"{_provider.provider}/{_provider.model}"
            except Exception:
                current_model = ""
            warning = check_model_compatibility(corpus_obj, current_model)
            if warning:
                typer.echo(f"\nWARNING: {warning}\n", err=True)
                if not yes and not typer.confirm("Continue with mixed-model corpus?", default=False):
                    typer.echo("Aborted. Create a new corpus with: corpus new")
                    raise typer.Exit(1)
            if corpus_obj is not None and current_model:
                corpus_obj.record_run(current_model, f"{venue} {years}")

        # ── Per-stage model overrides ─────────────────────────────────────
        from wireless_taxonomy.commands._shared import parse_model_override
        classify_settings = parse_model_override(classify_model) if classify_model else None
        extract_settings = parse_model_override(extract_model) if extract_model else None

        from wireless_taxonomy.analyze.cache import MetadataCache
        metadata_cache = MetadataCache(".wt_cache.json")

        from wireless_taxonomy.analyze.dataset_extractor import _is_acm_blocked

        oa_pdf_urls: dict[str, str] = {}
        if oa_json:
            oa_paths = sorted(_glob.glob(oa_json)) if "*" in oa_json else [oa_json]
            for p in oa_paths:
                try:
                    data = json.loads(Path(p).read_text(encoding="utf-8"))
                    for run in data.get("runs", []):
                        for paper in run.get("papers", []):
                            url = paper.get("pdf_url") or ""
                            if url and not _is_acm_blocked(url):
                                oa_pdf_urls[paper["title"]] = url
                except Exception as exc:
                    typer.echo(f"Warning: could not load OA JSON {p}: {exc}", err=True)
        if oa_pdf_urls:
            typer.echo(f"Loaded {len(oa_pdf_urls)} PDF URLs from fetch-coverage output.")

        all_results: list[dict] = []
        pipeline = make_pipeline(db)

        # ── Prerequisite check: warn if fetch-coverage hasn't been run ────
        for yr in year_list:
            cov = pipeline.check_coverage_ready(venue, yr)
            if cov["total_papers"] > 0 and not cov["ready"]:
                typer.echo(
                    f"\nWARNING: fetch-coverage has not been run for {venue} {yr}.\n"
                    f"  {cov['total_papers']} papers ingested but 0 have OA/PDF resolution.\n"
                    f"  Extraction will fall back to abstract-only (poor quality).\n"
                    f"  Run first:  wt fetch-coverage --venue {venue} --years {yr}\n",
                    err=True,
                )
                if not yes and not typer.confirm("Continue without PDF coverage?", default=False):
                    pipeline.close()
                    raise typer.Exit(1)

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
                    workers=workers,
                    refresh_titles=set(refresh_paper) if refresh_paper else None,
                    verbose=verbose,
                    retry_failed=retry_failed,
                    classify_settings=classify_settings,
                    extract_settings=extract_settings,
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

        # Audit log: dropped items + ungrounded evidence
        all_dropped = []
        all_ungrounded = []
        for p in all_papers:
            for d in (p.get("dropped") or []):
                all_dropped.append({"paper": p["title"], **d})
            for d in p["datasets"]:
                if d.get("grounded") is False:
                    all_ungrounded.append({"paper": p["title"], "dataset": d["name"], "evidence_text": d["evidence_text"]})
        if all_dropped or all_ungrounded:
            audit_path = out_dir / f"{slug}_audit.jsonl"
            with audit_path.open("w", encoding="utf-8") as fh:
                for item in all_dropped:
                    fh.write(json.dumps({"type": "dropped", **item}, ensure_ascii=False) + "\n")
                for item in all_ungrounded:
                    fh.write(json.dumps({"type": "ungrounded_evidence", **item}, ensure_ascii=False) + "\n")
            typer.echo(f"  Audit log: {audit_path.name} ({len(all_dropped)} dropped, {len(all_ungrounded)} ungrounded)")

        # Sheet 1: Papers — matches manual "List of Papers" sheet
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
