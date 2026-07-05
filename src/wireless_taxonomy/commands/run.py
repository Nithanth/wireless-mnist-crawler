"""Unified `run` command: orchestrates the full pipeline in the correct order.

Replaces run_loop.sh with a proper CLI command that enforces stage ordering,
supports per-stage model selection, and provides cost visibility.

Pipeline order per venue/year:
  1. fetch-coverage   — resolve open-access PDF URLs
  2. extract-datasets — classify + extract datasets from PDFs/abstracts
Then once at the end:
  3. merge-results    — combine per-venue CSVs into master files
  4. report           — generate corpus summary
"""

from typing import Optional

import typer

from wireless_taxonomy.commands._shared import parse_years


def register(app: typer.Typer) -> None:
    @app.command("run")
    def run(
        venues: str = typer.Option(
            ..., "--venues",
            help="Comma-separated venue names (e.g. SIGCOMM,IMC,NSDI).",
        ),
        years: str = typer.Option(
            ..., "--years",
            help="A year (2024), comma list (2023,2024), or inclusive range (2022:2025).",
        ),
        workers: int = typer.Option(
            6, "--workers", min=1, max=16,
            help="Thread parallelism for PDF fetch + LLM stages.",
        ),
        web_search: bool = typer.Option(
            False, "--web-search",
            help="Enable Brave/Google-CSE PDF discovery fallback.",
        ),
        classify_model: Optional[str] = typer.Option(
            None, "--classify-model",
            help="Override LLM for classification (provider/model, e.g. google/gemini-2.0-flash).",
        ),
        extract_model: Optional[str] = typer.Option(
            None, "--extract-model",
            help="Override LLM for extraction (provider/model, e.g. google/gemini-2.5-flash).",
        ),
        wireless_only: bool = typer.Option(
            True, "--wireless-only/--all-papers",
            help="Only extract from wireless-classified papers.",
        ),
        verbose: bool = typer.Option(False, "--verbose", help="Per-paper classification output."),
        fresh: bool = typer.Option(False, "--fresh", help="Clear LLM cache and re-extract everything."),
        retry_failed: bool = typer.Option(False, "--retry-failed", help="Retry previously failed PDF downloads."),
        out: str = typer.Option("./src/results", "--out", help="Output directory for CSV sheets."),
        db: str = typer.Option("taxonomy.sqlite", "--db"),
        corpus: str = typer.Option(
            None, "--corpus",
            help="Corpus name (corpora/<name>/). Overrides --db and --out.",
        ),
        estimate: bool = typer.Option(
            False, "--estimate",
            help="Dry-run: show estimated paper counts and LLM calls per stage without running anything.",
        ),
        skip_coverage: bool = typer.Option(False, "--skip-coverage", help="Skip fetch-coverage (reuse existing OA data)."),
        skip_merge: bool = typer.Option(False, "--skip-merge", help="Skip merge-results at the end."),
        skip_report: bool = typer.Option(False, "--skip-report", help="Skip report generation at the end."),
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive confirmations."),
    ) -> None:
        """Run the full pipeline: fetch-coverage, extract-datasets, merge, report.

        Orchestrates all stages in the correct dependency order for one or more
        venue/year combinations.  Equivalent to run_loop.sh but with proper
        prerequisite enforcement, per-stage model selection, and cost visibility.

        \b
        Examples:
          wt run --venues SIGCOMM,IMC --years 2022:2025 --workers 6
          wt run --venues ICC --years 2023 --classify-model google/gemini-2.0-flash
          wt run --venues NSDI --years 2025 --skip-coverage  # reuse existing OA data
        """
        import csv as _csv
        import glob as _glob
        import json
        import os
        import time
        from pathlib import Path

        from wireless_taxonomy.analyze.cache import MetadataCache
        from wireless_taxonomy.commands._shared import make_pipeline, parse_model_override
        from wireless_taxonomy.config import load_dotenv

        load_dotenv()

        venue_list = [v.strip() for v in venues.split(",") if v.strip()]
        # Parse years: support comma lists and ranges
        year_list: list[int] = []
        for part in years.split(","):
            part = part.strip()
            if not part:
                continue
            year_list.extend(parse_years(part))
        year_list = sorted(set(year_list))

        if not venue_list or not year_list:
            typer.echo("No venues or years specified.", err=True)
            raise typer.Exit(1)

        # ── Corpus resolution ─────────────────────────────────────────────
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
                    raise typer.Exit(1)

        # ── Per-stage model overrides ─────────────────────────────────────
        cls_settings = parse_model_override(classify_model) if classify_model else None
        ext_settings = parse_model_override(extract_model) if extract_model else None

        if classify_model or extract_model:
            typer.echo("Model configuration:")
            if classify_model:
                typer.echo(f"  Classification: {classify_model}")
            else:
                typer.echo("  Classification: (default)")
            if extract_model:
                typer.echo(f"  Extraction:     {extract_model}")
            else:
                typer.echo("  Extraction:     (default)")
            typer.echo("")

        # ── Plan summary ──────────────────────────────────────────────────
        pairs = [(v, y) for v in venue_list for y in year_list]
        stages = []
        if not skip_coverage:
            stages.append("fetch-coverage")
        stages.append("extract-datasets")
        if not skip_merge:
            stages.append("merge-results")
        if not skip_report:
            stages.append("report")

        typer.echo(f"Pipeline: {' -> '.join(stages)}")
        typer.echo(f"Venues:   {', '.join(venue_list)}")
        typer.echo(f"Years:    {', '.join(str(y) for y in year_list)}")
        typer.echo(f"Total:    {len(pairs)} venue-year combinations")
        typer.echo(f"Workers:  {workers}")
        typer.echo(f"Output:   {out}")
        typer.echo("")

        # ── Estimate mode ─────────────────────────────────────────────
        if estimate:
            pipeline = make_pipeline(db)
            typer.echo("Estimate (dry-run):\n")
            typer.echo(f"  {'Venue':<12} {'Year':<6} {'Papers':<8} {'With PDF':<10} {'OA Done':<9} {'Est. LLM calls'}")
            typer.echo(f"  {'─'*12} {'─'*6} {'─'*8} {'─'*10} {'─'*9} {'─'*15}")
            total_papers = 0
            total_classify = 0
            total_extract_est = 0
            for venue, year in pairs:
                cov = pipeline.check_coverage_ready(venue, year)
                n = cov["total_papers"]
                n_pdf = cov["papers_with_pdf"]
                oa_done = "yes" if cov["ready"] else "no"
                # Classify: 1 LLM call per paper; Extract: ~10-20% wireless
                est_classify = n
                est_extract = max(1, int(n * 0.15)) if n > 0 else 0
                total_papers += n
                total_classify += est_classify
                total_extract_est += est_extract
                typer.echo(
                    f"  {venue:<12} {year:<6} {n:<8} {n_pdf:<10} {oa_done:<9} "
                    f"~{est_classify} classify + ~{est_extract} extract"
                )
            pipeline.close()
            typer.echo(f"\n  Total: ~{total_papers} papers")
            typer.echo(f"  Est. LLM calls: ~{total_classify} classify + ~{total_extract_est} extract")
            typer.echo(f"  (Classification calls are cached; only uncached papers incur API cost)")
            if not skip_coverage:
                typer.echo(f"  OA resolution: ~{total_papers} Unpaywall/OpenAlex lookups (free)")
            typer.echo("\nRe-run without --estimate to execute.")
            raise typer.Exit()

        metadata_cache = MetadataCache(".wt_cache.json")
        pipeline = make_pipeline(db)
        t0 = time.monotonic()
        all_extract_results: list[dict] = []

        try:
            for vi, venue in enumerate(venue_list, 1):
                for yi, year in enumerate(year_list, 1):
                    label = f"[{venue} {year}] ({vi * len(year_list) + yi - len(year_list)}/{len(pairs)})"

                    # ── Stage 1: fetch-coverage ───────────────────────
                    if not skip_coverage:
                        typer.echo(f"\n{label} Resolving open-access PDF URLs...")
                        pipeline.text_availability_conference(
                            venue, year,
                            source_type="dblp",
                            resolve_dois=True,
                            cache=metadata_cache,
                            workers=workers,
                            web_search=web_search,
                        )
                        metadata_cache.save()
                        cov = pipeline.check_coverage_ready(venue, year)
                        typer.echo(
                            f"  Coverage: {cov['papers_with_pdf']}/{cov['total_papers']} "
                            f"papers have PDF URLs"
                        )

                    # ── Stage 2: extract-datasets ─────────────────────
                    typer.echo(f"\n{label} Extracting datasets...")
                    result = pipeline.extract_datasets_conference(
                        venue=venue,
                        year=year,
                        source_type="dblp",
                        resolve_dois=True,
                        cache=metadata_cache,
                        fresh=fresh,
                        wireless_only=wireless_only,
                        workers=workers,
                        verbose=verbose,
                        retry_failed=retry_failed,
                        classify_settings=cls_settings,
                        extract_settings=ext_settings,
                    )
                    all_extract_results.append(result)
                    typer.echo(
                        f"  {result['papers_with_datasets']}/{result['total_papers']} papers "
                        f"with datasets - {result['total_dataset_records']} records"
                    )
                    metadata_cache.save()

                    # Record in corpus metadata
                    if corpus_obj is not None:
                        model_label = extract_model or classify_model or ""
                        if not model_label:
                            from wireless_taxonomy.config import load_settings as _ls
                            from wireless_taxonomy.llm import LlmRouter as _R
                            try:
                                _p = _R(_ls(db).llm).select_provider()
                                model_label = f"{_p.provider}/{_p.model}"
                            except Exception:
                                pass
                        if model_label:
                            corpus_obj.record_run(
                                model_label, f"{venue} {year}",
                                classify_model=classify_model,
                                extract_model=extract_model,
                            )

        finally:
            pipeline.close()
            metadata_cache.save()

        elapsed = time.monotonic() - t0

        # ── Stage 3: merge-results ────────────────────────────────────
        out_dir = Path(out)
        if not skip_merge:
            typer.echo("\nMerging results into master CSVs...")
            # Write per-venue/year CSVs first
            for result in all_extract_results:
                v, y = result["venue"], result["year"]
                _write_venue_year_csvs(out_dir, v, y, result)

            # Now merge
            from wireless_taxonomy.commands.merge import register as _unused  # noqa: F841 — just need the module
            import wireless_taxonomy.commands.merge as _merge_mod
            # Call merge logic directly via the CLI
            _do_merge(out_dir)

        # ── Stage 4: report ───────────────────────────────────────────
        if not skip_report:
            typer.echo("\nGenerating corpus report...")
            _do_report(out_dir)

        # ── Summary ──────────────────────────────────────────────────
        total_papers = sum(r["total_papers"] for r in all_extract_results)
        total_datasets = sum(r["total_dataset_records"] for r in all_extract_results)
        typer.echo(f"\nDone in {elapsed:.0f}s.")
        typer.echo(f"  {len(pairs)} venue-years processed")
        typer.echo(f"  {total_papers} wireless papers")
        typer.echo(f"  {total_datasets} dataset records extracted")
        typer.echo(f"  Output: {out_dir}/")


def _write_venue_year_csvs(out_dir, venue, year, result):
    """Write the 3 per-venue/year CSV sheets from extraction results."""
    import csv as _csv
    import json
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{venue.lower()}_{year}"
    papers = result.get("papers", [])

    # Raw JSON
    json_path = out_dir / f"{slug}_raw.json"
    json_path.write_text(
        json.dumps({"venue": venue, "years": [year], "runs": [result]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Papers CSV
    papers_path = out_dir / f"{slug}_papers.csv"
    with papers_path.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=[
            "Paper Title", "Authors", "Conference", "Year",
            "Datasets", "Bibtex Citation Key",
        ])
        writer.writeheader()
        for p in papers:
            writer.writerow({
                "Paper Title": p["title"],
                "Authors": p["authors"],
                "Conference": p["venue"],
                "Year": p["year"],
                "Datasets": "; ".join(d["name"] for d in p["datasets"]),
                "Bibtex Citation Key": p["bibtex_key"],
            })

    # BibTeX CSV
    bibtex_path = out_dir / f"{slug}_bibtex.csv"
    with bibtex_path.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=[
            "Bibtex Citation Key", "DOI Version of Key", "Bibtex Citation",
        ])
        writer.writeheader()
        for p in papers:
            doi_key = f"doi:{p['doi']}" if p["doi"] else ""
            writer.writerow({
                "Bibtex Citation Key": p["bibtex_key"],
                "DOI Version of Key": doi_key,
                "Bibtex Citation": p["bibtex"],
            })

    # Datasets CSV
    seen_datasets: dict[str, dict] = {}
    for p in papers:
        for d in p["datasets"]:
            name = d["name"]
            if name not in seen_datasets:
                seen_datasets[name] = d.copy()
                seen_datasets[name]["_paper_count"] = 1
                seen_datasets[name]["_first_key"] = p["bibtex_key"]
                seen_datasets[name]["_introducing_key"] = (
                    p["bibtex_key"] if d.get("relationship_type") == "introduced" else ""
                )
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


def _do_merge(results_dir):
    """Run merge-results logic programmatically."""
    import csv as _csv
    import glob as _glob
    import json
    import os
    import re as _re
    from pathlib import Path

    import typer

    src = results_dir
    dst = results_dir

    # Papers
    papers_files = sorted(
        p for p in _glob.glob(str(src / "*_papers.csv"))
        if not os.path.basename(p).startswith(("master_", "consolidated_"))
    )
    all_paper_rows: list[dict] = []
    paper_fields: set[str] = set()
    for f in papers_files:
        with open(f, newline="", encoding="utf-8") as fh:
            reader = _csv.DictReader(fh)
            if reader.fieldnames:
                paper_fields.update(reader.fieldnames)
            all_paper_rows.extend(reader)
    if all_paper_rows:
        p = dst / "master_papers.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=sorted(paper_fields))
            writer.writeheader()
            writer.writerows(all_paper_rows)
        typer.echo(f"  {p.name}: {len(all_paper_rows)} papers from {len(papers_files)} files")

    # BibTeX
    bibtex_files = sorted(
        p for p in _glob.glob(str(src / "*_bibtex.csv"))
        if not os.path.basename(p).startswith(("master_", "consolidated_"))
    )
    all_bib_rows: list[dict] = []
    seen_bib_keys: set[str] = set()
    bib_fields: set[str] = set()
    for f in bibtex_files:
        with open(f, newline="", encoding="utf-8") as fh:
            reader = _csv.DictReader(fh)
            if reader.fieldnames:
                bib_fields.update(reader.fieldnames)
            for row in reader:
                key = row.get("Bibtex Citation Key", "")
                if key not in seen_bib_keys:
                    seen_bib_keys.add(key)
                    all_bib_rows.append(row)
    if all_bib_rows:
        p = dst / "master_bibtex.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=sorted(bib_fields))
            writer.writeheader()
            writer.writerows(all_bib_rows)
        typer.echo(f"  {p.name}: {len(all_bib_rows)} entries from {len(bibtex_files)} files")

    # Datasets
    datasets_files = sorted(
        p for p in _glob.glob(str(src / "*_datasets.csv"))
        if not os.path.basename(p).startswith(("master_", "consolidated_"))
    )
    merged_ds: dict[str, dict] = {}
    ds_fields: set[str] = set()
    for f in datasets_files:
        with open(f, newline="", encoding="utf-8") as fh:
            reader = _csv.DictReader(fh)
            if reader.fieldnames:
                ds_fields.update(reader.fieldnames)
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
    if merged_ds:
        p = dst / "master_datasets.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=sorted(ds_fields))
            writer.writeheader()
            for name in sorted(merged_ds):
                writer.writerow(merged_ds[name])
        typer.echo(f"  {p.name}: {len(merged_ds)} datasets from {len(datasets_files)} files")


def _do_report(results_dir):
    """Run report generation programmatically."""
    from pathlib import Path

    import typer

    try:
        from wireless_taxonomy.commands.report import (
            _load_coverage,
            _load_raw_runs,
            _venue_year_stats,
        )
    except ImportError:
        typer.echo("  (report module not available, skipping)", err=True)
        return

    results = Path(results_dir)
    entries = _load_raw_runs(results)
    if not entries:
        typer.echo("  No raw JSON results found for report.", err=True)
        return

    coverage = _load_coverage([results])
    stats = _venue_year_stats(entries, coverage)

    total_papers = sum(s["wireless_papers"] for s in stats)
    total_datasets = sum(s["datasets"] for s in stats)
    typer.echo(f"  Report: {len(stats)} venue-years, {total_papers} papers, {total_datasets} datasets")
