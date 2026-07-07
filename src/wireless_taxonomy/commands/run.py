"""The `add` command: fetch + extract venues/years into the active corpus.

Pipeline order per venue/year:
  1. fetch-coverage   — resolve open-access PDF URLs
  2. extract-datasets — classify + extract datasets from PDFs/abstracts
Then once at the end:
  3. merge-results    — combine per-venue CSVs into master files
"""

from typing import Optional

import typer

from wireless_taxonomy.commands._shared import parse_years


def register(app: typer.Typer) -> None:
    @app.command("add")
    def add(
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
        estimate: bool = typer.Option(
            False, "--estimate",
            help="Dry-run: show estimated paper counts and LLM calls per stage without running anything.",
        ),
        skip_coverage: bool = typer.Option(False, "--skip-coverage", help="Skip fetch-coverage (reuse existing OA data)."),
        classify_no_pdf: bool = typer.Option(
            False, "--classify-no-pdf/--classify-with-pdf",
            help="Classify using keyword snippets from PDF text instead of full PDF (56x cheaper).",
        ),
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive confirmations."),
    ) -> None:
        """Add venues/years to the corpus: fetch PDFs, classify, extract datasets.

        Runs the full extraction pipeline for each venue/year pair, then merges
        results into master CSVs. The corpus is incremental — re-running with the
        same venues/years is cheap (cached papers skip instantly).

        \b
        Examples:
          wt add --venues SIGCOMM,IMC --years 2022:2025 --workers 6
          wt add --venues ICC --years 2023 --classify-model google/gemini-2.0-flash
          wt add --venues NSDI --years 2025 --skip-coverage --estimate
        """
        import time
        from pathlib import Path

        from wireless_taxonomy.analyze.cache import MetadataCache
        from wireless_taxonomy.commands._shared import make_pipeline, parse_model_override
        from wireless_taxonomy.config import load_dotenv
        from wireless_taxonomy.corpus import (
            active_corpus,
            check_model_compatibility,
            resolve_corpus,
        )

        load_dotenv()

        venue_list = [v.strip() for v in venues.split(",") if v.strip()]
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

        # ── Corpus resolution (mandatory — auto-creates if needed) ────
        from wireless_taxonomy.corpus import next_auto_name

        corpus_obj = active_corpus()
        if corpus_obj is None:
            auto_name = next_auto_name()
            corpus_obj = resolve_corpus(auto_name, create=True)
            typer.echo(f"Created corpus: {corpus_obj.name}")

        db = str(corpus_obj.db_path)
        out = str(corpus_obj.results_dir)

        try:
            from wireless_taxonomy.config import load_settings as _load_settings
            from wireless_taxonomy.llm import LlmRouter as _Router

            _provider = _Router(_load_settings(db).llm).select_provider()
            current_model = f"{_provider.provider}/{_provider.model}"
        except RuntimeError as exc:
            typer.echo(f"ERROR: {exc}", err=True)
            raise typer.Exit(1)
        except Exception:
            current_model = ""

        if not _provider.api_key_configured:
            typer.echo(
                f"ERROR: No API key for {_provider.provider}. "
                f"Set {_provider.api_key_env} in .env",
                err=True,
            )
            raise typer.Exit(1)

        warning = check_model_compatibility(corpus_obj, current_model)
        if warning:
            typer.echo(f"\nWARNING: {warning}\n", err=True)
            if not yes and not typer.confirm("Continue with mixed-model corpus?", default=False):
                raise typer.Exit(1)

        # Snapshot before mutating — always reversible
        snap = corpus_obj.snapshot()
        if snap:
            typer.echo(f"Snapshot: {snap.stem} (rollback with: wt rollback {snap.stem})")

        # ── Per-stage model overrides ─────────────────────────────────────
        cls_settings = parse_model_override(classify_model) if classify_model else None
        ext_settings = parse_model_override(extract_model) if extract_model else None

        if classify_model or extract_model:
            typer.echo("Models:")
            if classify_model:
                typer.echo(f"  classify: {classify_model}")
            if extract_model:
                typer.echo(f"  extract:  {extract_model}")
            typer.echo("")

        # ── Plan summary ──────────────────────────────────────────────────
        pairs = [(v, y) for v in venue_list for y in year_list]
        stages = ["fetch-coverage"] if not skip_coverage else []
        stages.append("extract-datasets")
        stages.append("merge-results")

        typer.echo(f"Corpus:   {corpus_obj.name}")
        typer.echo(f"Pipeline: {' -> '.join(stages)}")
        typer.echo(f"Venues:   {', '.join(venue_list)}")
        typer.echo(f"Years:    {', '.join(str(y) for y in year_list)}")
        typer.echo(f"Total:    {len(pairs)} venue-year combinations")
        typer.echo(f"Workers:  {workers}")
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

        # ── Logging setup ──────────────────────────────────────────────
        import datetime as _dt
        import sys

        log_dir = corpus_obj.dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        _log_fh = log_file.open("w", encoding="utf-8")

        def _echo(msg: str = "") -> None:
            """Print to both console and log file."""
            typer.echo(msg)
            _log_fh.write(msg + "\n")
            _log_fh.flush()

        def _ts() -> str:
            return _dt.datetime.now().strftime("%H:%M:%S")

        def _fmt_dur(secs: float) -> str:
            s = int(secs)
            h, s = divmod(s, 3600)
            m, s = divmod(s, 60)
            if h > 0:
                return f"{h}h {m}m {s}s"
            if m > 0:
                return f"{m}m {s}s"
            return f"{s}s"

        # ── Start ─────────────────────────────────────────────────────
        _echo("")
        _echo("╔══════════════════════════════════════════════════════╗")
        _echo(f"║  wt add: {len(venue_list)} venues x {len(year_list)} years = {len(pairs)} jobs")
        _echo(f"║  Venues: {', '.join(venue_list)}")
        _echo(f"║  Years:  {', '.join(str(y) for y in year_list)}")
        _echo(f"║  Corpus: {corpus_obj.name}")
        _echo(f"║  Log:    {log_file}")
        _echo("╚══════════════════════════════════════════════════════╝")
        _echo("")

        metadata_cache = MetadataCache(".wt_cache.json")
        pipeline = make_pipeline(db)
        t0 = time.monotonic()
        all_extract_results: list[dict] = []
        completed = 0
        failed: list[str] = []

        try:
            for vi, venue in enumerate(venue_list, 1):
                for yi, year in enumerate(year_list, 1):
                    idx = (vi - 1) * len(year_list) + yi
                    _echo("")
                    _echo(f"┌─ [{idx}/{len(pairs)}] {venue} {year}")
                    _echo(f"│  {_ts()} Starting...")

                    try:
                        # ── Stage 1: fetch-coverage ───────────────────────
                        if not skip_coverage:
                            _echo(f"│  {_ts()} fetch-coverage...")
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
                            pct = f"({100*cov['papers_with_pdf']//max(cov['total_papers'],1)}%)" if cov['total_papers'] else ""
                            _echo(
                                f"│  {_ts()} Coverage: {cov['papers_with_pdf']}/{cov['total_papers']} PDFs {pct}"
                            )

                        # ── Stage 2: extract-datasets ─────────────────────
                        _echo(f"│  {_ts()} extract-datasets...")
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
                            classify_no_pdf=classify_no_pdf,
                        )
                        all_extract_results.append(result)
                        _echo(
                            f"│  {_ts()} Done: {result['papers_with_datasets']}/{result['total_papers']} papers "
                            f"-> {result['total_dataset_records']} datasets"
                        )
                        metadata_cache.save()

                        # Record in corpus metadata
                        model_label = extract_model or classify_model or current_model
                        if model_label:
                            corpus_obj.record_run(
                                model_label, f"{venue} {year}",
                                classify_model=classify_model,
                                extract_model=extract_model,
                            )
                        completed += 1

                    except Exception as exc:
                        _echo(f"│  {_ts()} FAILED: {exc}")
                        failed.append(f"{venue} {year}")
                        continue

                    # ETA
                    elapsed_so_far = time.monotonic() - t0
                    remaining = len(pairs) - idx
                    if idx > 0 and remaining > 0:
                        avg = elapsed_so_far / idx
                        eta = _fmt_dur(avg * remaining)
                        _echo(f"│  ETA: ~{eta} remaining ({remaining} jobs left)")
                    _echo(f"└─ [{idx}/{len(pairs)}] {venue} {year} complete ({_fmt_dur(elapsed_so_far)} elapsed)")

        finally:
            pipeline.close()
            metadata_cache.save()

        elapsed = time.monotonic() - t0

        # ── Stage 3: merge-results ────────────────────────────────────
        out_dir = Path(out)
        _echo(f"\n{_ts()} Merging results...")
        for result in all_extract_results:
            v, y = result["venue"], result["year"]
            _write_venue_year_csvs(out_dir, v, y, result)
        _do_merge(out_dir)

        # ── Summary ──────────────────────────────────────────────────
        total_papers = sum(r["total_papers"] for r in all_extract_results)
        total_datasets = sum(r["total_dataset_records"] for r in all_extract_results)
        _echo("")
        _echo("╔══════════════════════════════════════════════════════╗")
        _echo(f"║  COMPLETE: {_fmt_dur(elapsed)}")
        _echo(f"║  {completed}/{len(pairs)} venue-years processed" + (f" ({len(failed)} failed)" if failed else ""))
        _echo(f"║  {total_papers} wireless papers")
        _echo(f"║  {total_datasets} dataset records extracted")
        _echo(f"║  Output: {out_dir}/")
        _echo(f"║  Log:    {log_file}")
        _echo("╚══════════════════════════════════════════════════════╝")
        if failed:
            _echo(f"\nFailed: {', '.join(failed)}")
        _echo(f"\nNext: wt export    (reconcile + final output)")
        _log_fh.close()


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
                seen_datasets[name]["_bibtex_keys"] = [p["bibtex_key"]]
                seen_datasets[name]["_introducing_key"] = (
                    p["bibtex_key"] if d.get("relationship_type") == "introduced" else ""
                )
            else:
                seen_datasets[name]["_paper_count"] += 1
                seen_datasets[name]["_bibtex_keys"].append(p["bibtex_key"])
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
            # Deduplicate and sort bibtex keys
            unique_keys = sorted(set(d["_bibtex_keys"]))
            writer.writerow({
                "Dataset Name": name,
                "Bibtex Citation Key": "; ".join(unique_keys),
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



