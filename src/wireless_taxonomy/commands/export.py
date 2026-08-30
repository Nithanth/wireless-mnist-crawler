"""The `export` command: reconcile + fill-availability + report → final output.

Runs the post-extraction pipeline on the active corpus to produce clean,
deduplicated output CSVs ready for use in papers/analysis.
"""

from typing import Optional

import typer


def register(app: typer.Typer) -> None:
    @app.command("export")
    def export(
        llm_confirm: bool = typer.Option(
            True, "--llm-confirm/--no-llm-confirm",
            help="Use LLM to confirm/reject reconciliation candidates.",
        ),
        fill: bool = typer.Option(
            True, "--fill/--no-fill",
            help="Run fill-availability pass for unknown datasets.",
        ),
        report: bool = typer.Option(
            True, "--report/--no-report",
            help="Generate the corpus summary report.",
        ),
        name_threshold: float = typer.Option(
            0.75, "--name-threshold", help="Reconciliation: minimum name similarity ratio.",
        ),
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive confirmations."),
    ) -> None:
        """Produce final output: reconcile duplicates, fill availability, generate report.

        Operates on the active corpus. Run after `wt add` has extracted all your
        venues/years. Produces:
          - consolidated_datasets.csv  (deduplicated, with reuse counts)
          - review_candidates.csv      (ambiguous pairs for manual review)
          - master_report.md           (corpus summary)

        \b
        Examples:
          wt export                        # full pipeline
          wt export --no-fill --no-report  # just reconcile
          wt export --no-llm-confirm       # skip LLM, similarity-only
        """
        import json
        from pathlib import Path

        from wireless_taxonomy.config import load_dotenv
        from wireless_taxonomy.corpus import active_corpus

        load_dotenv()

        corpus_obj = active_corpus()
        if corpus_obj is None:
            typer.echo(
                "No active corpus. Run `wt init` or `wt add` first.",
                err=True,
            )
            raise typer.Exit(1)

        db = str(corpus_obj.db_path)
        results_dir = corpus_obj.results_dir

        typer.echo(f"Corpus: {corpus_obj.name}")
        typer.echo(f"Output: {results_dir}/")
        typer.echo("")

        # ── Step 1: Reconcile datasets ────────────────────────────────────
        typer.echo("Reconciling datasets...")
        from wireless_taxonomy.postprocess.entity_resolution import (
            DatasetRecord,
            consolidate,
            reconcile,
        )

        # Load records from raw JSONs (richer than CSV for reconciliation)
        import glob as _glob

        records: list[DatasetRecord] = []
        for f in sorted(_glob.glob(str(results_dir / "*_raw.json"))):
            name = Path(f).name
            if name.startswith(("master_", "consolidated_")):
                continue
            try:
                data = json.loads(Path(f).read_text(encoding="utf-8"))
                for run in (data.get("runs") or ([data] if "papers" in data else [])):
                    for paper in run.get("papers") or []:
                        key = paper.get("bibtex_key", "")
                        for ds in paper.get("datasets") or []:
                            ds_name = ds.get("name", "").strip()
                            if not ds_name:
                                continue
                            records.append(DatasetRecord(
                                name=ds_name,
                                bibtex_keys=[key] if key else [],
                                modalities="; ".join(ds.get("modalities", [])),
                                osi_layers="; ".join(ds.get("osi_layers", [])),
                                environment=ds.get("collection_environment", ""),
                                availability_url=ds.get("availability_url", ""),
                                availability_notes=ds.get("availability_notes", ""),
                            ))
            except Exception:
                continue

        if not records:
            typer.echo("  No dataset records found. Run `wt add` first.", err=True)
            raise typer.Exit(1)

        typer.echo(f"  {len(records)} dataset records loaded")

        # Lower thresholds when LLM is confirming (it handles precision)
        effective_name = 0.60 if llm_confirm else name_threshold
        effective_combined = 0.55 if llm_confirm else 0.70

        matches = reconcile(
            records,
            url_dedup=True,
            similarity=True,
            llm_confirm=llm_confirm,
            similarity_name_threshold=effective_name,
            similarity_combined_threshold=effective_combined,
            llm_workers=4,
        )

        # Write consolidated CSV
        canonical = consolidate(records, matches)
        consolidated_path = results_dir / "consolidated_datasets.csv"
        _write_consolidated_csv(canonical, consolidated_path)
        reused = [d for d in canonical if d.reuse_count > 1]
        typer.echo(
            f"  {len(canonical)} unique datasets ({len(reused)} reused across papers)"
        )
        typer.echo(f"  Wrote: {consolidated_path.name}")

        # Write review CSV for ambiguous pairs
        from wireless_taxonomy.postprocess.entity_resolution import CanonicalDataset

        review_candidates = [m for m in matches if m.method in ("llm_unsure", "llm_skipped", "similarity")]
        if review_candidates:
            import csv as _csv

            review_path = results_dir / "review_candidates.csv"
            with review_path.open("w", newline="", encoding="utf-8") as fh:
                writer = _csv.writer(fh)
                writer.writerow([
                    "decision (merge/keep)", "confidence", "method", "reason",
                    "name_a", "keys_a", "name_b", "keys_b",
                ])
                for m in sorted(review_candidates, key=lambda m: -m.confidence):
                    writer.writerow([
                        "", f"{m.confidence:.2f}", m.method, m.reason,
                        m.a.name, ", ".join(m.a.bibtex_keys),
                        m.b.name, ", ".join(m.b.bibtex_keys),
                    ])
            typer.echo(f"  {len(review_candidates)} pairs need review: {review_path.name}")

        url_merged = sum(1 for m in matches if m.method == "url_dedup")
        llm_merged = sum(1 for m in matches if m.method == "llm_confirmed")
        typer.echo(f"  Auto-merged: {url_merged} (URL) + {llm_merged} (LLM confirmed)")
        typer.echo("")

        # ── Step 2: Fill availability ─────────────────────────────────────
        if fill:
            typer.echo("Filling unknown dataset availability...")
            from wireless_taxonomy.analyze.cache import MetadataCache
            from wireless_taxonomy.analyze.dataset_extractor import _check_url_live
            from wireless_taxonomy.db import connect
            from wireless_taxonomy.llm import LlmRequest, LlmRouter

            from wireless_taxonomy.config import load_settings
            settings = load_settings(db)
            conn = connect(db)
            import sqlite3
            conn.row_factory = sqlite3.Row
            cache = MetadataCache(".wt_cache.json")
            router = LlmRouter(settings.llm)

            rows = conn.execute("""
                SELECT DISTINCT
                    d.id AS dataset_id, d.canonical_name,
                    p.id AS paper_id, p.title, p.abstract,
                    pta.content_text AS pdf_text
                FROM datasets d
                JOIN paper_analysis_dataset_claims c ON c.dataset_id = d.id
                JOIN papers p ON p.id = c.paper_id
                LEFT JOIN paper_text_artifacts pta
                       ON pta.paper_id = p.id AND pta.fetch_status = 'ok'
                WHERE d.availability_status = 'unknown'
                ORDER BY d.canonical_name
            """).fetchall()

            if rows:
                filled = 0
                for row in rows:
                    text = row["pdf_text"] or row["abstract"] or ""
                    if not text:
                        continue
                    prompt = (
                        f"Dataset: {row['canonical_name']}\n"
                        f"Paper: {row['title']}\n\n"
                        f"Text:\n{text[:8000]}\n\n"
                        "Is this dataset publicly/openly available? "
                        "Answer JSON: {\"available\": true/false/null, \"url\": \"...\" or null}"
                    )
                    cache_key = f"fill_avail:{row['dataset_id']}:{row['paper_id']}"
                    cached = cache.llm.get(cache_key)
                    if cached is not None:
                        result_json = cached
                    else:
                        try:
                            resp = router.complete(LlmRequest(task="fill_availability", prompt=prompt))
                            result_json = resp.parsed or {}
                            cache.llm[cache_key] = result_json
                            cache.dirty = True
                        except Exception:
                            continue

                    avail = result_json.get("available")
                    url = result_json.get("url") or ""
                    if avail is True and url and _check_url_live(url):
                        status = "open"
                    elif avail is False:
                        status = "closed"
                    else:
                        continue
                    conn.execute(
                        "UPDATE datasets SET availability_status = ?, availability_url = COALESCE(?, availability_url) WHERE id = ?",
                        (status, url or None, row["dataset_id"]),
                    )
                    filled += 1

                conn.commit()
                cache.save()
                typer.echo(f"  Filled {filled}/{len(rows)} datasets")
            else:
                typer.echo("  No unknown-availability datasets to fill")
            conn.close()
            typer.echo("")

        # ── Step 3: Report ────────────────────────────────────────────────
        if report:
            typer.echo("Generating report...")
            try:
                from wireless_taxonomy.commands.report import (
                    _load_coverage,
                    _load_raw_runs,
                    _venue_year_stats,
                )

                entries = _load_raw_runs(results_dir)
                if entries:
                    coverage = _load_coverage([results_dir])
                    stats = _venue_year_stats(entries, coverage)
                    total_papers = sum(s["wireless_papers"] for s in stats)
                    total_datasets = sum(s["datasets"] for s in stats)
                    typer.echo(f"  {len(stats)} venue-years, {total_papers} papers, {total_datasets} datasets")
                else:
                    typer.echo("  No raw JSON results found for report.")
            except Exception as exc:
                typer.echo(f"  Report generation failed: {exc}", err=True)

        # ── Step 4: Consolidated papers & bibtex CSVs ─────────────────────
        typer.echo("Writing consolidated papers & bibtex...")
        _write_consolidated_papers_bibtex(corpus_obj, results_dir)
        typer.echo(f"  Wrote: consolidated_papers.csv, consolidated_bibtex.csv")

        # ── Step 5: PDF-only variant ─────────────────────────────────────
        typer.echo("Writing PDF-only datasets variant...")
        _write_pdf_only_datasets(results_dir)
        typer.echo(f"  Wrote: consolidated_datasets_pdf_only.csv")

        typer.echo("\nExport complete.")
        typer.echo(f"  {consolidated_path}")
        if review_candidates:
            typer.echo(f"  {results_dir / 'review_candidates.csv'}")


def _write_consolidated_csv(canonical, path):
    """Write the consolidated dataset list to CSV."""
    import csv as _csv
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.writer(fh)
        writer.writerow([
            "Canonical Name", "All Name Variants", "Bibtex Citation Keys",
            "Reuse Count", "Modality(ies)", "OSI Layers",
            "Collection Environment", "Availability URL", "Merge Reason",
        ])
        for ds in canonical:
            writer.writerow([
                ds.canonical_name,
                "; ".join(ds.all_names) if len(ds.all_names) > 1 else "",
                ", ".join(ds.bibtex_keys),
                ds.reuse_count,
                ds.modalities,
                ds.osi_layers,
                "; ".join(ds.environments),
                ds.availability_url,
                ds.merge_reason,
            ])


def _write_consolidated_papers_bibtex(corpus_obj, results_dir):
    """Write consolidated_papers.csv and consolidated_bibtex.csv.

    Filters master CSVs to only include papers classified as wireless.
    """
    import csv as _csv
    import sqlite3
    from pathlib import Path

    from wireless_taxonomy.db import connect

    conn = connect(str(corpus_obj.db_path))
    conn.row_factory = sqlite3.Row

    # Get wireless paper IDs
    wireless_ids = set(
        row["paper_id"]
        for row in conn.execute(
            "SELECT paper_id FROM wireless_candidate_predictions WHERE label IN ('yes', 'maybe')"
        ).fetchall()
    )

    # Map bibtex_key -> paper_id
    key_to_pid = {
        row["citation_key"]: row["paper_id"]
        for row in conn.execute("SELECT paper_id, citation_key FROM bibtex_entries").fetchall()
    }
    conn.close()

    # Filter master_papers.csv
    master_papers_path = results_dir / "master_papers.csv"
    if master_papers_path.exists():
        with master_papers_path.open(encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        wireless_papers = [
            r for r in rows
            if key_to_pid.get(r.get("Bibtex Citation Key", "")) in wireless_ids
        ]

        out = results_dir / "consolidated_papers.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(wireless_papers)

    # Filter master_bibtex.csv
    master_bibtex_path = results_dir / "master_bibtex.csv"
    if master_bibtex_path.exists():
        with master_bibtex_path.open(encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        wireless_bibtex = [
            r for r in rows
            if key_to_pid.get(r.get("Bibtex Citation Key", "")) in wireless_ids
        ]

        out = results_dir / "consolidated_bibtex.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(wireless_bibtex)


def _write_pdf_only_datasets(results_dir):
    """Write consolidated_datasets_pdf_only.csv excluding abstract-only entries."""
    import csv as _csv
    import json
    from pathlib import Path

    results_dir = Path(results_dir)

    # Build bibtex_key -> extraction_source mapping
    def normalize_title(value: str) -> str:
        return " ".join(value.lower().split())

    title_to_final_key: dict[str, str] = {}
    papers_path = results_dir / "consolidated_papers.csv"
    if papers_path.exists():
        with papers_path.open(encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                title = normalize_title(row.get("Paper Title", ""))
                key = row.get("Bibtex Citation Key", "").strip()
                if title and key:
                    title_to_final_key[title] = key

    key_to_source: dict[str, str] = {}
    for f in sorted(results_dir.glob("*_raw.json")):
        if f.name.startswith(("master_", "consolidated_")):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for run in data.get("runs") or ([data] if "papers" in data else []):
                for paper in run.get("papers") or []:
                    key = paper.get("bibtex_key", "")
                    src = paper.get("extraction_source", "unknown")
                    if key:
                        key_to_source[key] = src
                    final_key = title_to_final_key.get(normalize_title(paper.get("title", "")))
                    if final_key:
                        key_to_source[final_key] = src
        except Exception:
            continue

    # Read consolidated and filter
    consolidated_path = results_dir / "consolidated_datasets.csv"
    if not consolidated_path.exists():
        return

    with consolidated_path.open(encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    pdf_sources = {"pdf", "pdf_text", "pdf_text_fallback"}
    pdf_only = []
    for r in rows:
        keys = [k.strip() for k in r["Bibtex Citation Keys"].split(",") if k.strip()]
        sources = set(key_to_source.get(k, "unknown") for k in keys)
        if sources & pdf_sources:
            pdf_only.append(r)

    out = results_dir / "consolidated_datasets_pdf_only.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pdf_only)
