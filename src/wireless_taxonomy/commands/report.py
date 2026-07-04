"""Corpus-wide run report: per-venue/year stats + dataset summary.

Reads the per-venue/year raw JSONs (extraction results), coverage JSONs
(OA/PDF availability), and the consolidated/master dataset CSVs, then echoes
a summary table and writes a Markdown report suitable for inclusion in the
project write-up.
"""

import csv as _csv
import glob as _glob
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer


def _load_raw_runs(results_dir: Path) -> list[dict[str, Any]]:
    """Load every per-venue/year *_raw.json (skipping master_*)."""
    entries: list[dict[str, Any]] = []
    for f in sorted(_glob.glob(str(results_dir / "*_raw.json"))):
        name = Path(f).name
        if name.startswith(("master_", "consolidated_")):
            continue
        try:
            data = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        venue = data.get("venue", "?")
        for run in data.get("runs") or []:
            entries.append({
                "venue": venue,
                "year": run.get("year", "?"),
                "papers": run.get("papers") or [],
            })
    return entries


def _load_coverage(cov_dirs: list[Path]) -> dict[tuple[str, int], dict[str, Any]]:
    """Load cov_<VENUE>_<YEAR>.json files keyed by (venue, year)."""
    coverage: dict[tuple[str, int], dict[str, Any]] = {}
    for d in cov_dirs:
        for f in sorted(_glob.glob(str(d / "cov_*.json"))):
            try:
                data = json.loads(Path(f).read_text(encoding="utf-8"))
            except Exception:
                continue
            venue = data.get("venue", "?")
            for run in data.get("runs") or []:
                year = run.get("year")
                if year is None:
                    continue
                coverage[(str(venue).lower(), int(year))] = {
                    "total": run.get("total_papers", 0),
                    "fetchable": run.get("fetchable", 0),
                    "by_source": dict(run.get("by_source") or {}),
                }
    return coverage


def _provider_census(coverage: dict[tuple[str, int], dict[str, Any]]) -> tuple[Counter, int, int]:
    """Aggregate PDF-retrieval attribution across all venue-years.

    Returns (per-provider fetch counts, total fetchable, total papers) so the
    report can show which providers actually delivered the corpus — e.g. how
    much came from free OA indexes vs. paid web search.
    """
    census: Counter = Counter()
    fetchable = 0
    total = 0
    for cov in coverage.values():
        total += cov.get("total", 0) or 0
        fetchable += cov.get("fetchable", 0) or 0
        for provider, n in (cov.get("by_source") or {}).items():
            census[provider or "unknown"] += n
    return census, fetchable, total


def _venue_year_stats(entries: list[dict], coverage: dict) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for entry in sorted(entries, key=lambda e: (str(e["venue"]).lower(), e["year"])):
        papers = entry["papers"]
        sources = Counter((p.get("extraction_source") or "unknown") for p in papers)
        n_datasets = sum(len(p.get("datasets") or []) for p in papers)
        open_datasets = sum(
            1 for p in papers for ds in (p.get("datasets") or []) if ds.get("availability") is True
        )
        introduced = sum(
            1 for p in papers for ds in (p.get("datasets") or [])
            if ds.get("relationship_type") == "introduced"
        )
        cov = coverage.get((str(entry["venue"]).lower(), int(entry["year"])), {})
        stats.append({
            "venue": entry["venue"],
            "year": entry["year"],
            "proceedings_total": cov.get("total", ""),
            "oa_fetchable": cov.get("fetchable", ""),
            "wireless_papers": len(papers),
            "pdf": sources.get("pdf", 0) + sources.get("pdf_text_fallback", 0),
            "abstract": sources.get("abstract", 0),
            "datasets": n_datasets,
            "open_datasets": open_datasets,
            "introduced": introduced,
        })
    return stats


def _model_census(entries: list[dict]) -> tuple[Counter, Counter]:
    """Census of actual models used per stage, for homogeneity auditing.

    Extraction census comes from the per-paper ``model_version`` in the raw
    JSONs. Classification census comes from the taxonomy DB when present.
    A clean single-model corpus shows exactly one model per stage; anything
    else means fragmentation (e.g. LLM fallbacks fired mid-run).
    """
    extraction = Counter(
        (p.get("model_version") or "unrecorded")
        for e in entries for p in e["papers"]
    )
    classification: Counter = Counter()
    try:
        import sqlite3

        for db in _glob.glob("**/taxonomy.sqlite", recursive=True):
            conn = sqlite3.connect(db)
            try:
                for model, n in conn.execute(
                    "SELECT model_version, COUNT(*) FROM wireless_candidate_predictions GROUP BY model_version"
                ).fetchall():
                    classification[model or "unrecorded"] += n
            finally:
                conn.close()
            break
    except Exception:
        pass
    return classification, extraction


def _echo_census(name: str, census: Counter, lines_md: list[str]) -> None:
    if not census:
        return
    real_models = [m for m in census if m not in ("unrecorded", "unknown-cached", "")]
    mixed = len(real_models) > 1
    typer.echo(f"\n{name} model census:")
    lines_md.append(f"\n### {name} model census\n")
    for model, n in census.most_common():
        typer.echo(f"  {n:>6}×  {model}")
        lines_md.append(f"- `{model}`: {n}")
    if mixed:
        warning = (
            f"⚠ MIXED MODELS in {name.lower()} — corpus is fragmented across "
            f"{len(real_models)} models. For a clean experiment, re-run the "
            "minority papers under the majority model (caches make this cheap)."
        )
        typer.echo(warning)
        lines_md.append(f"\n**{warning}**")


def _top_reused(results_dir: Path, limit: int = 20) -> list[tuple[str, int, str]]:
    """Top reused datasets: prefer consolidated CSV, fall back to master CSV."""
    consolidated = results_dir / "consolidated_datasets.csv"
    master = results_dir / "master_datasets.csv"
    rows: list[tuple[str, int, str]] = []
    if consolidated.exists():
        with consolidated.open(newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                try:
                    count = int(row.get("Reuse Count") or 0)
                except ValueError:
                    count = 0
                rows.append((row.get("Canonical Name", ""), count, row.get("Bibtex Citation Keys", "")))
    elif master.exists():
        with master.open(newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                try:
                    count = int(row.get("Number of Papers using Dataset") or 0)
                except ValueError:
                    count = 0
                rows.append((row.get("Dataset Name", ""), count, row.get("Bibtex Citation Key", "")))
    rows.sort(key=lambda r: -r[1])
    return [r for r in rows if r[1] >= 2][:limit]


def register(app: typer.Typer) -> None:
    @app.command("report")
    def report(
        results_dir: str = typer.Option("./src/results", "--dir", help="Directory with per-venue/year results."),
        cov_dir: str = typer.Option(".", "--cov-dir", help="Directory with cov_<VENUE>_<YEAR>.json coverage files."),
        out: str = typer.Option("./src/results/master_report.md", "--out", help="Markdown report output path."),
    ) -> None:
        """Summarize the whole corpus run: coverage, PDF yield, dataset stats.

        Echoes a per-venue/year table (papers, PDFs fetched, extraction sources,
        datasets found) plus corpus-wide totals and the top reused datasets, and
        writes the same content as a Markdown report for the project write-up.
        """
        results = Path(results_dir)
        entries = _load_raw_runs(results)
        if not entries:
            typer.echo(f"No *_raw.json results found in {results_dir}", err=True)
            raise typer.Exit(1)
        coverage = _load_coverage([Path(cov_dir), results])
        stats = _venue_year_stats(entries, coverage)

        header = (
            f"{'Venue':<10} {'Year':<6} {'Proc.':>6} {'OA':>5} {'Wireless':>9} "
            f"{'PDF':>5} {'Abs':>5} {'Datasets':>9} {'Open':>5} {'Introduced':>10}"
        )
        typer.echo("\n" + "═" * len(header))
        typer.echo("CORPUS RUN REPORT")
        typer.echo("═" * len(header))
        typer.echo(header)
        typer.echo("─" * len(header))
        lines_md = [
            "| Venue | Year | Proceedings | OA fetchable | Wireless papers | PDF-extracted | Abstract-only | Datasets | Open | Introduced |",
            "|-------|------|------------:|-------------:|----------------:|--------------:|--------------:|---------:|-----:|-----------:|",
        ]
        totals = Counter()
        for s in stats:
            typer.echo(
                f"{s['venue']:<10} {s['year']:<6} {str(s['proceedings_total']):>6} {str(s['oa_fetchable']):>5} "
                f"{s['wireless_papers']:>9} {s['pdf']:>5} {s['abstract']:>5} {s['datasets']:>9} "
                f"{s['open_datasets']:>5} {s['introduced']:>10}"
            )
            lines_md.append(
                f"| {s['venue']} | {s['year']} | {s['proceedings_total']} | {s['oa_fetchable']} "
                f"| {s['wireless_papers']} | {s['pdf']} | {s['abstract']} | {s['datasets']} "
                f"| {s['open_datasets']} | {s['introduced']} |"
            )
            for key in ("wireless_papers", "pdf", "abstract", "datasets", "open_datasets", "introduced"):
                totals[key] += s[key]

        typer.echo("─" * len(header))
        typer.echo(
            f"{'TOTAL':<10} {'':<6} {'':>6} {'':>5} {totals['wireless_papers']:>9} "
            f"{totals['pdf']:>5} {totals['abstract']:>5} {totals['datasets']:>9} "
            f"{totals['open_datasets']:>5} {totals['introduced']:>10}"
        )
        pdf_pct = 100.0 * totals["pdf"] / max(totals["wireless_papers"], 1)
        typer.echo(
            f"\nFull-text (PDF) extraction rate: {totals['pdf']}/{totals['wireless_papers']} "
            f"({pdf_pct:.1f}%) — the rest used title+abstract fallback."
        )

        provider_census, cov_fetchable, cov_total = _provider_census(coverage)
        provider_md: list[str] = []
        if provider_census:
            typer.echo(
                f"\nPDF retrieval by provider ({cov_fetchable}/{cov_total} proceedings papers resolved):"
            )
            provider_md += [
                "\n### PDF retrieval by provider\n",
                f"{cov_fetchable}/{cov_total} proceedings papers resolved to a fetchable PDF.\n",
                "| Provider | PDFs found | Share of resolved |",
                "|----------|-----------:|------------------:|",
            ]
            for provider, n in provider_census.most_common():
                share = 100.0 * n / max(cov_fetchable, 1)
                typer.echo(f"  {n:>6}×  {provider:<18} ({share:.1f}% of resolved)")
                provider_md.append(f"| {provider} | {n} | {share:.1f}% |")

        census_md: list[str] = []
        classification_census, extraction_census = _model_census(entries)
        _echo_census("Classification", classification_census, census_md)
        _echo_census("Extraction", extraction_census, census_md)

        top = _top_reused(results)
        if top:
            typer.echo(f"\nTop reused datasets (≥2 papers):")
            for name, count, keys in top:
                typer.echo(f"  {count:>3}×  {name}")

        md = [
            "# Corpus Run Report",
            "",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "## Per-venue/year results",
            "",
            *lines_md,
            f"| **TOTAL** | | | | **{totals['wireless_papers']}** | **{totals['pdf']}** "
            f"| **{totals['abstract']}** | **{totals['datasets']}** | **{totals['open_datasets']}** "
            f"| **{totals['introduced']}** |",
            "",
            f"- **Full-text (PDF) extraction rate:** {totals['pdf']}/{totals['wireless_papers']} ({pdf_pct:.1f}%)",
            "- Papers without a fetchable PDF fall back to title+abstract extraction",
            "  (`extraction_source` records which path each paper used).",
            "",
        ]
        if provider_md:
            md += ["## PDF retrieval attribution", *provider_md, ""]
        if census_md:
            md += ["## Model homogeneity audit", *census_md, ""]
        if top:
            md += [
                "## Top reused datasets (≥2 papers in corpus)",
                "",
                "| Uses | Dataset | Papers |",
                "|-----:|---------|--------|",
                *(f"| {count} | {name} | {keys} |" for name, count, keys in top),
                "",
            ]
        consolidated = results / "consolidated_datasets.csv"
        md += [
            "## Artifacts",
            "",
            "- `master_papers.csv` — all wireless-classified papers",
            "- `master_datasets.csv` — merged datasets (light name normalization)",
        ]
        if consolidated.exists():
            md.append(
                "- `consolidated_datasets.csv` — canonical deduplicated datasets "
                "(URL + LLM-confirmed merges only; source of truth for reuse metrics)"
            )
        md.append("")

        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(md), encoding="utf-8")
        typer.echo(f"\nWrote report: {out_path}")
