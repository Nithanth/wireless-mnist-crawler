"""CLI command: reconcile-datasets — entity resolution postprocessing."""

import csv as _csv
import json
from pathlib import Path
from typing import Optional

import typer

from wireless_taxonomy.postprocess.entity_resolution import (
    CanonicalDataset,
    DatasetRecord,
    consolidate,
    reconcile,
)


def _load_datasets_from_csv(path: Path) -> list[DatasetRecord]:
    """Load dataset records from a merged datasets CSV."""
    records: list[DatasetRecord] = []
    with path.open(encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            name = (row.get("Dataset Name") or row.get("dataset name") or "").strip()
            if not name:
                continue
            key_field = (
                row.get("Bibtex Citation Key")
                or row.get("bibtex citation key")
                or ""
            ).strip()
            keys = [k.strip() for k in key_field.split(",") if k.strip()]
            records.append(DatasetRecord(
                name=name,
                bibtex_keys=keys,
                modalities=(
                    row.get("Modality(ies)")
                    or row.get("modality(ies)")
                    or ""
                ).strip(),
                osi_layers=(
                    row.get("OSI Layer (L1-L7)")
                    or row.get("OSI layer at which dataset is measured")
                    or ""
                ).strip(),
                environment=(
                    row.get("Collection Environment")
                    or row.get("Collection environment")
                    or ""
                ).strip(),
                availability_url=(
                    row.get("Availability URL")
                    or ""
                ).strip(),
                availability_notes=(
                    row.get("Annotations on Availability")
                    or row.get("Availability Annotations")
                    or ""
                ).strip(),
            ))
    return records


def _load_datasets_from_json(path: Path) -> list[DatasetRecord]:
    """Load dataset records from a raw JSON extraction file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    records: list[DatasetRecord] = []

    # Handle both single-file and master_raw.json (list of runs)
    runs_list = data if isinstance(data, list) else [data]

    for run_data in runs_list:
        for run in run_data.get("runs", []):
            for paper in run.get("papers", []):
                key = paper.get("bibtex_key", "")
                for ds in paper.get("datasets", []):
                    name = ds.get("name", "").strip()
                    if not name:
                        continue
                    records.append(DatasetRecord(
                        name=name,
                        bibtex_keys=[key] if key else [],
                        modalities="; ".join(ds.get("modalities", [])),
                        osi_layers="; ".join(ds.get("osi_layers", [])),
                        environment=ds.get("collection_environment", ""),
                        availability_url=ds.get("availability_url", ""),
                        availability_notes=ds.get("availability_notes", ""),
                    ))

    return records


def _write_consolidated_csv(canonical: list[CanonicalDataset], path: Path) -> None:
    """Write the consolidated (deduplicated) dataset list to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.writer(fh)
        writer.writerow([
            "Canonical Name",
            "All Name Variants",
            "Bibtex Citation Keys",
            "Reuse Count",
            "Modality(ies)",
            "OSI Layers",
            "Collection Environment",
            "Availability URL",
            "Merge Reason",
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


def register(app: typer.Typer) -> None:
    @app.command("reconcile-datasets")
    def reconcile_datasets(
        datasets_csv: Optional[str] = typer.Option(
            None, "--csv", help="Merged datasets CSV (e.g. master_datasets.csv)."
        ),
        raw_json: Optional[str] = typer.Option(
            None, "--json", help="Raw JSON extraction output (e.g. master_raw.json)."
        ),
        name_threshold: float = typer.Option(
            0.75, "--name-threshold", help="Minimum name similarity ratio to flag."
        ),
        combined_threshold: float = typer.Option(
            0.70, "--combined-threshold", help="Minimum combined (name+modality+OSI) similarity."
        ),
        no_url: bool = typer.Option(
            False, "--no-url", help="Disable URL/DOI dedup (only run similarity)."
        ),
        no_similarity: bool = typer.Option(
            False, "--no-similarity", help="Disable similarity flagging (only run URL/DOI dedup)."
        ),
        llm_confirm: bool = typer.Option(
            False, "--llm-confirm", help="Use LLM to confirm/reject similarity candidates."
        ),
        out: Optional[str] = typer.Option(
            None, "--out", help="Write JSON report here."
        ),
        consolidated: Optional[str] = typer.Option(
            None, "--consolidated", help="Write consolidated (deduplicated) datasets CSV."
        ),
    ) -> None:
        """Post-merge entity resolution: flag datasets that are likely the same.

        \b
        Three strategies (in order):
          1. URL/DOI dedup — high confidence (0.95). Datasets sharing an
             availability URL or DOI are near-certainly the same artifact.
          2. Similarity flagging — medium confidence (≤0.80). Normalized name +
             modality + OSI layer similarity surfaces candidates.
          3. LLM confirmation (--llm-confirm) — the LLM reviews similarity
             candidates and returns yes/no/unsure verdicts. "no" pairs are
             dropped, "yes" are auto-merged, "unsure" are flagged for review.

        \b
        Input: either --csv (merged datasets CSV) or --json (raw extraction JSON).
        If both are given, records are combined for broader coverage.

        \b
        Use --consolidated to write a deduplicated dataset list with proper
        reuse counts (needed for downstream metrics/figures).
        """
        if not datasets_csv and not raw_json:
            typer.echo("Provide --csv and/or --json input.", err=True)
            raise typer.Exit(1)

        records: list[DatasetRecord] = []
        if datasets_csv:
            p = Path(datasets_csv)
            if not p.exists():
                typer.echo(f"File not found: {p}", err=True)
                raise typer.Exit(1)
            records.extend(_load_datasets_from_csv(p))
        if raw_json:
            p = Path(raw_json)
            if not p.exists():
                typer.echo(f"File not found: {p}", err=True)
                raise typer.Exit(1)
            records.extend(_load_datasets_from_json(p))

        typer.echo(f"Loaded {len(records)} dataset records.")

        # When LLM confirm is active, use more aggressive similarity thresholds
        # for candidate generation (the LLM handles precision filtering).
        effective_name = name_threshold
        effective_combined = combined_threshold
        if llm_confirm and name_threshold == 0.75:
            effective_name = 0.60
        if llm_confirm and combined_threshold == 0.70:
            effective_combined = 0.55

        matches = reconcile(
            records,
            url_dedup=not no_url,
            similarity=not no_similarity,
            llm_confirm=llm_confirm,
            similarity_name_threshold=effective_name,
            similarity_combined_threshold=effective_combined,
        )

        # Group matches by method
        url_matches = [m for m in matches if m.method == "url_dedup"]
        llm_yes = [m for m in matches if m.method == "llm_confirmed"]
        llm_unsure = [m for m in matches if m.method == "llm_unsure"]
        sim_matches = [m for m in matches if m.method == "similarity"]

        if url_matches:
            typer.echo(f"\n{'─'*60}")
            typer.echo(f"URL/DOI MATCHES (auto-merge): {len(url_matches)}")
            typer.echo(f"{'─'*60}")
            for m in url_matches:
                typer.echo(f"\n  [{m.confidence:.2f}] {m.reason}")
                typer.echo(f"    A: {m.a.name}  [{', '.join(m.a.bibtex_keys)}]")
                typer.echo(f"    B: {m.b.name}  [{', '.join(m.b.bibtex_keys)}]")

        if llm_yes:
            typer.echo(f"\n{'─'*60}")
            typer.echo(f"LLM CONFIRMED (auto-merge): {len(llm_yes)}")
            typer.echo(f"{'─'*60}")
            for m in llm_yes:
                typer.echo(f"\n  [{m.confidence:.2f}] {m.reason}")
                typer.echo(f"    A: {m.a.name}  [{', '.join(m.a.bibtex_keys)}]")
                typer.echo(f"    B: {m.b.name}  [{', '.join(m.b.bibtex_keys)}]")

        if llm_unsure:
            typer.echo(f"\n{'─'*60}")
            typer.echo(f"LLM UNSURE (human review): {len(llm_unsure)}")
            typer.echo(f"{'─'*60}")
            for m in llm_unsure:
                typer.echo(f"\n  [{m.confidence:.2f}] {m.reason}")
                typer.echo(f"    A: {m.a.name}  [{', '.join(m.a.bibtex_keys)}]")
                typer.echo(f"    B: {m.b.name}  [{', '.join(m.b.bibtex_keys)}]")

        if sim_matches:
            typer.echo(f"\n{'─'*60}")
            typer.echo(f"SIMILARITY CANDIDATES (review): {len(sim_matches)}")
            typer.echo(f"{'─'*60}")
            for m in sim_matches:
                typer.echo(f"\n  [{m.confidence:.2f}] {m.reason}")
                typer.echo(f"    A: {m.a.name}  [{', '.join(m.a.bibtex_keys)}]")
                typer.echo(f"       mod={m.a.modalities[:60]}  osi={m.a.osi_layers}  env={m.a.environment}")
                typer.echo(f"    B: {m.b.name}  [{', '.join(m.b.bibtex_keys)}]")
                typer.echo(f"       mod={m.b.modalities[:60]}  osi={m.b.osi_layers}  env={m.b.environment}")

        total = len(url_matches) + len(llm_yes) + len(llm_unsure) + len(sim_matches)
        typer.echo(f"\nTotal: {total} candidates")
        if llm_confirm:
            typer.echo(f"  URL/DOI auto-merge: {len(url_matches)}")
            typer.echo(f"  LLM confirmed: {len(llm_yes)}")
            typer.echo(f"  LLM unsure (needs review): {len(llm_unsure)}")
        else:
            typer.echo(f"  URL/DOI: {len(url_matches)}, Similarity: {len(sim_matches)}")

        # Consolidated output
        if consolidated:
            canonical = consolidate(records, matches)
            _write_consolidated_csv(canonical, Path(consolidated))
            reused = [d for d in canonical if d.reuse_count > 1]
            typer.echo(f"\nConsolidated: {len(canonical)} unique datasets "
                       f"({len(reused)} reused across multiple papers)")
            typer.echo(f"Wrote: {consolidated}")

        if out:
            report = {
                "total_records": len(records),
                "matches": [
                    {
                        "method": m.method,
                        "confidence": m.confidence,
                        "reason": m.reason,
                        "a": {"name": m.a.name, "keys": m.a.bibtex_keys},
                        "b": {"name": m.b.name, "keys": m.b.bibtex_keys},
                    }
                    for m in matches
                ],
            }
            out_path = Path(out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            typer.echo(f"Wrote report: {out_path}")
