"""CLI command: reconcile-datasets — entity resolution postprocessing."""

import csv as _csv
import json
from pathlib import Path
from typing import Any, Optional

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

    # Handle multiple formats:
    #  1. A dict with "runs" key: {venue, years, runs: [run, ...]}
    #  2. A list of such dicts (master_raw.json)
    #  3. A list containing a mix of dicts and nested lists (merge edge case)
    #  4. A single run dict with "papers" directly

    def _iter_runs(obj: Any) -> list[dict]:
        """Flatten any shape into a list of run dicts (each has 'papers')."""
        if isinstance(obj, dict):
            if "papers" in obj:
                return [obj]
            return obj.get("runs", [])
        if isinstance(obj, list):
            out: list[dict] = []
            for item in obj:
                out.extend(_iter_runs(item))
            return out
        return []

    for run in _iter_runs(data):
        if not isinstance(run, dict):
            continue
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
        llm_workers: int = typer.Option(
            4, "--llm-workers", min=1, max=16, help="Concurrent LLM confirmation calls."
        ),
        max_llm_pairs: int = typer.Option(
            500, "--max-llm-pairs", help="Cap on pairs sent to the LLM (highest-similarity first); overflow is flagged 'llm_skipped' for review. 0 = unlimited."
        ),
        out: Optional[str] = typer.Option(
            None, "--out", help="Write JSON report here."
        ),
        consolidated: Optional[str] = typer.Option(
            None, "--consolidated", help="Write consolidated (deduplicated) datasets CSV."
        ),
        review_csv: Optional[str] = typer.Option(
            None, "--review-csv",
            help="Write human-review candidates (LLM-unsure + similarity-only pairs) to a CSV "
                 "with a blank 'decision' column (merge/keep) for manual annotation.",
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
            llm_workers=llm_workers,
            llm_max_pairs=None if max_llm_pairs <= 0 else max_llm_pairs,
        )

        # Group matches by method
        url_matches = [m for m in matches if m.method == "url_dedup"]
        llm_yes = [m for m in matches if m.method == "llm_confirmed"]
        llm_unsure = [m for m in matches if m.method in ("llm_unsure", "llm_skipped")]
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

        # Human-review CSV: everything that is NOT auto-merged, with a blank
        # decision column the reviewer fills in (merge / keep).
        review_candidates = llm_unsure + sim_matches
        if review_csv and review_candidates:
            rp = Path(review_csv)
            rp.parent.mkdir(parents=True, exist_ok=True)
            with rp.open("w", newline="", encoding="utf-8") as fh:
                writer = _csv.writer(fh)
                writer.writerow([
                    "decision (merge/keep)", "confidence", "method", "reason",
                    "name_a", "bibtex_keys_a", "modalities_a", "osi_a", "env_a", "url_a",
                    "name_b", "bibtex_keys_b", "modalities_b", "osi_b", "env_b", "url_b",
                ])
                for m in sorted(review_candidates, key=lambda m: -m.confidence):
                    writer.writerow([
                        "", f"{m.confidence:.2f}", m.method, m.reason,
                        m.a.name, ", ".join(m.a.bibtex_keys), m.a.modalities, m.a.osi_layers, m.a.environment, m.a.availability_url,
                        m.b.name, ", ".join(m.b.bibtex_keys), m.b.modalities, m.b.osi_layers, m.b.environment, m.b.availability_url,
                    ])
            typer.echo(f"\nReview CSV: {rp} ({len(review_candidates)} pairs to review)")
        elif review_csv:
            typer.echo("\nNo review candidates — nothing written to --review-csv.")

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
