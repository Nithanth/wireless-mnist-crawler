"""CLI command: reconcile-datasets — entity resolution postprocessing."""

import csv as _csv
import json
from pathlib import Path
from typing import Optional

import typer

from wireless_taxonomy.postprocess.entity_resolution import DatasetRecord, reconcile


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
        out: Optional[str] = typer.Option(
            None, "--out", help="Write JSON report here."
        ),
    ) -> None:
        """Post-merge entity resolution: flag datasets that are likely the same.

        \b
        Two strategies:
          1. URL/DOI dedup — high confidence (0.95). Datasets sharing an
             availability URL or DOI are near-certainly the same artifact.
          2. Similarity flagging — medium confidence (≤0.80). Normalized name +
             modality + OSI layer similarity surfaces candidates for human review.

        \b
        Input: either --csv (merged datasets CSV) or --json (raw extraction JSON).
        If both are given, records are combined for broader coverage.

        \b
        Future: LLM confirmation step for candidate pairs (not yet implemented).
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

        matches = reconcile(
            records,
            url_dedup=not no_url,
            similarity=not no_similarity,
            similarity_name_threshold=name_threshold,
            similarity_combined_threshold=combined_threshold,
        )

        if not matches:
            typer.echo("No potential duplicates found.")
            raise typer.Exit()

        # Report
        url_matches = [m for m in matches if m.method == "url_dedup"]
        sim_matches = [m for m in matches if m.method == "similarity"]

        if url_matches:
            typer.echo(f"\n{'─'*60}")
            typer.echo(f"URL/DOI MATCHES (high confidence): {len(url_matches)}")
            typer.echo(f"{'─'*60}")
            for m in url_matches:
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

        typer.echo(f"\nTotal: {len(url_matches)} URL/DOI + {len(sim_matches)} similarity = {len(matches)} candidates")

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
