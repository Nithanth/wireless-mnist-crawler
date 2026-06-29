
import csv as _csv
import glob as _glob
import json
import re as _re
from pathlib import Path

import typer


def register(app: typer.Typer) -> None:
    @app.command("merge-results")
    def merge_results(
        results_dir: str = typer.Option("./src/results", "--dir", help="Directory containing per-venue/year CSV + JSON files."),
        out: str = typer.Option("./src/results", "--out", help="Output directory for merged master files."),
        min_corpus_reuse: int = typer.Option(
            1, "--min-corpus-reuse",
            help="Only include datasets mentioned in at least this many papers across the entire corpus. "
                 "Set to 2+ to filter to cross-paper reuse only.",
        ),
    ) -> None:
        """Merge all per-venue/year CSVs and JSONs into master files.

        Reads all *_papers.csv, *_bibtex.csv, *_datasets.csv, and *_raw.json files
        from --dir and produces:
          master_papers.csv, master_bibtex.csv, master_datasets.csv, master_raw.json

        With --min-corpus-reuse=2 (the default), only datasets referenced by at
        least 2 papers in the merged corpus survive into master_datasets.csv.
        """
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

        def _norm_ds(name: str) -> str:
            n = name.strip().lower()
            n = _re.sub(r"[- ]+(dataset|traces|measurements|data|evaluation)s?$", "", n)
            return _re.sub(r"\s+", " ", n)

        # Map normalized name -> canonical (first-seen) raw name
        norm_to_canonical: dict[str, str] = {}
        corpus_paper_counts: dict[str, int] = {}  # keyed by normalized name
        for f in sorted(_glob.glob(str(src / "*_raw.json"))):
            try:
                run_data = json.loads(Path(f).read_text(encoding="utf-8"))
                # Structure: {"venue": ..., "years": [...], "runs": [{"papers": [...]}]}
                for run in run_data.get("runs") or []:
                    for paper in run.get("papers") or []:
                        seen_in_paper: set[str] = set()
                        for ds in paper.get("datasets") or []:
                            ds_name = ds.get("name", "")
                            if not ds_name:
                                continue
                            norm = _norm_ds(ds_name)
                            if norm not in norm_to_canonical:
                                norm_to_canonical[norm] = ds_name
                            if norm not in seen_in_paper:
                                seen_in_paper.add(norm)
                                corpus_paper_counts[norm] = corpus_paper_counts.get(norm, 0) + 1
            except Exception:
                pass

        # Merge rows whose names normalize to the same thing
        for name in list(merged_ds):
            norm = _norm_ds(name)
            canonical = norm_to_canonical.get(norm, name)
            if canonical != name and canonical in merged_ds:
                existing = merged_ds[canonical]
                try:
                    old_count = int(existing.get("Number of Papers using Dataset") or 0)
                    new_count = int(merged_ds[name].get("Number of Papers using Dataset") or 0)
                    existing["Number of Papers using Dataset"] = str(old_count + new_count)
                except ValueError:
                    pass
                if not existing.get("Bibtex Citation Key") and merged_ds[name].get("Bibtex Citation Key"):
                    existing["Bibtex Citation Key"] = merged_ds[name]["Bibtex Citation Key"]
                del merged_ds[name]
            elif canonical != name and canonical not in merged_ds:
                merged_ds[canonical] = merged_ds.pop(name)
                merged_ds[canonical]["Dataset Name"] = canonical

        # Update paper counts from corpus-wide scan and apply reuse filter
        total_before_filter = len(merged_ds)
        for name in list(merged_ds):
            norm = _norm_ds(name)
            corpus_count = corpus_paper_counts.get(norm, 1)
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
