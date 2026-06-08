#!/usr/bin/env python3
"""Coverage-evaluation harness — drives the wireless-taxonomy CLI end to end.

Runnable from the repo root. Drop in a sheet and it auto-detects which
conferences to evaluate:

    python scripts/evaluate_coverage.py \
        --gold "List of Papers.csv" \
        --classifier llm --drop-workshops \
        --db build/eval.sqlite --out-dir build/results

Pass `--gold` more than once to union several sheets, or `--venue-year` to pin an
explicit set instead of auto-detecting. When `--venue-year` is omitted the
harness derives the DBLP-ingestable conferences from the sheet(s) itself.

For each venue+year it calls `classify` (sheet-free: DBLP ingest -> DOI/abstract
backfill -> classify), which writes the **full** labelled CSV. It then runs the
single, DB-free `eval` command over all those CSVs to score them against the
curated sheet(s). The CLI is the single source of truth; this script only
orchestrates it and collects the reports, so the same commands can be reproduced
by hand.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{existing}" if existing else str(SRC_DIR)
    return env


def run_cli(args: list[str]) -> None:
    cmd = [sys.executable, "-m", "wireless_taxonomy.cli", *args]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=_cli_env(), cwd=REPO_ROOT)


def venue_years_from_gold(gold_paths: list[str]) -> list[tuple[str, int]]:
    """Derive the DBLP-ingestable conferences the gold sheet(s) contain."""
    from wireless_taxonomy.ingest.dblp import resolve_stream
    from wireless_taxonomy.ingest.gold import distinct_venue_years

    pairs = distinct_venue_years(gold_paths)
    ingestable: list[tuple[str, int]] = []
    skipped: list[str] = []
    for venue, year in pairs:
        if resolve_stream(venue) is None:
            skipped.append(f"{venue}:{year}")
            continue
        ingestable.append((venue, year))
    if skipped:
        print("skipped (no DBLP stream mapping): " + ", ".join(skipped), file=sys.stderr)
    return ingestable


def parse_venue_year(value: str) -> tuple[str, int]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"--venue-year must be VENUE:YEAR, got {value!r}")
    venue, _, year = value.rpartition(":")
    try:
        return venue.strip(), int(year)
    except ValueError as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(f"Invalid year in {value!r}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--gold",
        dest="gold",
        action="append",
        required=True,
        metavar="SHEET",
        help="Curated gold sheet (csv/xlsx). Repeat to evaluate several sheets at once.",
    )
    parser.add_argument(
        "--venue-year",
        dest="venue_years",
        action="append",
        type=parse_venue_year,
        metavar="VENUE:YEAR",
        help="Conference-year to evaluate (repeatable). Omit to auto-detect from the gold sheet(s).",
    )
    parser.add_argument("--classifier", choices=["llm", "keyword"], default="llm", help="Classifier to score.")
    parser.add_argument("--pass", dest="pass_mode", choices=["high", "low"], default="high")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.92)
    parser.add_argument("--drop-workshops", action="store_true", help="Drop curated papers absent from main proceedings.")
    parser.add_argument("--no-resolve-dois", action="store_true", help="Skip programmatic DOI backfill.")
    parser.add_argument("--db", default="build/eval.sqlite", help="SQLite work DB (shared across all conferences).")
    parser.add_argument("--out-dir", default="build/results", help="Directory for per-conference lists and the report.")
    parser.add_argument("--source", default="dblp", help="Paper-list source for classify.")
    args = parser.parse_args(argv)

    venue_years = args.venue_years or venue_years_from_gold(args.gold)
    if not venue_years:
        parser.error(
            "No conferences to evaluate: the gold sheet(s) yielded no DBLP-ingestable "
            "venue/year pairs. Pass --venue-year explicitly."
        )
    print(f"Evaluating {len(venue_years)} conference-year(s): "
          + ", ".join(f"{v}:{y}" for v, y in venue_years))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    use_llm_flag = "--llm" if args.classifier == "llm" else "--no-llm"
    resolve_flag = "--no-resolve-dois" if args.no_resolve_dois else "--resolve-dois"

    # 1) Sheet-free classify loop -> one full labelled CSV per conference-year.
    classified_csvs: list[Path] = []
    for venue, year in venue_years:
        slug = f"{venue.replace(' ', '_')}_{year}"
        csv_path = out_dir / f"{slug}.csv"
        run_cli(
            [
                "classify",
                "--venue", venue,
                "--years", str(year),
                use_llm_flag,
                resolve_flag,
                "--source", args.source,
                "--db", str(db_path),
                "--json", str(out_dir / f"{slug}.json"),
                "--csv", str(csv_path),
            ]
        )
        classified_csvs.append(csv_path)

    # 2) DB-free snapshot eval: score the full labelled CSVs against the sheet(s).
    eval_args = ["eval", "--pass", args.pass_mode, "--fuzzy-threshold", str(args.fuzzy_threshold)]
    for csv_path in classified_csvs:
        eval_args += ["--classified", str(csv_path)]
    for gold_path in args.gold:
        eval_args += ["--gold", gold_path]
    eval_args += ["--out", str(out_dir / "report.json"), "--md", str(out_dir / "report.md")]
    eval_args.append("--drop-workshops" if args.drop_workshops else "--keep-workshops")
    run_cli(eval_args)

    print(f"\nDone. Reports in {out_dir}/report.md and {out_dir}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
