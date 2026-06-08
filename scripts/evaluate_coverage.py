#!/usr/bin/env python3
"""Coverage-evaluation harness — drives the wireless-taxonomy CLI end to end.

Runnable from the repo root:

    python scripts/evaluate_coverage.py \
        --gold "List of Papers.csv" \
        --venue-year SIGCOMM:2024 --venue-year IMC:2023 --venue-year NSDI:2024 \
        --classifier llm --drop-workshops \
        --db build/eval.sqlite --out-dir build/results

For each venue+year it calls `classify-conference` (sheet-free: DBLP ingest ->
DOI/abstract backfill -> classify), then imports the gold sheet once and runs
`eval-overlap` to score the automated set against the curated sheet. The CLI is
the single source of truth; this script only orchestrates it and collects the
reports, so the same commands can be reproduced by hand.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

# Convenience default — the CS venues with clean DBLP main-track TOCs.
DEFAULT_VENUE_YEARS = [
    ("SIGCOMM", 2022),
    ("SIGCOMM", 2023),
    ("SIGCOMM", 2024),
    ("IMC", 2023),
    ("IMC", 2024),
    ("IMC", 2025),
    ("NSDI", 2023),
    ("NSDI", 2024),
    ("NSDI", 2025),
]


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{existing}" if existing else str(SRC_DIR)
    return env


def run_cli(args: list[str]) -> None:
    cmd = [sys.executable, "-m", "wireless_taxonomy.cli", *args]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=_cli_env(), cwd=REPO_ROOT)


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
    parser.add_argument("--gold", required=True, help="Curated gold sheet (csv/xlsx) of wireless papers.")
    parser.add_argument(
        "--venue-year",
        dest="venue_years",
        action="append",
        type=parse_venue_year,
        metavar="VENUE:YEAR",
        help="Conference-year to evaluate (repeatable). Defaults to the CS venue set.",
    )
    parser.add_argument("--classifier", choices=["llm", "keyword"], default="llm", help="Classifier to score.")
    parser.add_argument("--pass", dest="pass_mode", choices=["high", "low"], default="high")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.92)
    parser.add_argument("--drop-workshops", action="store_true", help="Drop curated papers absent from main proceedings.")
    parser.add_argument("--no-resolve-dois", action="store_true", help="Skip programmatic DOI backfill.")
    parser.add_argument("--db", default="build/eval.sqlite", help="SQLite DB path (shared across all conferences).")
    parser.add_argument("--out-dir", default="build/results", help="Directory for per-conference lists and the report.")
    parser.add_argument("--source", default="dblp", help="Paper-list source for classify-conference.")
    args = parser.parse_args(argv)

    venue_years = args.venue_years or DEFAULT_VENUE_YEARS
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    use_llm_flag = "--llm" if args.classifier == "llm" else "--no-llm"
    resolve_flag = "--no-resolve-dois" if args.no_resolve_dois else "--resolve-dois"

    # 1) Sheet-free classify loop, one conference at a time, into a shared DB.
    for venue, year in venue_years:
        slug = f"{venue.replace(' ', '_')}_{year}"
        run_cli(
            [
                "classify-conference",
                "--venue", venue,
                "--year", str(year),
                use_llm_flag,
                resolve_flag,
                "--pass", args.pass_mode,
                "--source", args.source,
                "--db", str(db_path),
                "--out", str(out_dir / f"{slug}.json"),
                "--csv", str(out_dir / f"{slug}.csv"),
            ]
        )

    # 2) Import the curated sheet once (matched per venue/year inside the DB).
    run_cli(["import-gold", "--path", args.gold, "--db", str(db_path)])

    # 3) Score the automated set against the curated sheet.
    eval_args = [
        "eval-overlap",
        "--classifier", args.classifier,
        "--pass", args.pass_mode,
        "--fuzzy-threshold", str(args.fuzzy_threshold),
        "--db", str(db_path),
        "--out", str(out_dir / "report.json"),
        "--md", str(out_dir / "report.md"),
    ]
    eval_args.append("--drop-workshops" if args.drop_workshops else "--keep-workshops")
    run_cli(eval_args)

    print(f"\nDone. Reports in {out_dir}/report.md and {out_dir}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
