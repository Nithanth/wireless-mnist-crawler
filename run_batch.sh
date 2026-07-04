#!/usr/bin/env bash
set -uo pipefail

# Batch extraction: fetch-coverage → extract-datasets (wireless-only) per venue/year
#
# Usage:
#   ./run_batch.sh                                    # defaults: SIGCOMM,IMC,NSDI × 2022-2024
#   ./run_batch.sh --venues "NSDI,IMC" --years "2024,2025"
#   ./run_batch.sh --venues "MobiCom" --years "2022:2025"
#   ./run_batch.sh --fresh                            # clear old results + LLM cache
#   ./run_batch.sh --fresh-results                    # archive old results only
#   ./run_batch.sh --fresh-llm                        # clear LLM cache only
#
# The corpus is INCREMENTAL: run once for your initial set, then run again
# with new --venues/--years to grow it. Re-run `merge-results` afterward to
# recompute cross-corpus dataset reuse across the full union.

# Prevent macOS from sleeping while this script runs (display + idle + disk + system).
# caffeinate is killed automatically when this script exits.
if [[ "$(uname)" == "Darwin" ]] && command -v caffeinate &>/dev/null; then
  caffeinate -dims -w $$ &
fi

# Activate the project venv (package is pip-installed in editable mode).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

# ── Pre-flight checks ────────────────────────────────────────────────────────
# Verify required API keys and configuration before starting a potentially
# multi-hour run. Fail loudly now rather than 2 hours in.
echo "$(date '+%H:%M:%S') Running pre-flight checks..."
python - <<'PREFLIGHT'
import os, sys

RED   = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW= "\033[0;33m"
RESET = "\033[0m"

ok = True

def check(label, val, required=True, note=""):
    global ok
    if val:
        print(f"  {GREEN}✓{RESET}  {label}: {val[:6]}{'*' * max(0, len(val)-6) if len(val) > 6 else ''}{'  ' + note if note else ''}")
    elif required:
        print(f"  {RED}✗{RESET}  {label}: NOT SET  ← required{('  ' + note) if note else ''}")
        ok = False
    else:
        print(f"  {YELLOW}–{RESET}  {label}: not set{('  (optional: ' + note + ')') if note else ' (optional)'}")

# Load .env if present
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

print()
print("  LLM")
check("GEMINI_API_KEY (primary LLM)",
      os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
      required=True)
check("WIRELESS_TAXONOMY_LLM_PROVIDER",
      os.getenv("WIRELESS_TAXONOMY_LLM_PROVIDER", "google"),
      required=False, note="defaults to google")
fallbacks = os.getenv("WIRELESS_TAXONOMY_LLM_FALLBACKS", "")
if fallbacks.strip():
    print(f"  {YELLOW}!{RESET}  WIRELESS_TAXONOMY_LLM_FALLBACKS={fallbacks!r}  "
          f"← fallbacks active; for a homogeneous run remove this from .env")
else:
    print(f"  {GREEN}✓{RESET}  WIRELESS_TAXONOMY_LLM_FALLBACKS: not set (single-model run, good)")

print()
print("  PDF / OA retrieval")
check("UNPAYWALL_EMAIL",
      os.getenv("UNPAYWALL_EMAIL") or os.getenv("WIRELESS_TAXONOMY_UNPAYWALL_EMAIL"),
      required=True, note="required for Unpaywall OA lookup")
check("BRAVE_SEARCH_API_KEY",
      os.getenv("BRAVE_SEARCH_API_KEY"),
      required=False, note="needed for --web-search PDF fallback")
check("GOOGLE_CSE_API_KEY",
      os.getenv("GOOGLE_CSE_API_KEY"),
      required=False, note="needed for --web-search PDF fallback")
check("GOOGLE_CSE_ID",
      os.getenv("GOOGLE_CSE_ID"),
      required=False, note="needed for --web-search PDF fallback")

# Warn if --web-search keys are missing but caller may pass --web-search
brave = os.getenv("BRAVE_SEARCH_API_KEY", "")
cse_k = os.getenv("GOOGLE_CSE_API_KEY", "")
cse_i = os.getenv("GOOGLE_CSE_ID", "")
if not (brave or (cse_k and cse_i)):
    print(f"  {YELLOW}!{RESET}  No web-search keys set — --web-search flag will fall back to static providers only")

print()
print("  Database")
import pathlib
db = pathlib.Path(os.getenv("WIRELESS_TAXONOMY_DB_PATH", "taxonomy.sqlite"))
if db.exists():
    size_mb = db.stat().st_size / 1_048_576
    print(f"  {GREEN}✓{RESET}  DB: {db} ({size_mb:.1f} MB)")
else:
    print(f"  {YELLOW}–{RESET}  DB: {db} (will be created on first run)")

print()
if not ok:
    print(f"  {RED}Pre-flight FAILED — fix the above before running.{RESET}")
    sys.exit(1)
else:
    print(f"  {GREEN}Pre-flight passed.{RESET}")
PREFLIGHT

if [ $? -ne 0 ]; then
  echo "Aborting."
  exit 1
fi
echo ""

# Defaults
VENUES_STR="SIGCOMM,IMC,NSDI"
YEARS_STR="2022,2023,2024"
FRESH_RESULTS=false
FRESH_LLM=false
EXTRACT_FRESH_FLAG=""
WORKERS=6
WEB_SEARCH_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venues)        VENUES_STR="$2"; shift 2 ;;
    --years)         YEARS_STR="$2"; shift 2 ;;
    --workers)       WORKERS="$2"; shift 2 ;;
    --web-search)    WEB_SEARCH_FLAG="--web-search"; shift ;;
    --fresh)         FRESH_RESULTS=true; FRESH_LLM=true; shift ;;
    --fresh-results) FRESH_RESULTS=true; shift ;;
    --fresh-llm)     FRESH_LLM=true; shift ;;
    *)
      echo "Unknown flag: $1"
      echo "Usage: $0 [--venues \"NSDI,IMC\"] [--years \"2022:2025\"] [--workers N] [--web-search] [--fresh | --fresh-results | --fresh-llm]"
      exit 1 ;;
  esac
done

# Parse venues (comma-separated)
IFS=',' read -ra VENUES <<< "$VENUES_STR"

# Parse years (comma-separated or range with colon, e.g. "2022:2025")
YEARS=()
IFS=',' read -ra YEAR_PARTS <<< "$YEARS_STR"
for part in "${YEAR_PARTS[@]}"; do
  if [[ "$part" == *":"* ]]; then
    IFS=':' read -r start end <<< "$part"
    for ((y=start; y<=end; y++)); do YEARS+=("$y"); done
  else
    YEARS+=("$part")
  fi
done

TOTAL=$(( ${#VENUES[@]} * ${#YEARS[@]} ))
CURRENT=0
COMPLETED=()
FAILED=()
START_TIME=$(date +%s)

ts() { date "+%H:%M:%S"; }

# Format seconds as "1h 23m 45s" / "23m 45s" / "45s"
fmt_dur() {
  local s=$1 h m
  h=$(( s / 3600 )); m=$(( (s % 3600) / 60 )); s=$(( s % 60 ))
  if [ "$h" -gt 0 ]; then echo "${h}h ${m}m ${s}s"
  elif [ "$m" -gt 0 ]; then echo "${m}m ${s}s"
  else echo "${s}s"; fi
}

# ── Archive old results ──────────────────────────────────
if [ "$FRESH_RESULTS" = true ]; then
  RESULTS_DIR="./src/results"
  if ls "$RESULTS_DIR"/*_papers.csv "$RESULTS_DIR"/*_datasets.csv "$RESULTS_DIR"/*_bibtex.csv "$RESULTS_DIR"/*_raw.json "$RESULTS_DIR"/master_*.csv "$RESULTS_DIR"/master_*.json 2>/dev/null | head -1 > /dev/null 2>&1; then
    ARCHIVE_DIR="$RESULTS_DIR/archive_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$ARCHIVE_DIR"
    mv "$RESULTS_DIR"/*_papers.csv "$RESULTS_DIR"/*_datasets.csv "$RESULTS_DIR"/*_bibtex.csv "$RESULTS_DIR"/*_raw.json "$ARCHIVE_DIR/" 2>/dev/null || true
    mv "$RESULTS_DIR"/master_*.csv "$RESULTS_DIR"/master_*.json "$ARCHIVE_DIR/" 2>/dev/null || true
    echo "$(ts) Archived old results to $ARCHIVE_DIR/"
  else
    echo "$(ts) No old results to archive."
  fi
fi

# ── Clear LLM cache ──────────────────────────────────────
if [ "$FRESH_LLM" = true ]; then
  echo "$(ts) Clearing LLM classification + extraction cache..."
  python -m wireless_taxonomy.cli cache clear-section llm 2>/dev/null || true
  EXTRACT_FRESH_FLAG="--fresh"
  echo "$(ts) LLM cache cleared. Papers will be re-classified and re-extracted."
fi

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  BATCH RUN: ${#VENUES[@]} venues × ${#YEARS[@]} years = ${TOTAL} loops     ║"
echo "║  Venues: ${VENUES[*]}"
echo "║  Years:  ${YEARS[*]}"
echo "║  Fresh results: ${FRESH_RESULTS}  Fresh LLM: ${FRESH_LLM}"
echo "║  Started: $(date)"
echo "╚══════════════════════════════════════════════╝"
echo ""

for VENUE in "${VENUES[@]}"; do
  for YEAR in "${YEARS[@]}"; do
    CURRENT=$((CURRENT + 1))
    LOOP_START=$(date +%s)

    echo ""
    echo "┌──────────────────────────────────────────────"
    echo "│ [${CURRENT}/${TOTAL}] ${VENUE} ${YEAR}"
    echo "│ $(ts) Starting..."
    echo "└──────────────────────────────────────────────"

    echo "  $(ts) Step 1/2: fetch-coverage — finding OA PDF URLs..."
    if ! python -m wireless_taxonomy.cli fetch-coverage \
      --venue "$VENUE" --years "$YEAR" \
      --workers "$WORKERS" $WEB_SEARCH_FLAG \
      --json "cov_${VENUE}_${YEAR}.json"; then
      echo "  $(ts) ✗ fetch-coverage FAILED for ${VENUE} ${YEAR} — skipping extraction"
      FAILED+=("${VENUE}_${YEAR}")
      COMPLETED+=("${VENUE}_${YEAR}:FAILED")
      continue
    fi
    echo "  $(ts) Step 1/2: fetch-coverage done."

    echo ""
    echo "  $(ts) Step 2/2: extract-datasets — fetching PDFs, classifying, extracting..."
    echo "         (wireless-only filter → LLM classification → dataset extraction)"
    if ! python -m wireless_taxonomy.cli extract-datasets \
      --venue "$VENUE" --years "$YEAR" \
      --oa-json "cov_${VENUE}_${YEAR}.json" \
      --workers "$WORKERS" \
      --out ./src/results $EXTRACT_FRESH_FLAG; then
      echo "  $(ts) ✗ extract-datasets FAILED for ${VENUE} ${YEAR}"
      FAILED+=("${VENUE}_${YEAR}")
      COMPLETED+=("${VENUE}_${YEAR}:FAILED")
      continue
    fi
    echo "  $(ts) Step 2/2: extract-datasets done."

    LOOP_END=$(date +%s)
    LOOP_ELAPSED=$(( LOOP_END - LOOP_START ))
    TOTAL_ELAPSED=$(( LOOP_END - START_TIME ))
    REMAINING=$(( TOTAL - CURRENT ))

    if [ "$CURRENT" -gt 0 ]; then
      AVG_PER_LOOP=$(( TOTAL_ELAPSED / CURRENT ))
      ETA_SECS=$(( AVG_PER_LOOP * REMAINING ))
      ETA_STR="$(fmt_dur "$ETA_SECS")"
    else
      ETA_STR="?"
    fi

    COMPLETED+=("${VENUE}_${YEAR}")

    echo ""
    echo "  ✓ ${VENUE} ${YEAR} complete in $(fmt_dur "$LOOP_ELAPSED")"
    echo "  ─ Progress: ${CURRENT}/${TOTAL} done | ${REMAINING} remaining | ETA ~${ETA_STR}"
    echo "  ─ Completed so far: ${COMPLETED[*]}"
    echo ""
  done
done

echo ""
echo "$(ts) Merging all results into master CSVs..."
python -m wireless_taxonomy.cli merge-results --dir ./src/results --out ./src/results

echo ""
# Reconciliation is intentionally run separately for the hero run — it can be
# slow/expensive at corpus scale and is not needed after every incremental batch.
# Run it manually when the full corpus is collected:
#   wireless-taxonomy reconcile-datasets \
#     --csv ./src/results/master_datasets.csv \
#     --json ./src/results/master_raw.json \
#     --out ./src/results/reconcile_report.json
# To also use LLM confirmation add --llm-confirm (parallel, capped via
# --llm-workers N and --max-llm-pairs N; overflow pairs flagged for review).
echo "$(ts) Skipping reconciliation in batch mode (run separately for final corpus)."

echo ""
echo "$(ts) Generating corpus report..."
python -m wireless_taxonomy.cli report \
  --dir ./src/results --cov-dir . \
  --out ./src/results/master_report.md \
  || echo "$(ts) ✗ report generation failed (re-run manually)"

END_TIME=$(date +%s)
TOTAL_TIME=$(( END_TIME - START_TIME ))

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ALL ${TOTAL} LOOPS DONE                           ║"
echo "║  Total time: $(fmt_dur "$TOTAL_TIME")                  "
echo "║  Finished: $(date)"
echo "║  Results in: ./src/results/                  "
echo "║  Master files: master_papers.csv,            "
echo "║    master_datasets.csv, master_bibtex.csv    "
if [ ${#FAILED[@]} -gt 0 ]; then
echo "║                                              "
echo "║  FAILED (${#FAILED[@]}): ${FAILED[*]}"
fi
echo "╚══════════════════════════════════════════════╝"
