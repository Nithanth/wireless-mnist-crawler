#!/usr/bin/env bash
set -uo pipefail

# Batch 1: NSDI, SIGCOMM, IMC, MobiCom × 2022–2025 (16 loops)
# Each loop: fetch-coverage → extract-datasets (wireless-only)
#
# Usage:
#   ./run_batch.sh                                    # defaults: NSDI,SIGCOMM,IMC,MobiCom × 2022-2025
#   ./run_batch.sh --venues "NSDI,IMC" --years "2024,2025"
#   ./run_batch.sh --venues "MobiCom" --years "2022:2025"
#   ./run_batch.sh --fresh                            # clear old results + LLM cache
#   ./run_batch.sh --fresh-results                    # archive old results only
#   ./run_batch.sh --fresh-llm                        # clear LLM cache only

# Prevent macOS from sleeping while this script runs (display + idle + disk + system).
# caffeinate is killed automatically when this script exits.
if [[ "$(uname)" == "Darwin" ]] && command -v caffeinate &>/dev/null; then
  caffeinate -dims -w $$ &
fi

# Activate the project venv (package is pip-installed in editable mode).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

# Defaults
VENUES_STR="NSDI,SIGCOMM,IMC,MobiCom"
YEARS_STR="2022,2023,2024,2025"
FRESH_RESULTS=false
FRESH_LLM=false
EXTRACT_FRESH_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venues)        VENUES_STR="$2"; shift 2 ;;
    --years)         YEARS_STR="$2"; shift 2 ;;
    --fresh)         FRESH_RESULTS=true; FRESH_LLM=true; shift ;;
    --fresh-results) FRESH_RESULTS=true; shift ;;
    --fresh-llm)     FRESH_LLM=true; shift ;;
    *)
      echo "Unknown flag: $1"
      echo "Usage: $0 [--venues \"NSDI,IMC\"] [--years \"2022:2025\"] [--fresh | --fresh-results | --fresh-llm]"
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
      ETA_MIN=$(( ETA_SECS / 60 ))
    else
      ETA_MIN="?"
    fi

    COMPLETED+=("${VENUE}_${YEAR}")

    echo ""
    echo "  ✓ ${VENUE} ${YEAR} complete in ${LOOP_ELAPSED}s"
    echo "  ─ Progress: ${CURRENT}/${TOTAL} done | ${REMAINING} remaining | ETA ~${ETA_MIN}min"
    echo "  ─ Completed so far: ${COMPLETED[*]}"
    echo ""
  done
done

echo ""
echo "$(ts) Merging all results into master CSVs..."
python -m wireless_taxonomy.cli merge-results --dir ./src/results --out ./src/results

END_TIME=$(date +%s)
TOTAL_TIME=$(( END_TIME - START_TIME ))
TOTAL_MIN=$(( TOTAL_TIME / 60 ))

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ALL ${TOTAL} LOOPS DONE                           ║"
echo "║  Total time: ${TOTAL_MIN} minutes                  "
echo "║  Finished: $(date)"
echo "║  Results in: ./src/results/                  "
echo "║  Master files: master_papers.csv,            "
echo "║    master_datasets.csv, master_bibtex.csv    "
if [ ${#FAILED[@]} -gt 0 ]; then
echo "║                                              "
echo "║  FAILED (${#FAILED[@]}): ${FAILED[*]}"
fi
echo "╚══════════════════════════════════════════════╝"
