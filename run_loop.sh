#!/usr/bin/env bash
set -uo pipefail

# run_loop.sh — the full extraction loop for a set of conferences and years.
#
# Pipeline per venue/year:  fetch-coverage → extract-datasets (wireless-only)
# Then once at the end:     merge-results → report
#
# Usage:
#   ./run_loop.sh --venues "SIGCOMM,IMC,NSDI" --years 2024
#   ./run_loop.sh --venues "NSDI,IMC" --years "2024,2025"       # comma list
#   ./run_loop.sh --venues "MobiCom" --years "2022:2025"        # inclusive range
#
# Inputs:
#   --venues "A,B,C"    Conference names, comma-separated (default: SIGCOMM,IMC,NSDI)
#   --years  "Y[,Y|:Y]" Years: single, comma list, or START:END range (default: 2022,2023,2024)
#   --corpus NAME       Corpus to run inside (corpora/<NAME>/). Without this flag
#                       the active corpus is reused, or corpus_v1 is auto-created
#                       (legacy repo-root layout used if corpora/ doesn't exist).
#   --workers N         Thread parallelism for PDF fetch + LLM calls (default: 6)
#   --web-search        Enable Brave/Google-CSE PDF discovery fallback
#   --verbose           Per-paper classification output
#   --fresh             Archive old results AND clear LLM cache (full redo)
#   --fresh-results     Archive old result CSVs only
#   --fresh-llm         Clear LLM cache only (papers re-classified + re-extracted)
#   --retry-failed      Retry PDF downloads that previously failed
#
# The corpus is INCREMENTAL: run once for your initial set, then run again with
# new --venues/--years to grow it. Already-resolved papers are never re-fetched
# or re-extracted (content-addressed cache); only new work costs anything.
#
# Model safety: if the corpus was built with a different LLM than currently
# configured, the run stops with a warning before any work happens.
#
# A DB snapshot is taken before each run (corpora/<name>/snapshots/) —
# roll back anytime with:  wireless-taxonomy corpus rollback <stamp>
#
# Logs: every run is tee'd to logs/YYYYMMDD_HHMMSS.log  (stdout + stderr).

# Prevent macOS from sleeping while this script runs (display + idle + disk + system).
# caffeinate is killed automatically when this script exits.
if [[ "$(uname)" == "Darwin" ]] && command -v caffeinate &>/dev/null; then
  caffeinate -dims -w $$ &
fi

# Activate the project venv (package is pip-installed in editable mode).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

# ── Logging setup ─────────────────────────────────────────────────────────────
# Tee everything (stdout + stderr) to a timestamped log file so you can inspect
# a completed or failed run without relying on scrollback.
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date '+%Y%m%d_%H%M%S').log"
# Redirect both stdout and stderr through tee into the log file, preserving
# terminal output in real time.
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"
echo ""

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
check("Unpaywall email (any alias)",
      os.getenv("WIRELESS_TAXONOMY_CONTACT_EMAIL")
      or os.getenv("WIRELESS_TAXONOMY_UNPAYWALL_EMAIL")
      or os.getenv("UNPAYWALL_EMAIL"),
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
RETRY_FAILED_FLAG=""
WORKERS=6
WEB_SEARCH_FLAG=""
VERBOSE_FLAG=""
CORPUS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venues)        VENUES_STR="$2"; shift 2 ;;
    --years)         YEARS_STR="$2"; shift 2 ;;
    --workers)       WORKERS="$2"; shift 2 ;;
    --corpus)        CORPUS="$2"; shift 2 ;;
    --web-search)    WEB_SEARCH_FLAG="--web-search"; shift ;;
    --verbose)       VERBOSE_FLAG="--verbose"; shift ;;
    --fresh)         FRESH_RESULTS=true; FRESH_LLM=true; shift ;;
    --fresh-results) FRESH_RESULTS=true; shift ;;
    --fresh-llm)     FRESH_LLM=true; shift ;;
    --retry-failed)  RETRY_FAILED_FLAG="--retry-failed"; shift ;;
    *)
      echo "Unknown flag: $1"
      echo "Usage: $0 [--venues \"NSDI,IMC\"] [--years \"2022:2025\"] [--corpus NAME] [--workers N] [--web-search] [--fresh | --fresh-results | --fresh-llm] [--retry-failed]"
      exit 1 ;;
  esac
done

# ── Corpus resolution ────────────────────────────────────────────────────────
# If corpora/ exists (or --corpus was given), operate inside a versioned
# corpus directory: corpora/<name>/{taxonomy.sqlite, results/, snapshots/}.
# Without --corpus the active corpus is reused (or corpus_v1 auto-created).
# Legacy mode (no corpora/ dir, no --corpus flag): repo-root layout unchanged.
DB_PATH="taxonomy.sqlite"
RESULTS_DIR="./src/results"
CORPUS_ARGS=()
if [[ -n "$CORPUS" || -d "corpora" ]]; then
  CORPUS_INFO=$(python - "$CORPUS" <<'CORPUS_RESOLVE'
import sys
from wireless_taxonomy.corpus import check_model_compatibility, resolve_corpus

name = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
c = resolve_corpus(name, create=True)
snap = c.snapshot()

# Model compatibility check — happens ONCE here at batch start.
current_model = ""
try:
    from wireless_taxonomy.config import load_settings
    from wireless_taxonomy.llm import LlmRouter
    p = LlmRouter(load_settings(str(c.db_path)).llm).select_provider()
    current_model = f"{p.provider}/{p.model}"
except Exception:
    pass
warning = check_model_compatibility(c, current_model) or "-"

print(c.name)
print(c.db_path)
print(c.results_dir)
print(snap.name if snap else "-")
print(warning)
CORPUS_RESOLVE
)
  if [ $? -ne 0 ] || [ -z "$CORPUS_INFO" ]; then
    echo "Corpus resolution failed (invalid name?). Aborting."
    exit 1
  fi
  CORPUS_NAME=$(echo "$CORPUS_INFO" | sed -n 1p)
  DB_PATH=$(echo "$CORPUS_INFO" | sed -n 2p)
  RESULTS_DIR=$(echo "$CORPUS_INFO" | sed -n 3p)
  SNAPSHOT=$(echo "$CORPUS_INFO" | sed -n 4p)
  MODEL_WARNING=$(echo "$CORPUS_INFO" | sed -n '5,$p')
  # Corpus names are validated (no spaces/slashes) so array expansion is safe.
  CORPUS_ARGS=(--corpus "$CORPUS_NAME")
  echo "Corpus: $CORPUS_NAME  (db: $DB_PATH)"
  if [[ "$SNAPSHOT" != "-" ]]; then
    echo "Snapshot taken: $SNAPSHOT  (rollback with: corpus rollback <stamp>)"
  fi
  if [[ "$MODEL_WARNING" != "-" ]]; then
    echo ""
    echo "⚠️  $MODEL_WARNING"
    if [[ ! -t 0 ]]; then
      # Non-interactive (CI / piped): fail safe, never mix models silently.
      echo "Non-interactive session — aborting. Start a new corpus with:"
      echo "  ./run_loop.sh --corpus <new-name> ..."
      exit 1
    fi
    read -r -p "Continue with mixed-model corpus? [y/N] " REPLY
    if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
      echo "Aborted. Start a new corpus with: ./run_loop.sh --corpus <new-name> ..."
      exit 1
    fi
  fi
  echo ""
fi

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

    COV_JSON="cov_${VENUE}_${YEAR}.json"
    # fetch-coverage always runs, but it is per-paper incremental:
    #   - papers with a cached PDF URL resolve instantly from the cache
    #   - papers with a cached negative verdict newer than the per-paper TTL
    #     (WIRELESS_TAXONOMY_OA_NEGATIVE_TTL_DAYS, default 14d) also skip
    #   - only papers with STALE negative verdicts hit the network/API again
    # So a fully cached venue takes seconds, and each run monotonically
    # improves coverage as stale negatives get retried. The JSON is rewritten
    # each run with the union of everything known.
    echo "  $(ts) Step 1/2: fetch-coverage — resolving OA PDF URLs (cached papers skip instantly)..."
    if ! python -m wireless_taxonomy.cli fetch-coverage \
      --venue "$VENUE" --years "$YEAR" \
      --workers "$WORKERS" $WEB_SEARCH_FLAG \
      --json "$COV_JSON"; then
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
      --oa-json "$COV_JSON" \
      --workers "$WORKERS" \
      --out "$RESULTS_DIR" --db "$DB_PATH" --yes ${CORPUS_ARGS[@]+"${CORPUS_ARGS[@]}"} $EXTRACT_FRESH_FLAG $RETRY_FAILED_FLAG $VERBOSE_FLAG; then
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
python -m wireless_taxonomy.cli merge-results --dir "$RESULTS_DIR" --out "$RESULTS_DIR"

echo ""
# Reconciliation is intentionally run separately for the hero run — it can be
# slow/expensive at corpus scale and is not needed after every incremental batch.
# Run it manually when the full corpus is collected:
#   wt reconcile-datasets \
#     --csv ./src/results/master_datasets.csv \
#     --json ./src/results/master_raw.json \
#     --out ./src/results/reconcile_report.json
# To also use LLM confirmation add --llm-confirm (parallel, capped via
# --llm-workers N and --max-llm-pairs N; overflow pairs flagged for review).
echo "$(ts) Skipping reconciliation in batch mode (run separately for final corpus)."

echo ""
echo "$(ts) Generating corpus report..."
python -m wireless_taxonomy.cli report \
  --dir "$RESULTS_DIR" --cov-dir . \
  --out "$RESULTS_DIR/master_report.md" \
  || echo "$(ts) ✗ report generation failed (re-run manually)"

END_TIME=$(date +%s)
TOTAL_TIME=$(( END_TIME - START_TIME ))

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ALL ${TOTAL} LOOPS DONE                           ║"
echo "║  Total time: $(fmt_dur "$TOTAL_TIME")                  "
echo "║  Finished: $(date)"
echo "║  Results in: $RESULTS_DIR/                  "
echo "║  Master files: master_papers.csv,            "
echo "║    master_datasets.csv, master_bibtex.csv    "
if [ ${#FAILED[@]} -gt 0 ]; then
echo "║                                              "
echo "║  FAILED (${#FAILED[@]}): ${FAILED[*]}"
fi
echo "╚══════════════════════════════════════════════╝"
