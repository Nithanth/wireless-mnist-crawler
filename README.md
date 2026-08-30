# wireless-taxonomy

Working repo for the **wireless-mnist** research project (CMU × NIST) on the
*openness and reuse of datasets in wireless research*.

The Python CLI (`wireless_taxonomy`) retrieves venue proceedings, enriches
paper metadata, classifies papers for wireless relevance, obtains legally
accessible full text, extracts structured dataset records, resolves repeated
dataset identities across papers, verifies availability evidence, and exports
a consolidated dataset taxonomy. It also evaluates the automated paper and
dataset results against a hand-curated validation set using precision, recall,
and Jaccard similarity.

Paper lists come primarily from **DBLP**, with metadata backfilled through
**OpenAlex, Crossref, Semantic Scholar, and arXiv**. Classification uses the
paper title and abstract and can use the full PDF when available. Dataset
extraction prefers full-paper text; strict abstract-only extractions are kept
separate from the PDF-backed catalog used for dataset-level analysis. Results,
PDF fetch state, and provenance are persisted in a corpus-specific SQLite
database, while resolved metadata and LLM responses are cached for repeatable
reruns.

---

## CLI usage

The primary workflow is `init` → `add` → `export`. Commands for individual
pipeline stages, validation, cache management, and debugging are available
under `wt advanced`. The complete LLM prompt inventory is documented in
[`PROMPTS.md`](PROMPTS.md).

### Setup

Requires Python ≥ 3.11. Runtime dependencies are `typer`, `click`, and `pypdf`.

```bash
pip install -e .
# optional: only needed to read a gold sheet saved as .xlsx (CSV needs nothing)
pip install -e ".[xlsx]"
```

Run via the installed entrypoint or the module directly (used below):

```bash
wt --help
PYTHONPATH=src python3 -m wireless_taxonomy.cli --help
# The legacy name `wireless-taxonomy` is also installed as an alias.
```

### Advanced: `classify` — loop a conference and label every paper

Pulls the accepted-paper list from DBLP (dropping poster/demo/workshop/keynote
records so only main-track papers remain), backfills missing DOIs + abstracts,
classifies each paper as wireless from title+abstract, and prints a
**yes/maybe/no breakdown** (counts + % of the conference set). No gold sheet
involved.

Abstracts are first **batch-fetched by DOI from Semantic Scholar** in one
request per conference (the single biggest coverage lever — see *Abstract
caching & providers* below), then any still-missing paper falls through a
per-paper provider chain tried in order: **USENIX page-scrape**
(NSDI/OSDI/ATC/Security) **→ OpenAlex → Crossref → Semantic Scholar → arXiv**
(title search, for preprints). An **ACM Digital Library** scrape is available as
an opt-in last resort for IMC/SIGCOMM/MobiCom — it's off by default because ACM
is Cloudflare-protected (see *Abstract caching & providers* below).

```bash
wt advanced classify \
  --venue NSDI --years 2023:2025 --llm \
  --csv nsdi.csv --json nsdi.json
```

`--years` takes a single year (`2024`) or an inclusive range (`2023:2025`); a
range prints a per-year breakdown plus an aggregate. Example output:

```
NSDI 2024 — 112 papers (abstracts: 112/112, 100%)
  yes     19  ( 17.0%)
  maybe    8  (  7.1%)
  no      85  ( 75.9%)
```

- `--no-llm` uses the keyword baseline (no API key needed); `--llm` (default)
  uses the configured provider.
- `--csv` / `--json` export the **full** labelled set — every paper with its
  `label`, `confidence`, and abstract flags, not just the wireless ones. This is
  exactly what `eval` consumes.
- `--source bibtex|csv|url --source-value <path-or-url>` swaps the paper-list
  source away from DBLP; `--no-resolve-dois` skips the programmatic DOI backfill.
- Resolved abstracts/DOIs **and LLM labels** are cached to `--cache-path`
  (default `.wt_cache.json`) so a re-run reads from disk instead of re-hitting
  the metadata APIs or the LLM — the cold run is network-bound, but a warm
  re-run is near-instant and deterministic. **Misses are cached too**, so the
  expensive no-hit papers aren't retried. Pass `--no-cache` to disable, or
  delete the cache file to force a full refresh (e.g. after enabling ACM).
- LLM labels are keyed by a hash of the exact prompt (title + abstract) and the
  model identity, so a re-run reuses each saved label **unless the title,
  abstract, or model changed**. Pass `--refresh-llm` to ignore cached labels and
  re-call the model (a fresh classification).

### Advanced: `eval` — DB-free snapshot scoring vs a gold sheet

Scoring is a pure, point-in-time computation, so it runs with **no DB and no
network** — straight from files. Give it the full labelled CSV from `classify
--csv` and a gold sheet; it matches **DOI → exact title → fuzzy title** per
(venue, year) and reports `jaccard / precision / recall / f1`.

```bash
wt advanced eval \
  --classified nsdi.csv --gold "List of Papers.csv" \
  --pass high --drop-workshops \
  --out report.json --md report.md
```

- `--pass high` scores `label == yes`; `--pass low` counts `yes|maybe`.
- `--drop-workshops` drops curated papers **absent from the classified
  universe** (co-located workshop papers not in the DBLP main proceedings) from
  the calculation, so they don't count as misses. This works purely from files
  because `classify --csv` writes the full proceedings universe. (Default is
  `--keep-workshops`.)
- Repeat `--classified` / `--gold` to union multiple files. Only conferences
  present in the classified CSV(s) are scored — unrun venue-years in the sheet
  are ignored, not penalised.
- `--exclude VENUE:YEAR` (repeatable) and `--min-gold N` pull thinly- or
  stale-curated venue-years out of the **overall** metrics and report them
  separately (with their would-be numbers), so a conference curated before its
  papers were released doesn't drag the headline. Example:
  `--exclude IMC:2025` or `--min-gold 3`.

### Abstract caching & providers

- **Cache.** `classify` keeps a JSON index (`--cache-path`, default
  `.wt_cache.json`) with three sections — resolved `abstracts` and `dois` (keyed
  by DOI and normalized title) and `llm` labels (keyed by prompt+model hash). It's
  read before any network/LLM call and written incrementally, so interrupted runs
  keep their progress and re-runs are fast and reproducible.
- **Semantic Scholar batch.** Before the per-paper loop, `classify` sends all
  DOIs for the conference to Semantic Scholar's batch endpoint in one request.
  This is what closes the ACM-venue gap: per-paper GETs get 429-throttled on a
  shared egress IP and silently drop most abstracts (IMC 2024 measured ~46%),
  whereas the single batched call recovers them all (IMC 2024 → 100%). Set
  `SEMANTIC_SCHOLAR_API_KEY` (a free key) to remove shared-IP throttling
  entirely; it's optional — the batch call works without one. Retryable
  responses honor the server's `Retry-After` header instead of failing.
- **arXiv.** Tried last in the abstract chain via a title search (guarded by a
  title match). Helpful for preprint-heavy systems papers; ACM measurement
  papers are rarely on arXiv, so yield there is low.
- **ACM (opt-in).** ACM paywalls full text *and* sits behind Cloudflare bot
  protection that blocks plain HTTP and headless browsers in most environments,
  so it's **off by default**. To attempt it: `pip install -e ".[acm]" &&
  playwright install chromium`, then set `WIRELESS_TAXONOMY_ACM_BROWSER=1`. It
  degrades to a no-op (never raises) when the challenge can't be cleared.

The same logic is importable:
`from wireless_taxonomy.eval.standalone import eval_files`.

### Advanced: `llm-config` — show configured LLM providers

```bash
wt advanced llm-config
```

### Experiment harness (`scripts/evaluate_coverage.py`)

Drives the two commands end to end across many conference-years and scores the
result against your curated sheet. Runnable from the repo root:

```bash
python scripts/evaluate_coverage.py \
  --gold "List of Papers.csv" \
  --classifier llm --drop-workshops \
  --db build/eval.sqlite --out-dir build/results
```

Drop in a sheet and the harness **auto-detects which conferences to evaluate**:
with no `--venue-year`, it derives the DBLP-ingestable venue-years from the
sheet(s) and loops over exactly those. Pass `--gold` more than once to union
several sheets, or `--venue-year SIGCOMM:2024` to pin an explicit set. For each
venue+year it runs `classify` (writing a full labelled CSV), then runs the
single `eval` over all those CSVs, writing `build/results/report.md` +
`report.json`. The CLI is the single source of truth; the script just
orchestrates it.

---

## Dataset Extraction Pipeline

The pipeline finds all papers from a conference year, classifies which are
wireless, fetches their PDFs, extracts structured dataset records via LLM, and
outputs CSV spreadsheets. Here's how to use it from scratch.

### Quick start

```bash
pip install -e .
cp .env.example .env
# Add at least one supported LLM API key to .env.

wt init wireless_v1
wt add --venues SIGCOMM,IMC,NSDI,ICC,TWC --years 2022:2025
wt export
wt status
```

Use `wt add --estimate` before a run to inspect expected paper and LLM-call
counts. Rerunning the same venue-years reuses cached metadata, PDFs, and LLM
responses unless `--fresh` is supplied.

### What happens under the hood

```text
DBLP proceedings and metadata APIs
                │
                ▼
Paper relevance classification (yes / maybe / no)
                │
                ▼
Open-access PDF discovery and full-text extraction
                │
                ▼
Structured dataset extraction from retained papers
                │
                ▼
Exact-name and URL matching → similarity candidates → LLM confirmation
                │
                ▼
Availability verification and consolidated CSV export
```

### Batch run (multiple venues × years)

The `run_batch.sh` script loops over all venue/year combos:

```bash
# Normal run — reuses all caches from previous runs
./run_batch.sh

# Full fresh run — archives old results, clears LLM cache, re-classifies everything
./run_batch.sh --fresh

# Other options:
./run_batch.sh --fresh-results   # archive old CSVs only, keep LLM cache
./run_batch.sh --fresh-llm       # clear LLM cache only, keep old CSVs as archive
```

Edit the VENUES and YEARS arrays at the top of the script to change what runs:

```bash
VENUES=("NSDI" "SIGCOMM" "IMC" "MobiCom")
YEARS=("2022" "2023" "2024" "2025")
```

The script shows live progress:

```text
┌──────────────────────────────────────────────
│ [3/16] IMC 2022
│ 14:32:01 Starting...
└──────────────────────────────────────────────
  [12/87] [+] yes(0.95): 5G Performance Measurement with mmWave...
  [13/87] [-] no(0.92): Scalable Zero-Knowledge Proofs for Non-Li...
  [14/87] [~] maybe(0.60): Mobile Edge Computing for Autonomous...
  ...
  Wireless filter: 23/87 papers pass low_pass (yes+maybe)
  [1/23] Extracting: 5G Performance Measurement with mmWave...
  ...
  ✓ IMC 2022 complete in 142s
  ─ Progress: 3/16 done | 13 remaining | ETA ~10min
```

If a venue/year fails (network drop, API error), it **skips** to the next one
instead of stopping. Failed loops are reported at the end.

### Merging results

After all runs, merge per-venue/year CSVs into master files:

```bash
# Automatically runs at the end of run_batch.sh, or run manually:
wt advanced merge-results --dir ./src/results --out ./src/results
```

Produces:

- `master_papers.csv` — all papers across all venues/years
- `master_datasets.csv` — deduplicated datasets with merged counts
- `master_bibtex.csv` — deduplicated BibTeX entries
- `master_raw.json` — all raw JSON combined

### Caching layers

| Layer | File | What it stores | How to clear |
|-------|------|---------------|--------------|
| **LLM cache** | `.wt_cache.json` | Classification labels, dataset extractions, abstracts, DOIs | Rerun `wt add` with `--fresh` |
| **PDF cache** | `taxonomy.sqlite` | Raw PDF bytes (expensive to re-download) | Almost never — prompt-independent |
| **Results** | `src/results/*.csv` | Output spreadsheets | `./run_batch.sh --fresh-results` (archives, doesn't delete) |

LLM classifications are **keyed by prompt + model hash**, so changing the
classification prompt automatically invalidates old cached results. You don't
need to manually clear the cache after editing the prompt.

Use `wt status` to inspect the active corpus and `wt advanced --help` for targeted cache-maintenance commands.

### Command reference

| Command | Purpose |
| --- | --- |
| `wt init` | Create or adopt a corpus workspace. |
| `wt add` | Retrieve papers, discover PDFs, classify wireless relevance, extract datasets, and merge venue-year results. |
| `wt export` | Reconcile dataset identities, fill unknown availability, and produce consolidated outputs. |
| `wt status` | Show corpus venues, years, paper counts, and extraction status. |
| `wt rollback` | Restore the snapshot preceding the most recent corpus run. |
| `wt advanced classify` | Run paper classification for a venue and year range. |
| `wt advanced eval` | Score classified papers against a curated validation file. |
| `wt advanced fetch-coverage` | Report legally fetchable open-access full text. |
| `wt advanced extract-datasets` | Run classification and structured dataset extraction as an individual stage. |
| `wt advanced merge-results` | Combine per-venue/year outputs into master files. |
| `wt advanced llm-config` | Show configured LLM providers and models. |

Run any command with `--help` for its full flags.

### Environment variables

Copy `.env.example` to `.env` and fill in:

| Variable | Required? | Purpose |
|----------|-----------|---------|
| `ANTHROPIC_API_KEY` | At least one LLM key | Anthropic Claude API key |
| `OPENAI_API_KEY` | At least one LLM key | OpenAI API key |
| `GEMINI_API_KEY` | At least one LLM key | Google Gemini API key |
| `WIRELESS_TAXONOMY_LLM_PROVIDER` | Yes | Primary LLM: `anthropic`, `openai`, or `google` |
| `WIRELESS_TAXONOMY_LLM_FALLBACKS` | No | Comma-separated fallback providers |
| `SEMANTIC_SCHOLAR_API_KEY` | Recommended | Better DOI resolution & abstract fetching |
| `WIRELESS_TAXONOMY_UNPAYWALL_EMAIL` | No | Email for Unpaywall OA lookups |

### Tests

```bash
PYTHONPATH=src python3 -m pytest -q
```
