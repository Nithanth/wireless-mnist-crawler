# wireless-taxonomy

Working repo for the **wireless-mnist** research project (CMU × NIST) on the
*openness of wireless datasets used to reproduce ML research*.

This is a focused **coverage-evaluation tool**: a Python CLI
(`wireless_taxonomy`) that, per conference/year, pulls the accepted-paper list,
backfills titles/abstracts from open metadata APIs, classifies which papers are
wireless (keyword or LLM, from title + abstract), and **scores that automated
set against a hand-curated gold list** (Jaccard / IoU, precision / recall / F1)
so we can quantify how well the automated path reproduces the manual curation.

Because ACM/IEEE block automated full-text fetching, the workflow is
deliberately metadata-only: it works from **DBLP** (authoritative paper list:
title/authors/DOI) plus **OpenAlex/Crossref/Semantic Scholar/arXiv** (abstracts,
with a USENIX page-scrape fallback and an opt-in ACM scrape), and compares on
title + abstract. Resolved abstracts/DOIs are cached to disk so re-runs are
fast and deterministic, and DBLP poster/demo/workshop records are dropped at
ingest so they don't pollute the proceedings set. Data is persisted in SQLite.

---

## CLI usage

The CLI is **three commands**: `classify` (the whole per-conference loop, with a
pretty yes/maybe/no breakdown), `eval` (DB-free snapshot scoring against a gold
sheet), and `llm-config` (which LLM providers are configured).

### Setup

Requires Python ≥ 3.11. The tool is intentionally light — its only runtime
dependencies are `typer`/`click`.

```bash
pip install -e .
# optional: only needed to read a gold sheet saved as .xlsx (CSV needs nothing)
pip install -e ".[xlsx]"
```

Run via the installed entrypoint or the module directly (used below):

```bash
wireless-taxonomy --help
PYTHONPATH=src python3 -m wireless_taxonomy.cli --help
```

### 1. `classify` — loop a conference and label every paper

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
PYTHONPATH=src python3 -m wireless_taxonomy.cli classify \
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

### 2. `eval` — DB-free snapshot scoring vs a gold sheet

Scoring is a pure, point-in-time computation, so it runs with **no DB and no
network** — straight from files. Give it the full labelled CSV from `classify
--csv` and a gold sheet; it matches **DOI → exact title → fuzzy title** per
(venue, year) and reports `jaccard / precision / recall / f1`.

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli eval \
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

### 3. `llm-config` — show configured LLM providers

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli llm-config
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

### Command reference

| Command | Purpose |
| --- | --- |
| `classify` | Loop a venue over a year (or `--years A:B` range): DBLP ingest → DOI/abstract backfill → classify; prints the yes/maybe/no breakdown and exports the full labelled set. |
| `eval` | DB-free snapshot eval: score a classified CSV vs a gold sheet (DOI→title→fuzzy), with optional `--drop-workshops`. No DB/network. |
| `llm-config` | Show which LLM providers are configured. |

Run any command with `--help` for its full flags.

### Tests

```bash
PYTHONPATH=src python3 -m pytest -q
```
