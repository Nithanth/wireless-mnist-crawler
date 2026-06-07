# wireless-taxonomy

Working repo for the **wireless-mnist** research project (CMU × NIST) on the
*openness of wireless datasets used to reproduce ML research*.

The study manually curates, per conference/year, which wireless papers were
published, which datasets they use, whether those datasets are open, and their
modalities. This repo is the tooling that supports that effort: a Python CLI
(`wireless_taxonomy`) that pulls a conference's accepted-paper list, backfills
titles/abstracts from open metadata APIs, classifies which papers are wireless,
and **scores that automated set against the hand-curated list** (Jaccard / IoU,
precision / recall) so we can quantify how well the automated path reproduces
the manual curation.

Because ACM/IEEE block automated full-text fetching, the current workflow is
deliberately metadata-only: it works from **DBLP** (authoritative paper list:
title/authors/DOI) plus **OpenAlex/Crossref/Semantic Scholar** (abstracts), and
compares on title + abstract. Data is persisted in SQLite; the package also
contains a longer LLM-backed taxonomy pipeline (analysis, dataset extraction,
export) that the evaluation work is layered on top of.

---

## CLI usage

### Setup

Requires Python ≥ 3.11.

```bash
pip install -e .                      # installs typer, pandas, openpyxl, ...
# optional: authenticated ACM browser fallback
pip install -e ".[browser]"
```

Every command is run through the Typer app. Either use the installed entrypoint:

```bash
wireless-taxonomy --help
```

or run the module directly (used in the examples below):

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli --help
```

All commands take `--db` (default `taxonomy.sqlite`). Initialize a database:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli init --db taxonomy.sqlite
```

### Coverage evaluation (current focus)

Goal: measure how well the automated wireless detection matches your manually
curated list for a conference. Pull the list from DBLP, backfill abstracts,
classify, then score.

```bash
# 1. Paper list from DBLP (title/authors/DOI; no abstracts)
curl "https://dblp.org/search/publ/api?q=toc:db/conf/sigcomm/sigcomm2024.bht:&h=1000&format=bib1" -o sigcomm2024.bib
PYTHONPATH=src python3 -m wireless_taxonomy.cli ingest \
  --venue SIGCOMM --year 2024 --bibtex sigcomm2024.bib --db taxonomy.sqlite
# -> Ingest completed. run_id=1

# 2. Backfill abstracts from OpenAlex/Crossref/Semantic Scholar (by DOI, then title)
PYTHONPATH=src python3 -m wireless_taxonomy.cli enrich-abstracts --run-id 1 --db taxonomy.sqlite

# 3. Classify which papers are wireless (keyword, title+abstract; no API key)
PYTHONPATH=src python3 -m wireless_taxonomy.cli classify-wireless --run-id 1 --db taxonomy.sqlite

# 4. Score the automated wireless set vs your curated CSV
PYTHONPATH=src python3 -m wireless_taxonomy.cli jaccard \
  --run-id 1 --manual "List of Papers.csv" \
  --csv comparison.csv --out report.json --db taxonomy.sqlite
```

`jaccard` prints a readable summary and self-filters the manual CSV to the run's
conference + year:

```text
SIGCOMM 2024  —  Jaccard (IoU) = 0.0980
  matched (intersection) :    5   (fuzzy: 1)
  union                  :   51
  automated / manual     :   47 / 9
  missed_by_cli          :    4  (curated wireless the CLI didn't flag)
  extra_from_cli         :   42  (CLI-flagged, not in your sheet)
  mean wireless confidence (automated): 0.91
```

- `--csv` writes one row per paper with a `status` column
  (`matched` / `fuzzy_matched` / `missed_by_cli` / `extra_from_cli`), plus
  `title_similarity`, `author_overlap`, `shared_authors`, and
  `wireless_label` / `wireless_confidence`.
- Manual-CSV columns (title / authors / conference / year) are auto-detected;
  override with `--title-col` / `--authors-col` / `--conference-col` / `--year-col`.
- Matching is exact-normalized-title, then fuzzy (difflib + author-surname boost);
  `--exact` disables fuzzy.
- `--all-papers` compares the full ingested list instead of the wireless subset;
  `--no-conference-filter` disables the conference/year filter.

**Across every conference in the DB** (repeat steps 1–2 per venue/year first):

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli jaccard-all \
  --manual "List of Papers.csv" --csv comparison_all.csv --db taxonomy.sqlite
```

```text
NSDI 2024:    index=0.8000 intersection=16 union=20 ...
SIGCOMM 2024: index=0.0980 intersection=5  union=51 ...
Aggregate coverage. conferences=2 skipped=0 micro=... macro=...
```

`jaccard-all` rolls up **micro** (pooled papers) and **macro** (mean of
per-conference indices). With `--wireless-only` it auto-runs `classify-wireless`
for any unclassified conference (`--no-auto-classify` to opt out).

**Export just the fetched paper set** (e.g. to inspect or diff manually):

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli paper-set \
  --run-id 1 --out papers.csv --format csv --db taxonomy.sqlite
```
Columns: `match_key, title, abstract, authors, doi, year, venue, wireless_label, wireless_confidence`.

#### Compare two automated sources (`diff-sets`)

To gauge how reliable a source is, export a `paper-set` from each approach and diff
them — e.g. a URL+LLM ingest vs the DBLP+OpenAlex ingest of the same conference.
Export each to its own file (papers accumulate per conference in one DB, so use
separate DBs or runs), then:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli diff-sets \
  --a url_llm.csv --b dblp_openalex.csv \
  --label-a "URL+LLM" --label-b "DBLP+OpenAlex" \
  --out diff.json --csv diff.csv
```

It prints the **Jaccard (IoU)** of the two sets, the papers unique to each side,
and **abstract coverage per side** (how many abstracts each source actually
supplies). `--csv` writes one row per paper with `status`
(`shared`/`only_in_a`/`only_in_b`), match type, title similarity, and
abstract-present flags. Matching is the same exact→fuzzy (author-boosted) logic as
`jaccard`; `--exact` disables fuzzy. No database needed — it reads the files.

#### Gold-set evaluation (precision / recall / F1)

An alternative scoring track using an imported gold sheet and candidate labels:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli import-gold \
  --path "List of Papers.csv" --db taxonomy.sqlite
PYTHONPATH=src python3 -m wireless_taxonomy.cli classify-candidates --run-id 1 --db taxonomy.sqlite
PYTHONPATH=src python3 -m wireless_taxonomy.cli eval-overlap --classifier keyword --db taxonomy.sqlite
```

`eval-overlap` reports per-conference and overall `jaccard / precision / recall /
f1` (`--pass low` counts `yes|maybe`; `--classifier llm` scores the LLM labels).

### Full taxonomy pipeline

The original end-to-end pipeline (text enrichment → analysis → dataset
extraction → export) is still available. Run it in one shot:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli run \
  --venue SIGCOMM --year 2025 \
  --url https://conferences.sigcomm.org/sigcomm/2025/program/papers-info/ \
  --out workbook.xlsx --format xlsx --db taxonomy.sqlite
```

Add `--llm` to use the configured LLM for paper analysis (see `llm-config`).
The same stages can be run individually: `ingest`, `verify-paper-list`,
`assess-scope`, `enrich-paper-text`, `discover-full-text`, `add-pdfs`,
`fetch-acm-browser`, `assess-paper-inputs`, `agentic-paper-analysis`,
`reflect-paper-analysis`, `extract-datasets`, `check-availability`,
`resolve-reuse`, `export`.

### Inspecting runs

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli status --db taxonomy.sqlite      # run history
PYTHONPATH=src python3 -m wireless_taxonomy.cli review --db taxonomy.sqlite      # items flagged for review
PYTHONPATH=src python3 -m wireless_taxonomy.cli llm-config --db taxonomy.sqlite  # configured LLM providers
```

### Command reference

| Command | Purpose |
| --- | --- |
| `init` | Create/upgrade the SQLite database. |
| `ingest` | Load a paper list from `--url`, `--bibtex`, or `--csv`. |
| `run` | Full pipeline ingest → analysis → export. |
| `enrich-abstracts` | Backfill abstracts from OpenAlex/Crossref/Semantic Scholar. |
| `classify-wireless` | Keyword wireless classification (title + abstract). |
| `classify-candidates` | Wireless-candidate labels (yes/no/maybe) for gold eval. |
| `paper-set` | Export the conference-scoped fetched paper set. |
| `diff-sets` | Diff two paper-set exports (IoU + abstract coverage) to compare sources. |
| `jaccard` | IoU of automated vs manual list for one run. |
| `jaccard-all` | IoU across every conference instance, with micro/macro roll-ups. |
| `import-gold` | Import a manual gold sheet of wireless papers. |
| `eval-overlap` | Precision/recall/F1/Jaccard vs the gold set. |
| `verify-paper-list` | Quality-check an ingested list. |
| `assess-scope` | Check a source is networking/wireless-relevant. |
| `export` | Write the taxonomy workbook (csv/xlsx/json). |
| `status` / `review` / `llm-config` | Inspect runs, review queue, LLM config. |

Run any command with `--help` for its full flags.

### Tests

```bash
PYTHONPATH=src python3 -m pytest -q
```
