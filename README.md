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
title/authors/DOI) plus **OpenAlex/Crossref/Semantic Scholar** (abstracts), and
compares on title + abstract. Data is persisted in SQLite.

---

## CLI usage

### Setup

Requires Python ≥ 3.11. The tool is intentionally light — its only runtime
dependencies are `typer`/`click`.

```bash
pip install -e .
# optional: only needed to import a gold sheet saved as .xlsx (CSV needs nothing)
pip install -e ".[xlsx]"
```

Every command runs through the Typer app. Use the installed entrypoint:

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

### Coverage evaluation (main workflow)

Goal: measure how well automated wireless detection matches a manually curated
list for a conference. Pull the list from DBLP, backfill abstracts, classify,
import the gold sheet, then score.

```bash
# 1. Paper list from DBLP (title/authors/DOI; no abstracts)
curl "https://dblp.org/search/publ/api?q=toc:db/conf/sigcomm/sigcomm2024.bht:&h=1000&format=bib1" -o sigcomm2024.bib
PYTHONPATH=src python3 -m wireless_taxonomy.cli ingest \
  --venue SIGCOMM --year 2024 --bibtex sigcomm2024.bib --db taxonomy.sqlite
# -> Ingest completed. run_id=1

# 2. Backfill abstracts from OpenAlex/Crossref/Semantic Scholar (by DOI, then title)
PYTHONPATH=src python3 -m wireless_taxonomy.cli enrich-abstracts --run-id 1 --db taxonomy.sqlite

# 3. Classify wireless candidates (title+abstract). Keyword needs no API key;
#    add --llm to use the configured LLM (see llm-config).
PYTHONPATH=src python3 -m wireless_taxonomy.cli classify-candidates --run-id 1 --db taxonomy.sqlite

# 4. Import the curated gold sheet (once; loads every venue/year it contains)
PYTHONPATH=src python3 -m wireless_taxonomy.cli import-gold \
  --path "List of Papers.csv" --db taxonomy.sqlite

# 5. Score the automated set vs the gold set
PYTHONPATH=src python3 -m wireless_taxonomy.cli eval-overlap --classifier keyword --db taxonomy.sqlite
```

`eval-overlap` reports per-conference-year, per-venue, and overall
`jaccard / precision / recall / f1` (`--pass low` counts `yes|maybe`;
`--classifier llm` scores the LLM labels). Matching is **DOI → exact title →
fuzzy title**. Two reporting flags:

- `--drop-workshops` — drop curated papers absent from the ingested main
  proceedings (co-located workshop papers) from the calculation, so they don't
  count as misses. They're still listed separately under "dropped workshop
  papers". Default is `--keep-workshops`.
- `--md results.md` — write a readable Markdown report (overall table, per
  conference-year + per-venue tables, and per-conference discrepancy lists:
  false positives, classifier misses, dropped/missing papers). Pair with
  `--out results.json` for the machine-readable version.

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli eval-overlap \
  --classifier llm --pass high --drop-workshops \
  --out results.json --md results.md --db taxonomy.sqlite
```

### Export the fetched paper set (`paper-set`)

Export the conference-scoped set of fetched papers (e.g. to inspect or diff):

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli paper-set \
  --run-id 1 --out papers.csv --format csv --db taxonomy.sqlite
```

Columns: `match_key, title, abstract, authors, doi, year, venue, wireless_label, wireless_confidence`.
`--wireless-only` filters to papers classified wireless (run `classify-wireless`
first for the keyword label).

### Compare two automated sources (`diff-sets`)

To gauge how reliable a source is, export a `paper-set` from each approach and
diff them — e.g. a URL+LLM ingest vs the DBLP+OpenAlex ingest of the same
conference. Export each to its own file, then:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli diff-sets \
  --a url_llm.csv --b dblp_openalex.csv \
  --label-a "URL+LLM" --label-b "DBLP+OpenAlex" \
  --reference b --out diff.json --csv diff.csv
```

It prints the **Jaccard (IoU)** of the two sets, the papers unique to each side,
and **abstract coverage per side**. Matching is **DOI-first** → exact title →
fuzzy (author-boosted); `--exact` disables fuzzy. Pass **`--reference a|b`** to
treat that side as ground truth and also report **precision / recall / F1** —
e.g. with `--reference b` and B=DBLP, recall = fraction of real papers the other
source caught, precision = fraction of its papers that are real. No database
needed — it reads the files.

### Inspecting runs

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli status --db taxonomy.sqlite      # run history
PYTHONPATH=src python3 -m wireless_taxonomy.cli llm-config --db taxonomy.sqlite  # configured LLM providers
```

### Command reference

| Command | Purpose |
| --- | --- |
| `init` | Create/upgrade the SQLite database. |
| `ingest` | Load a paper list from `--url`, `--bibtex`, or `--csv`. |
| `enrich-abstracts` | Backfill abstracts from OpenAlex/Crossref/Semantic Scholar. |
| `classify-wireless` | Keyword wireless classification (title + abstract). |
| `classify-candidates` | Wireless-candidate labels (yes/no/maybe) for gold eval (`--llm` optional). |
| `import-gold` | Import a manual gold sheet (csv/xlsx) of wireless papers. |
| `eval-overlap` | Precision/recall/F1/Jaccard of the automated set vs the gold set. |
| `paper-set` | Export the conference-scoped fetched paper set. |
| `diff-sets` | Diff two paper-set exports (IoU + abstract coverage) to compare sources. |
| `status` / `llm-config` | Inspect run history and configured LLM providers. |

Run any command with `--help` for its full flags.

### Tests

```bash
PYTHONPATH=src python3 -m pytest -q
```
