# wireless-taxonomy

`wireless-taxonomy` is a Python CLI for building a wireless research taxonomy from a conference or proceedings source. The target workflow is:

1. Point the CLI at a page, BibTeX file, or CSV containing accepted papers.
2. Extract and verify the paper list.
3. Gather paper text, PDF text, links, and evidence snippets.
4. Run taxonomy analysis over the papers.
5. Export a workbook or CSV/JSON artifact shaped like the manual taxonomy spreadsheet.

The intended final workbook mirrors the manual Google Sheets taxonomy:

- `List of Papers`
- `List of Datasets`
- `Bibtex`
- `Review Needed`
- `Evidence`
- `Paper Dataset Links`

The project is built around a simple principle: use deterministic code for database writes, thresholds, review gating, reuse counts, and exports; use LLM/agentic behavior for messy extraction, paper understanding, dataset synthesis, and evidence gathering.

## Current Status

The project currently has a working sequential pipeline with SQLite persistence, JSONL evidence logging, CLI commands, regression tests, and multiple full-text retrieval strategies.

The tested pipeline can currently:

- ingest paper lists from URL, BibTeX, or CSV
- parse known/simple proceedings pages deterministically
- use an LLM fallback for heterogeneous URL extraction
- verify paper-list quality
- assess whether a source appears relevant to networking/wireless research
- enrich papers with abstracts, links, landing-page text, and snippets
- discover full text through open resolvers
- ingest local PDFs as a fallback
- optionally use an authenticated ACM browser fallback when the user has legitimate access
- assess whether each paper has enough input text for taxonomy analysis
- run deterministic or LLM-backed paper analysis
- extract dataset claims, modalities, and OSI layers
- run a deterministic reflection pass over analysis outputs
- check dataset availability
- resolve dataset identity/reuse
- export CSV, XLSX, and JSON

The regression suite is currently green:

```bash
PYTHONPATH=src python3 -m pytest -q
```

Current result after the latest cleanup:

```text
43 passed
```

## Architecture

The package lives under:

```text
src/wireless_taxonomy
```

Main areas:

```text
wireless_taxonomy/
  analyze/        Paper text enrichment, full-text discovery, analysis, reflection
  db.py           SQLite connection, migrations, transactions
  evidence.py     JSONL evidence/event logging
  export/         Spreadsheet and JSON export
  ingest/         URL, BibTeX, CSV ingestion and verification
  resolve/        Dataset identity, reuse, resolver cache
  review/         Review queue helpers
  cli.py          Typer CLI entrypoint
  pipeline.py     Sequential pipeline orchestration
```

The main public entrypoint is:

```text
wireless_taxonomy.cli:app
```

Run commands locally with:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli --help
```

## Data Model

The pipeline writes to SQLite. Migrations live in:

```text
migrations/
```

Core persisted records include:

- conference instances
- pipeline runs
- papers
- paper sources
- paper text artifacts
- paper text links
- paper text snippets
- resolver cache entries
- paper input readiness reports
- paper agentic analyses
- paper analysis dataset claims
- paper analysis reflections
- datasets
- paper-dataset links
- evidence claims
- review items

Evidence is persisted in two places:

- canonical structured rows in SQLite
- JSONL event logs under the configured evidence directory

By default, evidence logs are stored near the selected database unless `WIRELESS_TAXONOMY_EVIDENCE_DIR` is set.

## Pipeline Stages

The current end-to-end pipeline is sequential. Each stage creates a new `pipeline_runs` row and writes its own artifacts/evidence.

### 1. Ingest

Command:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli ingest \
  --venue SIGCOMM \
  --year 2025 \
  --url https://conferences.sigcomm.org/sigcomm/2025/program/papers-info/ \
  --db taxonomy.sqlite
```

Supported inputs:

- `--url`
- `--bibtex`
- `--csv`

URL ingestion fetches and cleans HTML while preserving links. Known/simple proceedings pages can be parsed deterministically. Heterogeneous pages can use the configured LLM fallback.

### 2. Verify Paper List

Command:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli verify-paper-list \
  --run-id 1 \
  --external \
  --llm \
  --db taxonomy.sqlite
```

Verification checks:

- missing title/authors/abstract/DOI
- duplicate titles
- malformed records
- low source confidence
- optional Crossref checks
- optional LLM verifier pass

Verification confidence is a source-level quality score for whether the extracted paper list looks complete and coherent enough to continue.

### 3. Assess Scope

Command:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli assess-scope \
  --run-id 1 \
  --db taxonomy.sqlite
```

This stage checks whether the source looks like a networking/wireless-relevant research source. It flags suspicious inputs such as random unrelated conference pages, malformed lists, or sources where most papers do not appear related to networking or wireless.

The `run` command can prompt before continuing when scope looks questionable. Use `--yes` to proceed without prompting.

### 4. Enrich Paper Text

Command:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli enrich-paper-text \
  --run-id 1 \
  --db taxonomy.sqlite
```

This gathers lower-cost paper context:

- abstract text
- paper landing page links
- source page text where available
- snippets around dataset/data/artifact terms

This stage is useful even when full PDFs are not yet available.

### 5. Discover Full Text

Command:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli discover-full-text \
  --run-id 1 \
  --db taxonomy.sqlite
```

For one paper:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli discover-full-text \
  --run-id 1 \
  --paper-id 42 \
  --db taxonomy.sqlite
```

Full-text discovery tries open and programmatic sources before any manual fallback:

1. paper/proceedings links already found during ingestion
2. OpenAlex
3. Crossref
4. Semantic Scholar
5. Unpaywall
6. arXiv title search
7. OpenReview title search
8. bounded web search
9. publisher DOI landing/PDF candidates

Semantic Scholar title/author matching is used because DOI-only matching can miss alternate open versions. DOI is still preferred when it resolves cleanly.

Rate limits are enforced for Semantic Scholar and OpenReview.

### 6. Add Local PDFs

Command:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli add-pdfs \
  --run-id 1 \
  --dir ./papers \
  --db taxonomy.sqlite
```

This is the fallback when PDFs cannot be retrieved programmatically. The importer matches local PDFs back to the paper list and extracts text/snippets from them.

### 7. Authenticated ACM Browser Fallback

Command for login:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli fetch-acm-browser \
  --run-id 1 \
  --login \
  --profile-dir .browser/acm \
  --db taxonomy.sqlite
```

Command for fetching:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli fetch-acm-browser \
  --run-id 1 \
  --profile-dir .browser/acm \
  --limit 10 \
  --db taxonomy.sqlite
```

With a manually launched Chrome/CDP session:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli fetch-acm-browser \
  --run-id 1 \
  --cdp-url http://127.0.0.1:9222 \
  --limit 10 \
  --db taxonomy.sqlite
```

This fallback is opt-in. It exists for legitimate ACM/institutional access and should not be used as a default bulk scraper. CDP lets the CLI connect to a real logged-in browser session and reuse the user's authenticated access, but it does not hide automation from ACM. Large automated downloads may violate publisher or institutional usage policies.

### 8. Assess Paper Inputs

Command:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli assess-paper-inputs \
  --run-id 1 \
  --db taxonomy.sqlite
```

This stage checks whether each paper has enough input to start taxonomy analysis.

Readiness levels include:

- abstract only
- abstract plus links
- full text
- full text plus links

The goal is to know which papers are ready for the intelligent taxonomy portion and which ones need review or manual PDF upload.

### 9. Agentic Paper Analysis

Command:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli agentic-paper-analysis \
  --run-id 1 \
  --llm \
  --db taxonomy.sqlite
```

For one paper:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli agentic-paper-analysis \
  --run-id 1 \
  --paper-id 42 \
  --llm \
  --db taxonomy.sqlite
```

This stage is the bridge into taxonomy synthesis. It analyzes paper text and snippets to produce:

- wireless/non-wireless label
- wireless confidence
- evidence
- paper summary
- dataset claims
- dataset relationship type
- modality evidence
- OSI L1-L7 evidence
- availability hints
- review-needed flags

The current implementation supports a deterministic analyzer and an LLM-backed analyzer. The deterministic analyzer is useful for tests and regression safety. The LLM analyzer is intended for higher-fidelity synthesis.

### 10. Reflect Paper Analysis

Command:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli reflect-paper-analysis \
  --run-id 1 \
  --db taxonomy.sqlite
```

This stage reviews prior analysis outputs and flags weak or unsupported claims. It is currently deterministic and is meant to reduce hallucination risk by checking whether claims are grounded in available text/evidence.

### 11. Extract Datasets

Command:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli extract-datasets \
  --run-id 1 \
  --db taxonomy.sqlite
```

This is an older deterministic extraction path. It still remains useful as a fallback/regression path while the agentic taxonomy synthesis matures.

### 12. Check Availability

Command:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli check-availability \
  --run-id 1 \
  --db taxonomy.sqlite
```

This checks whether dataset URLs appear open, closed, missing, or uncertain.

### 13. Resolve Reuse

Command:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli resolve-reuse \
  --run-id 1 \
  --db taxonomy.sqlite
```

This computes reuse counts and identity relationships across datasets.

### 14. Export

Command:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli export \
  --run-id 1 \
  --format xlsx \
  --out taxonomy.xlsx \
  --db taxonomy.sqlite
```

JSON export:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli export \
  --run-id 1 \
  --format json \
  --scope related \
  --out taxonomy.json \
  --db taxonomy.sqlite
```

Supported formats:

- `csv`
- `xlsx`
- `json`

The intended end state is a clean CSV/XLSX workbook that matches the manual taxonomy structure and sends uncertain rows to `Review Needed`.

## Paper-List Coverage (Jaccard)

When automated full-text fetching is blocked (e.g. ACM), the pipeline can still extract titles and abstracts and classify whether a paper is wireless. To measure how well the automated path reproduces a hand-curated wireless paper list, the CLI can emit a flat paper set and compute a Jaccard (intersection-over-union) score keyed on normalized titles.

Because a manual taxonomy sheet typically holds the *wireless* papers across *many* conferences, the comparison is made like-for-like: the automated side defaults to papers the pipeline classified as wireless, and the manual side is filtered to the run's conference + year.

### Export the fetched paper set

```bash
# Full ingested list for the run's conference:
PYTHONPATH=src python3 -m wireless_taxonomy.cli paper-set \
  --run-id 1 --out sigcomm-2024-papers.csv --format csv --db taxonomy.sqlite

# Only the papers the pipeline classified as wireless (run classify-wireless first):
PYTHONPATH=src python3 -m wireless_taxonomy.cli paper-set \
  --run-id 1 --out sigcomm-2024-wireless.csv --wireless-only --db taxonomy.sqlite
```

Each row has these columns (`match_key` is the normalized title — lowercased, alphanumeric-only):

```text
match_key, title, abstract, authors, doi, year, venue
```

`--format json` is also supported. Output is scoped to the run's conference instance. `--wireless-source` selects the wireless decision source: `classify` (keyword rules over title+abstract, default) or `agentic` (the LLM analysis stage).

### Compute Jaccard against a manual list

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli classify-wireless --run-id 1 --db taxonomy.sqlite
PYTHONPATH=src python3 -m wireless_taxonomy.cli jaccard \
  --run-id 1 \
  --manual "Wireless Taxonomy Record - List of Papers.csv" \
  --out coverage-report.json \
  --db taxonomy.sqlite
```

Defaults, all overridable:

- **Wireless-only** automated set (`--all-papers` compares the full ingested list instead).
- **Conference + year filtering** of the manual CSV to the run (`--no-conference-filter` to disable). The `Conference`/`Venue` and `Year` columns are auto-detected; override with `--conference-col` / `--year-col`. The manual conference value must match the run's `--venue` (case-insensitive).
- **Title column** auto-detected (`title` / `paper title` / `paper_title`); override with `--title-col "Paper Title"`.

Both sides are normalized with the same `normalize_title`, so keys line up deterministically. The command prints the index and counts:

```text
Paper-list coverage (Jaccard/IoU). venue=SIGCOMM year=2024 wireless_only=True conference_filtered=True index=0.8421 intersection=8 union=10 automated=9 manual=9 missed_by_cli=1 extra_from_cli=1 title_column='Paper Title'
```

`--out` writes a diff report listing `matched`, `missed_by_cli` (curated wireless papers the pipeline didn't flag), and `extra_from_cli` (pipeline-flagged papers absent from the manual list) so coverage gaps are diagnosable, not just a single number.

## One-Command Run

The CLI has a convenience `run` command:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli run \
  --venue SIGCOMM \
  --year 2025 \
  --url https://conferences.sigcomm.org/sigcomm/2025/program/papers-info/ \
  --out sigcomm-2025.xlsx \
  --format xlsx \
  --llm \
  --db taxonomy.sqlite
```

Use `--yes` to proceed past scope warnings without an interactive prompt:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli run \
  --venue SIGCOMM \
  --year 2025 \
  --url https://conferences.sigcomm.org/sigcomm/2025/program/papers-info/ \
  --out sigcomm-2025.csv \
  --format csv \
  --yes \
  --db taxonomy.sqlite
```

For debugging, prefer running individual stages. This makes it easier to inspect each artifact and rerun only the failing stage.

## Environment

Copy `.env.example` to `.env` and fill in the providers you want.

Important variables:

```text
WIRELESS_TAXONOMY_LLM_PROVIDER=openai
WIRELESS_TAXONOMY_LLM_FALLBACKS=anthropic,google

WIRELESS_TAXONOMY_ENABLE_WEB_SEARCH=1

SEMANTIC_SCHOLAR_API_KEY=
S2_API_KEY=
WIRELESS_TAXONOMY_UNPAYWALL_EMAIL=
UNPAYWALL_EMAIL=

OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
GOOGLE_API_KEY=
```

Semantic Scholar's keyed API limit is currently treated conservatively:

```text
WIRELESS_TAXONOMY_SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS=1.10
WIRELESS_TAXONOMY_SEMANTIC_SCHOLAR_RETRIES=2
```

Unpaywall requires an email address, not an API key.

Check the detected LLM configuration with:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli llm-config
```

## Full-Text Strategy

Full text is required for high-quality taxonomy synthesis. The pipeline tries to maximize legal, programmatic retrieval before asking for manual input.

Preferred order:

1. source/proceedings links
2. open-access resolvers
3. Semantic Scholar title/author/DOI lookup
4. Unpaywall DOI lookup
5. arXiv/OpenReview title lookup
6. bounded web search
7. local PDF directory
8. authenticated browser fallback for sources like ACM, only when the user has legitimate access

The system records every candidate, artifact, snippet, and failure reason so missing full text is diagnosable instead of silent.

## Review Philosophy

The pipeline should not pretend uncertain claims are certain. When evidence is weak, missing, ambiguous, or contradicted, the item should be routed to review.

Review rows are created for cases such as:

- malformed or suspicious paper lists
- missing important paper metadata
- failed full-text retrieval despite PDF candidates
- unmatched local PDFs
- weak dataset claims
- missing modality or OSI evidence
- reflection-stage grounding failures
- uncertain dataset availability

## Refactor/Cleanup Completed

Recent cleanup focused on reducing duplicated code while preserving behavior.

Completed:

- removed placeholder/dead modules
- removed unused embedding and metadata-check config
- added `.gitignore` entries for generated artifacts
- added optional browser dependency group
- split PDF text extraction into `analyze/pdf_text.py`
- split title/DOI/author matching into `analyze/text_match.py`
- split full-text resolvers into `analyze/full_text_resolvers.py`
- reduced `analyze/full_text.py` from roughly 1061 lines to roughly 585 lines
- centralized paper text persistence in `Pipeline._persist_paper_text_enrichment`
- refactored `enrich_paper_text`, `discover_full_text`, `add_pdfs`, and `fetch_acm_browser` to share one persistence path

Behavior was verified after cleanup with:

```bash
PYTHONPATH=src python3 -m pytest -q
python3 -m compileall -q src tests
```

## Current Known Gaps

The project is functional but not final.

Known remaining work:

- split `tests/test_pipeline.py` by pipeline stage
- continue reducing `pipeline.py` by moving stage-specific logic into smaller service modules
- improve final CSV/XLSX schema fidelity against the manual Google Sheet
- harden LLM JSON contracts and reflection prompts
- add more regression fixtures for full-text retrieval and taxonomy synthesis
- decide whether older deterministic dataset extraction remains as fallback or is replaced by the agentic path
- add stronger guardrails around authenticated publisher fallbacks

## Development Notes

Run tests:

```bash
PYTHONPATH=src python3 -m pytest -q
```

Run compile check:

```bash
python3 -m compileall -q src tests
```

Inspect CLI:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli --help
```

Recommended debugging workflow:

1. Run `ingest`.
2. Run `verify-paper-list`.
3. Run `assess-scope`.
4. Run `enrich-paper-text`.
5. Run `discover-full-text`.
6. If needed, run `add-pdfs`.
7. Run `assess-paper-inputs`.
8. Run `agentic-paper-analysis` for one paper first.
9. Run `reflect-paper-analysis`.
10. Export JSON before CSV/XLSX for easier inspection.

Example JSON export for debugging:

```bash
PYTHONPATH=src python3 -m wireless_taxonomy.cli export \
  --run-id 1 \
  --format json \
  --scope related \
  --out debug-run.json \
  --db taxonomy.sqlite
```
