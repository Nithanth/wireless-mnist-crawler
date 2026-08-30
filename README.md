# Wireless Dataset Reuse Taxonomy

This repository contains the AI-assisted analysis pipeline for the paper **“Why Aren't We Reusing Datasets More in Wireless Research?”** The pipeline retrieves scholarly metadata, classifies papers for wireless relevance, extracts structured dataset annotations from available paper text, consolidates dataset identities across papers, and evaluates public availability and reuse.

The accompanying annotated dataset is available on [Hugging Face](https://huggingface.co/datasets/nithanthram/wireless-reuse-taxonomy).

## Pipeline overview

```text
Venue proceedings and scholarly metadata
                    │
                    ▼
       Wireless relevance classification
                    │
                    ▼
       Open-access full-text discovery
                    │
                    ▼
       Structured dataset extraction
                    │
                    ▼
 Dataset identity resolution and deduplication
                    │
                    ▼
 Availability evaluation and consolidated export
```

The main stages are:

1. **Paper retrieval:** retrieve venue proceedings from DBLP and enrich missing metadata through OpenAlex, Crossref, Semantic Scholar, arXiv, and venue sources.
2. **Paper classification:** classify papers as `yes`, `no`, or `maybe` according to whether wireless communication, sensing, or networking is central to the work.
3. **Dataset extraction:** extract dataset names, modalities, OSI-layer coverage, collection environments, availability evidence, and supporting metadata from paper text.
4. **Entity resolution:** consolidate repeated mentions using exact names, normalized URLs, similarity-based candidates, and LLM confirmation.
5. **Availability evaluation:** classify practical public access using extracted evidence, URL verification, and targeted review.
6. **Export:** produce canonical dataset, paper, bibliography, and validation files with dataset reuse counts.

The complete prompt inventory is documented in [`PROMPTS.md`](PROMPTS.md).

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Configure at least one supported LLM provider in `.env`. Optional metadata and web-search credentials can improve abstract and open-access PDF coverage. Secret files are excluded by `.gitignore`.

## Quick start

```bash
wt init wireless_v1
wt add --venues SIGCOMM,IMC,NSDI,ICC,TWC --years 2022:2025
wt export
wt status
```

Use `wt add --estimate` to inspect the expected workload before running the pipeline. Corpus state, results, and snapshots are stored under `corpora/<name>/`.

### Primary commands

| Command | Purpose |
|---|---|
| `wt init` | Create or adopt a corpus workspace. |
| `wt add` | Retrieve papers, discover full text, classify papers, and extract datasets. |
| `wt export` | Reconcile dataset identities, evaluate availability, and write consolidated outputs. |
| `wt status` | Inspect active-corpus coverage and extraction status. |
| `wt use` | List corpora or switch the active corpus. |
| `wt rollback` | Restore the snapshot preceding the latest corpus run. |
| `wt advanced` | Access individual pipeline stages, evaluation, and maintenance commands. |

Run `wt COMMAND --help` for command-specific options.

## Exported data

The export stage produces:

| File | Description |
|---|---|
| `consolidated_datasets.csv` | Canonical dataset catalog across all extraction sources. |
| `consolidated_datasets_pdf_only.csv` | Canonical catalog with full-paper-text-backed extraction evidence; this is the primary dataset used for the paper's dataset-level analysis. |
| `consolidated_papers.csv` | Papers retained as wireless-relevant, with citation keys and extracted dataset names. |
| `consolidated_bibtex.csv` | Bibliographic records for retained papers. |
| `review_candidates.csv` | Ambiguous entity-resolution candidates for review. |

`Reuse Count` is computed after entity resolution as the number of distinct corpus papers associated with a canonical dataset. It is separate from the per-paper relationship between a paper and a dataset.

## Dataset identity resolution

Dataset mentions are consolidated conservatively:

1. Exact-name matches are grouped.
2. Records sharing a normalized availability URL are matched.
3. Fuzzy similarity over names, modalities, and OSI-layer information generates candidate pairs.
4. An LLM confirms or rejects ambiguous candidates.
5. Uncertain pairs remain separate and can be reviewed manually.

This strategy prioritizes avoiding false merges, although it may under-count reuse when two descriptions lack enough evidence to establish identity.

## Validation and figures

The repository includes a manually curated validation workflow for comparing paper and dataset recovery using precision, recall, and Jaccard similarity. The publication figure and table generator is:

```bash
python scripts/generate_validation_figures.py
```

An alternate output directory can be supplied as the first argument. The released validation CSVs on Hugging Face provide the manual and matched AI-assisted subsets used in the paper.

## Testing

```bash
pytest -q
```

## Public dataset

The annotated catalog, validation files, bibliography, schema documentation, and checksums are hosted at:

https://huggingface.co/datasets/nithanthram/wireless-reuse-taxonomy

The release contains scholarly metadata and derived annotations. It does not redistribute the underlying wireless datasets or paper full text.
