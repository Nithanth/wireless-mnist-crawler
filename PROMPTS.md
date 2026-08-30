# LLM Prompts Used in the Wireless Taxonomy Pipeline

This document lists every LLM prompt used in the pipeline, organized by stage.
Each entry includes the source file, line numbers, the full prompt text, and
what inputs are injected.

---

## Pipeline Overview

```
DBLP / Venue Website
        │
        ▼
[1] Paper List Ingest ──► papers table (title, authors, DOI, abstract)
        │
        ▼
[2] Wireless Classification ──► is_wireless = yes/no/maybe
        │                    (uses abstract OR full PDF, whichever is available)
        ▼
[3] PDF Fetch + OA Search ──► pdf_bytes / pdf_text (or abstract fallback)
        │
        ▼
[4] Dataset Extraction ──► per-paper dataset records
        │                 (requires PDF; abstract-only fallback is strict)
        ▼
[5] Entity Resolution / Dedup ──► consolidated_datasets.csv
        │                          (URL dedup → similarity match → LLM confirm)
        ▼
[6] Availability Fill ──► availability_status + availability_url
        │                  (LLM reads paper text, then URL is verified)
        ▼
consolidated_datasets_pdf_only.csv (322 datasets)
```

**Key distinction for the paper:**
- **Paper classification** (6,560 papers): uses abstracts OR PDFs — the AI
  classifies as best it can with whatever text is available.
- **Dataset extraction** (322 datasets in the final PDF-backed set): prefers full PDF.
  Abstract-only extractions are heavily gated (only explicitly named datasets)
  and excluded from the pdf-only set used in the paper's dataset stats.

---

## 1. Paper List Ingest

**Stage:** Paper list extraction from conference websites (non-DBLP sources)
**File:** `src/wireless_taxonomy/ingest/url.py`, lines 256–305
**Function:** `_paper_list_prompt(page, venue, year, source_hint)`
**Task name:** `paper_list_extraction`
**Schema:** `PaperSeedList`

**When used:** When ingesting papers from a URL source (not DBLP). DBLP
ingestion (`ingest/dblp.py`) parses the DBLP API directly without an LLM.

**Inputs injected:**
- `venue` — conference venue name (e.g., "SIGCOMM")
- `year` — conference year
- `source_hint` — detected source type (e.g., "generic", "usenix")
- `page.source_url` — URL of the page being scraped
- `text` — cleaned page text (truncated to 120,000 chars)
- `links` — preserved hyperlinks from the page (up to 500)

**Full prompt:**

```
You extract accepted conference paper records from heterogeneous conference webpages.

Venue: {venue}
Year: {year}
Source type hint: {source_hint}
Source URL: {page.source_url}

Return JSON only, with this exact top-level shape:
{{
  "papers": [
    {{
      "title": "full paper title",
      "authors": ["author names or the exact author line if individual names are ambiguous"],
      "venue": "{venue}",
      "year": {year},
      "session": "session name if visible, otherwise null",
      "abstract": "full abstract if visible, otherwise null",
      "doi": "DOI if visible or inferable from ACM/IEEE DOI URL, otherwise null",
      "paper_url": "canonical paper landing page URL if visible, otherwise null",
      "pdf_url": "PDF URL if visible, otherwise null",
      "confidence": 0.0,
      "evidence_text": "short source snippet proving this record"
    }}
  ]
}}

Extraction rules:
- Extract accepted/research/full conference papers, not navigation, sessions,
  awards without paper records, keynotes, chairs, workshops, or videos alone.
- Preserve titles exactly.
- Preserve author information with high fidelity; if individual author splitting
  is uncertain, put the exact author line as one array item.
- Include all visible abstracts and continue across wrapped lines.
- Use preserved links to assign DOI, paper URL, and PDF URL when possible.
- If uncertain about a record, include it with confidence below 0.90 rather
  than omitting it.
- Do not invent missing metadata.

Cleaned page text:
<<<PAGE_TEXT
{text}
PAGE_TEXT

Preserved links:
<<<LINKS
{links}
LINKS
```

---

## 2. Wireless Classification

**Stage:** Classifying whether a paper is about wireless networking
**File:** `src/wireless_taxonomy/analyze/candidates.py`, lines 324–408
**Function:** `_prompt(paper, has_pdf, keyword_snippets)`
**Task name:** `wireless_candidate_classification`
**Schema:** `WirelessCandidate`

**When used:** Twice in the pipeline:
1. **Standalone `classify` command** — uses title + abstract only (no PDF).
2. **`extract-datasets` command** — reclassifies using full PDF when available,
   or keyword snippets from PDF text (`--classify-no-pdf` mode), or falls back
   to title + abstract.

**Inputs injected:**
- `context_note` — varies based on input type:
  - Full PDF: `"You have the full paper text below."`
  - Keyword snippets: `"You have keyword-matched excerpts from the full paper."`
  - Abstract only: `"You only have the paper's title and abstract."`
- `paper_json` — JSON with `title` and `abstract`
- `snippet_section` — keyword-matched excerpts from PDF text (only in
  `--classify-no-pdf` mode; empty otherwise)

**Full prompt:**

```
You screen one research paper to decide if it is a WIRELESS / wireless-networking paper.
{context_note}

Wireless covers (non-exhaustive):
- Cellular: 4G/LTE, 5G/NR, 6G, RAN, base stations, O-RAN, small cells, HetNets
- Wi-Fi / 802.11: any variant (ax, be, ad, ay), WLAN, access points
- Millimeter-wave (mmWave), sub-THz, terahertz communications
- MIMO, massive MIMO, beamforming, antenna design, phased arrays
- RF / spectrum: channel modeling, CSI, RSSI, SINR, propagation, spectrum sharing,
  cognitive radio, dynamic spectrum access
- IoT wireless: LoRa, LPWAN, Zigbee, Bluetooth/BLE, UWB, RFID, NFC, backscatter
- Satellite / non-terrestrial networks: LEO, GEO, direct-to-cell, satellite IoT
- Wireless sensing: radar, WiFi sensing, RF sensing, localization, positioning
- Vehicular: V2X, V2V, DSRC, C-V2X
- Device-to-device (D2D), mesh, ad-hoc networks
- Mobile edge / MEC when the wireless link is central
- ML/AI *applied to* wireless problems (e.g., deep learning for channel estimation,
  RL for spectrum management) — these ARE wireless papers

NOT wireless (label "no"):
- Wired/datacenter networking (switches, RDMA, flow scheduling, optical fiber)
- Pure distributed systems, storage, OS, or databases
- Video/streaming/CDN where the radio link is merely the last-hop access
- Pure ML/theory with no wireless application

Judge by the paper's *central* contribution. Label "yes" when the wireless
link/medium is core to the work, not just incidentally mentioned.

Examples:
- Title: "EdgeRIC: Empowering Real-time Intelligent Optimization and Control in NextG Cellular Networks"
  → {"label": "yes", "confidence": "high", "evidence": "Core contribution is a RAN intelligent controller for 5G cellular."}
- Title: "DINT: Fast In-Kernel Distributed Transactions with eBPF"
  → {"label": "no", "confidence": "high", "evidence": "In-kernel datacenter transactions, no wireless component."}
- Title: "Habitus: Boosting Mobile Immersive Content Delivery through Full-body Pose Tracking and Multipath Networking"
  → {"label": "maybe", "confidence": "medium", "evidence": "Uses mmWave multipath but main focus is immersive content delivery."}

Return JSON only:
{{
  "label": "yes" or "no" or "maybe",
  "confidence": "high" or "medium" or "low",
  "evidence": "short reason grounded in the paper content"
}}

Rules:
- "yes": wireless link/medium is clearly central to the paper.
- "no": clearly not about wireless.
- "maybe": ambiguous, mixed, or not enough information to be sure.
- confidence: "high" = very certain, "medium" = fairly sure, "low" = uncertain.
- Keep evidence to one short sentence.

Paper:
{paper_json}
{snippet_section}
```

**Output:** `{"label": "yes"/"no"/"maybe", "confidence": "high"/"medium"/"low", "evidence": "..."}`

---

## 3. Open Access / PDF Web Search

**Stage:** Finding a downloadable PDF for a paper when none is in the metadata
**File:** `src/wireless_taxonomy/analyze/oa_availability.py`, lines 581–598
**Function:** `_llm_web_search(title, doi)`
**Task name:** `oa_pdf_web_search`
**Schema:** `PdfUrlSearch`
**Special:** Uses `use_web_search=True` (LLM web search capability enabled)

**When used:** When a paper has no PDF URL in its metadata and the pipeline
needs the full text for dataset extraction. The LLM searches the web for a
direct PDF download link.

**Inputs injected:**
- `title` — paper title
- `doi` — paper DOI (if available)

**Full prompt:**

```
Find a direct, publicly downloadable PDF for this exact research paper.
  Title: {title}
  DOI: {doi}

Search strategy:
1. Search for the title in quotes plus 'pdf' or 'filetype:pdf'.
2. Also try the title plus 'paper.pdf' or the last name of the first author.
3. Examine the first 5 results, including any from .edu, university
   homepages, institutional repositories, or preprint servers.
4. Prefer URLs that end in .pdf and are hosted on author/university pages.

Do NOT return ResearchGate, IEEE Xplore, ACM DL, arXiv abstract pages, or
any landing page that requires clicking. The URL must point directly at a
downloadable .pdf file (or redirect immediately to one).

Return JSON only: {"pdf_url": "<direct PDF URL or empty string if none found>"}
```

**Output:** `{"pdf_url": "<URL or empty string>"}`

---

## 4. Dataset Extraction

**Stage:** Extracting dataset metadata from a paper's full text
**File:** `src/wireless_taxonomy/analyze/dataset_extractor.py`, lines 665–791
**Variable:** `_EXTRACTION_PROMPT_TMPL`
**Task name:** `dataset_extraction`
**Schema:** `DatasetExtraction`

**When used:** For every wireless-classified paper that has full PDF text
(or abstract-only with strict gating — see variant 4b below).

**Inputs injected:**
- `title` — paper title
- `authors` — paper authors
- `venue` — conference/journal venue
- `year` — publication year
- `doi` — paper DOI
- `text_section` — one of:
  - Full PDF text (when PDF bytes available)
  - Pre-extracted PDF text (when cached)
  - Abstract-only text with STRICT ABSTRACT MODE rules (see 4b)

**Full prompt:**

```
You are a research assistant extracting structured dataset metadata from a wireless networking/systems paper.

Paper metadata:
  Title: {title}
  Authors: {authors}
  Venue: {venue} {year}
  DOI: {doi}
{text_section}
Extract TWO kinds of datasets:

(A) REUSED datasets — data the paper uses from someone else.
    These MUST have a proper, searchable name (e.g. "CRAWDAD dartmouth/campus",
    "Widar 3.0", "FCC MBA dataset", "Ookla Speedtest Intelligence").
    If the paper references a dataset only by citation (e.g. "the traces from [54]"),
    look up that reference in the paper's bibliography and use the ACTUAL dataset
    name or the cited paper's title as the name.

(B) INTRODUCED datasets — data the paper itself collected/curated/released.
    Even if the paper does NOT give it a branded name, extract it IF:
    - The paper collected real measurements, traces, or recordings
    - The data has enough detail to be reproducible (description of what was
      measured, duration, scale, environment)
    - It's wireless/networking data (not generic compute benchmarks)
    For the name field: use the paper's own branded name if it has one
    (e.g. "Lumos5G", "5G-Trace-NYC", "OpenLoRa"). Otherwise, construct
    a descriptive canonical name in the format:
      "[Technology/Protocol] [What Was Measured] [Dataset|Traces|Measurements]"
    The name should describe WHAT the data is, not WHO collected it.
    Do NOT include author names, years, or citation keys in dataset names.
    Examples of good unnamed-dataset names:
      - "Cellular UAV Video Delivery Traces"
      - "5G Mobility Drive-Test Dataset"
      - "Indoor WiFi CSI Sensing Measurements"
      - "LEO Satellite RF Link Measurements"
      - "mmWave Backscatter Indoor Testbed Data"
    The name should be generic enough that another paper referencing the same
    data would plausibly use the same or very similar name.

ONLY extract datasets that measure wireless or networking phenomena directly:
- RF/signal measurements (RSSI, SNR, CSI, spectrum, channel impulse response, IQ samples)
- Network performance traces (throughput, latency, packet loss, RTT, PCAP)
- Cellular/WiFi/satellite/IoT protocol data (RRC, NAS, LTE, 5G NR, LoRa logs)
- Wireless system deployment data (base station locations, coverage maps, link budgets)
- Sensor network or IoT measurement traces (when the sensing medium is wireless)

DO NOT extract auxiliary/contextual datasets the paper uses as inputs but which
do not measure wireless phenomena — even if used by a wireless paper:
- Demographic, census, population, or socioeconomic data
- Weather, climate, or environmental data (temperature, rain, wind)
- Social media, news, or text corpora
- Geographic/map data not specifically measuring wireless coverage
- Financial, economic, or government administrative data
- Generic ML benchmarks (ImageNet, MNIST, CIFAR) unless applied to wireless data
- Software tools, simulators, or libraries (ns-3, MATLAB, PyTorch)
- Synthetic data generated on-the-fly without a persistent shareable artifact
- Figure or table references ("Figure 10", "Table 2")
- Bare technical terms without context ("testbed", "pcap", "pings", "traces")
- Data that is only mentioned in passing without detail

For each dataset extract:
- name: For REUSED: the exact proper name (searchable, distinctive). For
  INTRODUCED: the paper's own branded name, or a descriptive canonical name
  (see naming rules above). NEVER use author names or years in dataset names.
- relationship_type: "introduced" (paper creates/releases), "reused" (uses existing),
  "extended" (augments existing), "compared_against", "unclear"
- modalities: list of data types (e.g. ["5G NR traces", "RSRP measurements", "PCAP"])
- osi_layers: list from ["L1","L2","L3","L4","L5","L6","L7"]
- availability: true if publicly available, false if restricted, null if unknown
- availability_url: exact URL from the paper (empty string if none — do NOT guess)
- availability_notes: exact sentence from paper about access/license, or empty string
- collection_environment: one of "Physical Lab Testbed", "Real World Deployment",
  "Simulation", "Crowdsourced", "Unknown"
- known_users: up to 5 OTHER papers that also use this dataset ([] if unsure — do not hallucinate)
- confidence: "high" (clear named dataset with strong evidence), "medium"
  (reasonable but less certain), or "low" (weak/speculative)
- evidence_text: one sentence quoting or closely paraphrasing the paper

Examples:
{
  "datasets": [
    {
      "name": "5G-Trace-NYC",
      "relationship_type": "introduced",
      "modalities": ["5G NR throughput logs", "GPS coordinates", "signal strength"],
      "osi_layers": ["L1", "L3"],
      "availability": true,
      "availability_url": "https://github.com/example/5g-trace-nyc",
      "availability_notes": "We release our dataset at https://github.com/example/5g-trace-nyc under MIT license.",
      "collection_environment": "Real World Deployment",
      "known_users": [],
      "confidence": "high",
      "evidence_text": "We collected 5G NR traces across 12 routes in NYC over 3 months."
    },
    {
      "name": "Cellular UAV Video Delivery Traces",
      "relationship_type": "introduced",
      "modalities": ["LTE/5G throughput logs", "video quality metrics", "GPS flight paths"],
      "osi_layers": ["L3", "L7"],
      "availability": null,
      "availability_url": "",
      "availability_notes": "",
      "collection_environment": "Real World Deployment",
      "known_users": [],
      "confidence": "medium",
      "evidence_text": "We collected 47 drone flights with concurrent LTE and 5G measurements across 3 operators."
    },
    {
      "name": "CRAWDAD dartmouth/campus",
      "relationship_type": "reused",
      "modalities": ["WiFi association logs", "AP locations"],
      "osi_layers": ["L2"],
      "availability": true,
      "availability_url": "https://crawdad.org/dartmouth/campus",
      "availability_notes": "",
      "collection_environment": "Real World Deployment",
      "known_users": ["Diversity in Smartphone Usage (IMC 2010)", "Modeling WiFi Availability (SIGCOMM 2005)"],
      "confidence": "high",
      "evidence_text": "We evaluate our model on the CRAWDAD dartmouth/campus WiFi trace."
    }
  ]
}

If the paper uses NO datasets and introduces none, return {"datasets": []}.
Prefer quality over quantity — only extract datasets you are reasonably sure about.

Return ONLY valid JSON — no markdown, no explanation outside the JSON.
```

**Output:** `{"datasets": [{name, relationship_type, modalities, osi_layers, availability, availability_url, availability_notes, collection_environment, known_users, confidence, evidence_text}, ...]}`

---

## 4b. Dataset Extraction — Abstract-Only Variant

**Stage:** Dataset extraction when only the abstract is available (no PDF)
**File:** `src/wireless_taxonomy/analyze/dataset_extractor.py`, lines 936–951
**Task name:** `dataset_extraction` (same as above, with appended rules)

**When used:** When a paper has no PDF text. This is a **strict gated mode**:
the abstract is first checked with a regex pre-filter
(`abstract_dataset_tier()`). Only if the abstract explicitly names a dataset
(tier "named") does the LLM get called. If the abstract only describes data
collection without naming a dataset (tier "collection"), or has no dataset
language at all (tier "none"), extraction is skipped entirely.

**Pre-filter tiers** (`dataset_extractor.py`, lines 122–139):
- `"none"` — no dataset language → skip, no LLM call
- `"collection"` — describes data collection but no named dataset → skip
- `"named"` — abstract contains a proper-noun dataset name → proceed with LLM

**Additional text appended to the main extraction prompt:**

```
STRICT ABSTRACT MODE: you only have the abstract, not the full paper.
Rules:
1. Extract a dataset ONLY if the abstract contains its EXPLICIT NAME —
   a proper noun like 'DeepSense 6G', 'MNIST', 'Ookla Speedtest', etc.
2. If the abstract only says 'we collected data', 'we measured', or
   'we used data from an operator/provider' WITHOUT naming the dataset,
   return {"datasets": []}. Do NOT invent a name.
3. Do NOT infer datasets from paper topic or domain knowledge. Only
   extract what is explicitly named in the abstract text above.
4. For each named dataset: fill in availability (Y/N + URL if stated).
   Leave OSI layers, modalities, and collection environment blank — they
   will be filled from the full PDF if available.
5. If in doubt, return {"datasets": []}.
```

**Inputs injected:** `abstract[:8000]` (truncated abstract text)

**Note:** Datasets extracted in abstract-only mode are excluded from
`consolidated_datasets_pdf_only.csv` (the 322-dataset set used in the paper).

---

## 5. Entity Resolution / Deduplication

**Stage:** Merging dataset records that refer to the same underlying dataset
**File:** `src/wireless_taxonomy/postprocess/entity_resolution.py`, lines 207–241
**Variable:** `_LLM_CONFIRM_PROMPT`
**Task name:** `entity_resolution`

**When used:** During the `export` command's reconciliation step. The pipeline
first applies URL-based dedup (same availability URL → same dataset) and
similarity-based matching (fuzzy name match above a threshold). For ambiguous
pairs, the LLM confirms whether two dataset records are the same.

**Inputs injected:**
- `a_name`, `b_name` — dataset names
- `a_keys`, `b_keys` — comma-separated BibTeX keys of papers using each dataset
- `a_mod`, `b_mod` — modalities
- `a_osi`, `b_osi` — OSI layers
- `a_env`, `b_env` — collection environment
- `a_url`, `b_url` — availability URLs

**Full prompt:**

```
You are a research dataset deduplication assistant. Given two dataset descriptions
extracted from different academic papers, determine whether they refer to the SAME
underlying dataset (possibly with different names or slightly different descriptions).

Dataset A:
  Name: {a_name}
  Papers: {a_keys}
  Modalities: {a_mod}
  OSI Layers: {a_osi}
  Environment: {a_env}
  Availability URL: {a_url}

Dataset B:
  Name: {b_name}
  Papers: {b_keys}
  Modalities: {b_mod}
  OSI Layers: {b_osi}
  Environment: {b_env}
  Availability URL: {b_url}

Are these the SAME dataset? Consider:
- Same name or clearly referring to the same artifact (e.g. one is an abbreviation)
- From the same measurement campaign or data collection effort
- Would a reader treat these as the same dataset if cited?

Be conservative: answer "yes" ONLY if the evidence clearly indicates the same
underlying artifact (same name/URL/campaign). Two datasets that merely measure
the same phenomenon (e.g. two different indoor WiFi CSI collections) are NOT
the same. If the evidence is insufficient to decide, answer "unsure" — never
guess "yes". A wrong merge corrupts reuse statistics; a missed merge is
recoverable in review.

Respond with JSON: {"verdict": "yes" | "no" | "unsure", "reason": "<brief explanation>"}
```

**Output:** `{"verdict": "yes"/"no"/"unsure", "reason": "..."}`

**Dedup pipeline order:**
1. **URL dedup** — same availability URL → auto-merge (no LLM)
2. **Similarity match** — fuzzy name similarity above threshold → LLM confirm
3. **LLM confirm** — the prompt above, for ambiguous pairs
4. **Review candidates** — LLM "unsure" or skipped pairs → written to
   `review_candidates.csv` for manual review

---

## 6. Availability Fill

**Stage:** Filling in availability status for datasets marked "unknown"
**Used in two places:**

### 6a. Admin command (detailed prompt)

**File:** `src/wireless_taxonomy/commands/admin.py`, lines 360–376
**Variable:** `_AVAIL_PROMPT`
**Task name:** `fill_availability`
**Schema:** `AvailabilityCheck`

**Inputs injected:**
- `dataset_name` — canonical dataset name
- `title` — paper title
- `text[:80_000]` — truncated paper text (PDF or abstract)

**Full prompt:**

```
You are a research data curator. Given the text of a research paper,
determine whether the dataset named below is publicly available.

Dataset name: {dataset_name}
Paper title: {title}

Paper text:
---
{text}
---

Answer in JSON only:
{"available": true|false|null,
 "url": "<exact URL from paper or empty string>",
 "notes": "<one sentence from the paper about availability or empty string>"}

Rules:
- available=true only if the paper explicitly states the dataset is publicly downloadable.
- available=false if the paper says restricted, proprietary, or available only on request.
- available=null if the paper says nothing about availability.
- url must be copied verbatim from the paper — do NOT guess or construct URLs.
Return ONLY the JSON object, no markdown.
```

### 6b. Export command (compact prompt)

**File:** `src/wireless_taxonomy/commands/export.py`, lines 198–204
**Task name:** `fill_availability`

**Inputs injected:**
- `canonical_name` — dataset name from DB
- `title` — paper title
- `text[:8000]` — truncated paper text

**Full prompt:**

```
Dataset: {canonical_name}
Paper: {title}

Text:
{text}

Is this dataset publicly/openly available?
Answer JSON: {"available": true/false/null, "url": "..." or null}
```

**Output (both variants):** `{"available": true/false/null, "url": "...", "notes": "..."}`

**Post-LLM verification:** After the LLM labels a dataset as available, the
pipeline verifies that the URL actually resolves (HTTP check). Datasets where
the URL is dead or requires registration are downgraded to "closed". This is
why some datasets that are actually open get marked as closed (link rot or
strict URL verification — see the caveat in the paper).

---

## Summary Table

| # | Stage | File | Lines | Task Name | Schema | Key Inputs |
|---|-------|------|-------|-----------|--------|------------|
| 1 | Paper List Ingest | `ingest/url.py` | 256–305 | `paper_list_extraction` | `PaperSeedList` | venue, year, page text, links |
| 2 | Wireless Classification | `analyze/candidates.py` | 324–408 | `wireless_candidate_classification` | `WirelessCandidate` | title, abstract, keyword snippets |
| 3 | OA PDF Search | `analyze/oa_availability.py` | 581–598 | `oa_pdf_web_search` | `PdfUrlSearch` | title, DOI |
| 4 | Dataset Extraction | `analyze/dataset_extractor.py` | 665–791 | `dataset_extraction` | `DatasetExtraction` | title, authors, venue, year, DOI, paper text |
| 4b | Dataset Extraction (abstract-only) | `analyze/dataset_extractor.py` | 936–951 | `dataset_extraction` | `DatasetExtraction` | abstract (with strict rules) |
| 5 | Entity Resolution | `postprocess/entity_resolution.py` | 207–241 | `entity_resolution` | — | two dataset records (name, papers, modalities, OSI, env, URL) |
| 6a | Availability Fill (admin) | `commands/admin.py` | 360–376 | `fill_availability` | `AvailabilityCheck` | dataset name, title, paper text |
| 6b | Availability Fill (export) | `commands/export.py` | 198–204 | `fill_availability` | — | dataset name, title, paper text |

**Notes:**
- All prompts are embedded as string literals in Python code (no external template files).
- Prompts use content-addressed caching based on paper content hash, model identity, and task name.
- 5 of 7 prompts use explicit schema names for structured JSON output validation.
- Only the OA PDF search prompt uses `use_web_search=True` (LLM web search capability).
- Classification has 3 input modes: full PDF, keyword snippets (`--classify-no-pdf`), or title+abstract only.
- Dataset extraction has 3 input modes: full PDF, pre-extracted text, or abstract-only with strict gating.
