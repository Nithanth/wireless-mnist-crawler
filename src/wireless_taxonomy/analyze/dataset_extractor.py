"""Dataset extraction from open-access paper full text using an LLM.

For each paper we:
1. Fetch the PDF bytes and pass them natively to Anthropic (document block) or
   Gemini (inline_data) — no lossy text extraction. Fall back to abstract text
   if no PDF is available.
2. Send to the LLM with a structured prompt returning datasets with modalities,
   OSI layers, availability (URL from paper text + live HEAD check), collection
   environment, and known reusers.
3. Generate a BibTeX entry via CrossRef DOI lookup or heuristic fallback.
4. Return structured ``DatasetExtractionResult`` objects for DB and CSV export.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

OSI_LAYERS = {"L1", "L2", "L3", "L4", "L5", "L6", "L7"}
COLLECTION_ENVS = {"Physical Lab Testbed", "Real World Deployment", "Simulation"}
RELATIONSHIP_TYPES = {"introduced", "reused", "extended", "compared_against", "unclear"}


@dataclass
class DatasetRecord:
    name: str
    relationship_type: str
    modalities: list[str]
    osi_layers: list[str]
    availability: bool | None
    availability_notes: str
    availability_url: str
    collection_environment: str
    known_users: list[str]
    confidence: float
    evidence_text: str


@dataclass
class DatasetExtractionResult:
    paper_id: int
    title: str
    authors: str
    venue: str
    year: int
    doi: str
    bibtex_key: str
    bibtex: str
    datasets: list[DatasetRecord]
    extraction_source: str
    error: str | None = None


_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _fetch_pdf_bytes(pdf_url: str, max_bytes: int = 1024 * 1024 * 10) -> bytes | None:
    """Download PDF and return raw bytes for native LLM attachment (up to 10 MB)."""
    try:
        req = urllib.request.Request(
            pdf_url,
            headers={"User-Agent": _BROWSER_UA, "Accept": "application/pdf,*/*"},
        )
        with urllib.request.urlopen(req, timeout=25) as r:
            raw = r.read(max_bytes)
        return raw if raw[:4] == b"%PDF" else None
    except Exception:
        return None


def _check_url_live(url: str) -> bool:
    """Return True if a URL responds with HTTP 2xx/3xx (HEAD, then GET fallback)."""
    if not url or not url.startswith("http"):
        return False
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": _BROWSER_UA},
                method=method,
            )
            with urllib.request.urlopen(req, timeout=8) as r:
                return r.status < 400
        except Exception:
            pass
    return False


def _fetch_crossref_bibtex(doi: str) -> str | None:
    """Retrieve BibTeX from CrossRef for a given DOI."""
    if not doi:
        return None
    url = f"https://doi.org/{urllib.parse.quote(doi, safe='/')}"
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/x-bibtex", "User-Agent": "wireless-taxonomy/0.1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            text = r.read().decode("utf-8", errors="replace")
        if text.strip().startswith("@"):
            return text.strip()
    except Exception:
        pass
    return None


def load_cached_pdf(conn, paper_id: int, pdf_url: str) -> bytes | None:
    """Return raw PDF bytes from paper_text_artifacts if previously fetched."""
    if conn is None:
        return None
    try:
        import base64
        row = conn.execute(
            "SELECT content_text, source_type FROM paper_text_artifacts "
            "WHERE paper_id = ? AND source_url = ? AND fetch_status = 'ok' LIMIT 1",
            (paper_id, pdf_url),
        ).fetchone()
        if row and row["content_text"]:
            if row["source_type"] == "pdf_b64":
                return base64.b64decode(row["content_text"])
            return None
    except Exception:
        pass
    return None


def store_cached_pdf(conn, paper_id: int, pdf_url: str, pdf_bytes: bytes) -> None:
    """Persist raw PDF bytes into paper_text_artifacts as base64 for lossless round-trip."""
    if conn is None:
        return
    try:
        import base64
        sha = hashlib.sha256(pdf_bytes).hexdigest()
        now = datetime.now(timezone.utc).isoformat()
        b64 = base64.b64encode(pdf_bytes).decode("ascii")
        conn.execute(
            """
            INSERT OR REPLACE INTO paper_text_artifacts
              (paper_id, source_type, source_url, fetch_status,
               content_text, content_sha256, fetched_at, created_at)
            VALUES (?, 'pdf_b64', ?, 'ok', ?, ?, ?, ?)
            """,
            (paper_id, pdf_url, b64, sha, now, now),
        )
    except Exception:
        pass


def _make_bibtex_key(authors: str, year: int, title: str) -> str:
    """Heuristic BibTeX citation key: firstauthorYYYYfirstword."""
    first_author = (authors.split(",")[0] if "," in authors else authors.split(" and ")[0]).strip()
    last_name = first_author.split()[-1].lower() if first_author.split() else "unknown"
    last_name = re.sub(r"[^a-z0-9]", "", last_name)
    first_word = re.sub(r"[^a-z0-9]", "", (title.split()[0] if title.split() else "paper").lower())
    return f"{last_name}{year}{first_word}"


def _make_minimal_bibtex(key: str, title: str, authors: str, year: int, venue: str, doi: str) -> str:
    author_field = authors.replace(";", " and ")
    lines = [
        f"@inproceedings{{{key},",
        f"  title     = {{{title}}},",
        f"  author    = {{{author_field}}},",
        f"  booktitle = {{{venue}}},",
        f"  year      = {{{year}}},",
    ]
    if doi:
        lines.append(f"  doi       = {{{doi}}},")
    lines.append("}")
    return "\n".join(lines)


_EXTRACTION_PROMPT_TMPL = """You are a research assistant extracting structured dataset metadata from a wireless networking/systems paper.

Paper metadata:
  Title: {title}
  Authors: {authors}
  Venue: {venue} {year}
  DOI: {doi}
{text_section}
A "dataset" is a NAMED, REUSABLE collection of measurement data, traces, or recordings
that other researchers could find, download, or request. It must have a PROPER NAME — a
distinctive identifier that someone could search for and recognise (e.g. "CRAWDAD
dartmouth/campus", "5G-Trace-NYC", "Widar 3.0", "FCC MBA dataset").

INCLUDE: wireless traces, spectrum measurements, channel recordings, network logs,
mobility traces, signal datasets with proper names.

DO NOT INCLUDE:
- Generic ML benchmarks (ImageNet, MNIST, CIFAR) unless applied to wireless data
- Software tools, simulators, or libraries (ns-3, MATLAB, PyTorch)
- Synthetic data generated on-the-fly without a persistent shareable artifact
- Figure or table references ("Figure 10", "Table 2")
- Generic pronoun references ("our dataset", "the data collected", "this dataset",
  "the same dataset", "the data", "collected traces")
- Descriptive phrases that are NOT proper names ("500,000 User Equipment",
  "2 Starlink flights", "25 flights operated by 7 airlines")
- Bare technical terms ("testbed", "pcap", "pings", "speed test", "traces",
  "logs", "artifact", "training set", "recorded signals")
- Vague time-based descriptions ("data collected during Feb-Apr 2022")

The litmus test: could another researcher Google this name and find the dataset?
If NOT, do not extract it.

For each dataset extract:
- name: the EXACT PROPER NAME from the paper. Must be a distinctive, searchable
  identifier — NOT a generic description. If the paper does not give data a
  specific proper name, skip it.
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
- confidence: 0.0–1.0 how sure this is a real, named, reusable dataset (0.9+ = clear
  proper name with evidence; 0.5–0.7 = ambiguous; below 0.5 = probably not a dataset)
- evidence_text: one sentence quoting or closely paraphrasing the paper

Example:
{{
  "datasets": [
    {{
      "name": "5G-Trace-NYC",
      "relationship_type": "introduced",
      "modalities": ["5G NR throughput logs", "GPS coordinates", "signal strength"],
      "osi_layers": ["L1", "L3"],
      "availability": true,
      "availability_url": "https://github.com/example/5g-trace-nyc",
      "availability_notes": "We release our dataset at https://github.com/example/5g-trace-nyc under MIT license.",
      "collection_environment": "Real World Deployment",
      "known_users": [],
      "confidence": 0.95,
      "evidence_text": "We collected 5G NR traces across 12 routes in NYC over 3 months."
    }},
    {{
      "name": "CRAWDAD dartmouth/campus",
      "relationship_type": "reused",
      "modalities": ["WiFi association logs", "AP locations"],
      "osi_layers": ["L2"],
      "availability": true,
      "availability_url": "https://crawdad.org/dartmouth/campus",
      "availability_notes": "",
      "collection_environment": "Real World Deployment",
      "known_users": ["Diversity in Smartphone Usage (IMC 2010)", "Modeling WiFi Availability (SIGCOMM 2005)"],
      "confidence": 0.90,
      "evidence_text": "We evaluate our model on the CRAWDAD dartmouth/campus WiFi trace."
    }}
  ]
}}

If the paper uses NO named datasets and introduces none, return {{"datasets": []}}.
Prefer returning fewer, high-confidence entries over many low-quality ones.

Return ONLY valid JSON — no markdown, no explanation outside the JSON.
"""


def _extraction_cache_key(paper_id: int, text_hash: str) -> str:
    digest = hashlib.sha256(f"dataset_extract:v1:{paper_id}:{text_hash}".encode()).hexdigest()
    return f"de:v1:{digest}"


class DatasetExtractor:
    """Extract dataset records from a paper using its PDF text and an LLM."""

    def __init__(self, router: Any, cache: Any | None = None, conn: Any | None = None) -> None:
        self.router = router
        self.cache = cache
        self.conn = conn

    def extract(
        self,
        paper_id: int,
        title: str,
        authors: str,
        venue: str,
        year: int,
        doi: str,
        pdf_url: str | None,
        abstract: str | None,
    ) -> DatasetExtractionResult:
        from wireless_taxonomy.llm import LlmRequest

        bibtex_key = _make_bibtex_key(authors, year, title)
        crossref_bibtex = _fetch_crossref_bibtex(doi)
        if crossref_bibtex:
            bibtex = re.sub(r"(@\w+\{)[^,]+,", rf"\g<1>{bibtex_key},", crossref_bibtex, count=1)
        else:
            bibtex = _make_minimal_bibtex(bibtex_key, title, authors, year, venue, doi)

        pdf_bytes: bytes | None = None
        extraction_source = "abstract"
        text_section = ""

        if pdf_url and "dl.acm.org" not in pdf_url:
            # Check DB text cache before hitting the network
            pdf_bytes = self._load_cached_pdf(paper_id, pdf_url)
            if not pdf_bytes:
                pdf_bytes = _fetch_pdf_bytes(pdf_url)
                if pdf_bytes:
                    self._store_cached_pdf(paper_id, pdf_url, pdf_bytes)
            if pdf_bytes:
                extraction_source = "pdf"

        if not pdf_bytes:
            fallback_text = abstract or f"Title: {title}\nAuthors: {authors}"
            extraction_source = "abstract" if abstract else "title_only"
            text_section = f"\nPaper text (abstract only — full text unavailable):\n---\n{fallback_text[:8000]}\n---\n"

        content_hash = hashlib.sha256((pdf_bytes or (abstract or title).encode())).hexdigest()[:16]
        cache_key = _extraction_cache_key(paper_id, content_hash)

        if self.cache is not None:
            cached = self.cache.get_llm(cache_key)
            if cached is not None:
                return self._from_cache(cached, paper_id, title, authors, venue, year, doi, bibtex_key, bibtex, extraction_source)

        prompt = _EXTRACTION_PROMPT_TMPL.format(
            title=title,
            authors=authors,
            venue=venue,
            year=year,
            doi=doi or "unknown",
            text_section=text_section,
        )

        try:
            response = self.router.complete(
                LlmRequest(
                    task="dataset_extraction",
                    schema_name="DatasetExtraction",
                    prompt=prompt,
                    metadata={"paper_id": paper_id, "title": title},
                    pdf_bytes=pdf_bytes,
                )
            )
            parsed = response.parsed
            if not isinstance(parsed, dict):
                raise ValueError(f"LLM returned non-dict: {response.content[:200]}")
        except Exception as exc:
            return DatasetExtractionResult(
                paper_id=paper_id, title=title, authors=authors, venue=venue, year=year,
                doi=doi, bibtex_key=bibtex_key, bibtex=bibtex,
                datasets=[], extraction_source=extraction_source, error=str(exc),
            )

        datasets = _parse_dataset_records(parsed.get("datasets") or [])

        # Verify availability URLs the LLM found in the paper text via live HTTP check.
        # Paper-stated availability is ground truth; live check upgrades null -> bool.
        for ds in datasets:
            if ds.availability_url:
                ds.availability = _check_url_live(ds.availability_url)
            elif ds.availability is None and ds.availability_notes:
                url_match = re.search(r'https?://\S+', ds.availability_notes)
                if url_match:
                    ds.availability_url = url_match.group(0).rstrip('.,)')
                    ds.availability = _check_url_live(ds.availability_url)

        if self.cache is not None:
            self.cache.set_llm(cache_key, {"datasets": [_record_to_dict(d) for d in datasets], "source": extraction_source})

        return DatasetExtractionResult(
            paper_id=paper_id, title=title, authors=authors, venue=venue, year=year,
            doi=doi, bibtex_key=bibtex_key, bibtex=bibtex,
            datasets=datasets, extraction_source=extraction_source,
        )

    def _load_cached_pdf(self, paper_id: int, pdf_url: str) -> bytes | None:
        return load_cached_pdf(self.conn, paper_id, pdf_url)

    def _store_cached_pdf(self, paper_id: int, pdf_url: str, pdf_bytes: bytes) -> None:
        store_cached_pdf(self.conn, paper_id, pdf_url, pdf_bytes)

    def _from_cache(
        self, cached: dict, paper_id: int, title: str, authors: str,
        venue: str, year: int, doi: str, bibtex_key: str, bibtex: str, extraction_source: str,
    ) -> DatasetExtractionResult:
        datasets = _parse_dataset_records(cached.get("datasets") or [])
        return DatasetExtractionResult(
            paper_id=paper_id, title=title, authors=authors, venue=venue, year=year,
            doi=doi, bibtex_key=bibtex_key, bibtex=bibtex,
            datasets=datasets, extraction_source=cached.get("source", extraction_source),
        )


_GARBAGE_PREFIXES = (
    "figure ", "fig.", "fig ", "table ", "our ", "the ", "this ", "same ",
    "a ", "an ", "some ", "collected ", "recorded ", "running ",
)
_GARBAGE_EXACT = frozenset({
    "data", "dataset", "trace", "traces", "log", "logs", "test", "testbed",
    "pcap", "pings", "artifact", "training set", "speed test", "testing dataset",
    "training sets", "collected traces", "real-world traces", "performance tests",
    "open-source code and data",
})
MIN_DATASET_CONFIDENCE = 0.50


def _is_garbage_name(name: str) -> bool:
    """Heuristic filter for names that are descriptions, not proper dataset names."""
    lower = name.lower().strip()
    if lower.startswith(_GARBAGE_PREFIXES):
        return True
    if lower in _GARBAGE_EXACT:
        return True
    # All-lowercase + very long → likely a description, not a proper name
    if name == name.lower() and len(name) > 60:
        return True
    # Starts with a digit + mostly descriptive (e.g. "500,000 User Equipment")
    if re.match(r"^\d[\d,.]+ ", name) and len(name) > 20:
        return True
    return False


def _parse_dataset_records(raw: list[Any]) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        confidence = float(item.get("confidence") or 0.0)
        if confidence < MIN_DATASET_CONFIDENCE:
            continue
        if _is_garbage_name(name):
            continue
        rel = str(item.get("relationship_type") or "unclear").lower()
        if rel not in RELATIONSHIP_TYPES:
            rel = "unclear"
        modalities = [str(m).strip() for m in (item.get("modalities") or []) if str(m).strip()]
        osi_raw = [str(o).strip().upper() for o in (item.get("osi_layers") or [])]
        osi = [o for o in osi_raw if o in OSI_LAYERS]
        avail_raw = item.get("availability")
        availability = bool(avail_raw) if avail_raw is not None else None
        avail_notes = str(item.get("availability_notes") or "").strip()
        avail_url = str(item.get("availability_url") or "").strip()
        env = str(item.get("collection_environment") or "").strip()
        if env not in COLLECTION_ENVS:
            env = "Real World Deployment"
        known_users = [str(u).strip() for u in (item.get("known_users") or []) if str(u).strip()][:5]
        evidence = str(item.get("evidence_text") or "").strip()
        records.append(DatasetRecord(
            name=name, relationship_type=rel, modalities=modalities, osi_layers=osi,
            availability=availability, availability_notes=avail_notes,
            availability_url=avail_url, collection_environment=env,
            known_users=known_users, confidence=confidence, evidence_text=evidence,
        ))
    return records


def _record_to_dict(r: DatasetRecord) -> dict[str, Any]:
    return {
        "name": r.name,
        "relationship_type": r.relationship_type,
        "modalities": r.modalities,
        "osi_layers": r.osi_layers,
        "availability": r.availability,
        "availability_url": r.availability_url,
        "availability_notes": r.availability_notes,
        "collection_environment": r.collection_environment,
        "known_users": r.known_users,
        "confidence": r.confidence,
        "evidence_text": r.evidence_text,
    }
