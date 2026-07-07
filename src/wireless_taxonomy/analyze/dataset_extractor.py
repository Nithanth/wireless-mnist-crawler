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


import contextlib
import hashlib
import io as _io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

OSI_LAYERS = {"L1", "L2", "L3", "L4", "L5", "L6", "L7"}
COLLECTION_ENVS = {"Physical Lab Testbed", "Real World Deployment", "Simulation"}

# ── Possible-non-wireless audit signal ───────────────────────────────────────
# The prompt tells the LLM to skip auxiliary non-wireless datasets (census,
# weather, social media, etc.).  As a lightweight backstop we flag records
# that look suspicious — L7-only OSI layers AND dataset name contains no
# wireless keyword — into the per-run audit log for human review.
# We never hard-drop on this signal: the LLM may assign L7 to entirely
# legitimate wireless datasets (HTTP/QUIC performance, DNS, streaming QoE).
_WIRELESS_NAME_RE = re.compile(
    r"\b(wireless|cellular|wifi|wi-fi|5g|4g|lte|nr\b|lora|bluetooth|ble|"
    r"satellite|spectrum|rf\b|signal|channel|network|packet|trace|throughput|"
    r"latency|rtt|rssi|csi|snr|pcap|measurement|deployment|mmwave|radar|"
    r"backscatter|mimo|antenna|basestation|handover|mobility)\b",
    re.IGNORECASE,
)


def _looks_non_wireless(name: str, osi_layers: list[str]) -> bool:
    """Return True if the record is suspicious: all OSI layers are L7 (or
    none assigned) AND the dataset name contains no wireless keyword.
    Used only for audit flagging — never for hard filtering.
    """
    if osi_layers and set(osi_layers) != {"L7"}:
        return False  # has sub-L7 layers → almost certainly wireless
    return not bool(_WIRELESS_NAME_RE.search(name))


# ── Abstract-only extraction pre-filter ──────────────────────────────────────
# When we only have abstract+title (no PDF), we require the abstract to contain
# at least one dataset-indicative phrase before spending an LLM call.  This
# blocks the dominant hallucination pattern where the model invents a dataset
# name from generic phrases like "we evaluated using data from an operator."
#
# Two tiers:
#   STRONG — explicit dataset language; almost certainly worth an LLM call.
#   WEAK   — data collection described but no explicit name; skip unless STRONG
#            also matches or a proper-noun dataset name is visible.
#
# Papers that pass neither tier get extraction_source="skipped_abstract_no_dataset"
# and return an empty dataset list with zero LLM cost.
# ── Tier 1: Named dataset signal ─────────────────────────────────────────────
# The abstract explicitly names a dataset/corpus/benchmark — justifies a full
# LLM extraction call (name + availability + metadata are all extractable).
_ABSTRACT_NAMED_DATASET_RE = re.compile(
    r"""
    \b(
        datasets?  |  corpora?  |  benchmarks?  |
        # Data release / open access
        publicly[\s-]available  |  openly[\s-]available  |  open[\s-]access[\s-]data  |
        we[\s]release  |  we[\s]open.source  |
        data[\s](?:is[\s])?available[\s]at  |  available[\s]at[\s]https?  |
        # Explicit "the X dataset/corpus/benchmark" phrasing
        the[\s]\w+[\s](?:dataset|corpus|benchmark|traces|measurements)  |
        (?:evaluated?|tested?)[\s]on[\s]the[\s]\w+  |
        using[\s]the[\s]\w+[\s](?:dataset|corpus|benchmark)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Tier 2: Collection signal ─────────────────────────────────────────────────
# The abstract describes real data collection but doesn't name a dataset.
# Papers in this tier likely have unnamed/proprietary data — the PDF (if ever
# available) is the right extraction source.  Skip the LLM for now; return
# extraction_source="skipped_abstract_collection_only" so the paper is
# retried automatically when a PDF becomes available.
_ABSTRACT_COLLECTION_RE = re.compile(
    r"""
    \b(
        measurement[\s]campaign  |  field[\s]measurement  |  field[\s]trial  |
        field[\s]study  |  field[\s]experiment  |
        real[\s-]world[\s](?:trace|measurement|deployment|experiment)  |
        packet[\s]trace  |  network[\s]trace  |  traffic[\s]trace  |
        passively[\s]collect  |  passive[\s]measurement  |  passive[\s]monitor  |
        we[\s]collect  |  we[\s]gathered  |  we[\s]captured  |  we[\s]recorded  |
        data[\s]collect  |  traces[\s]collect  |  logs[\s]collect  |
        we[\s]fabricate  |  (?:is[\s])?prototype[d]?  |  we[\s]prototype  |
        validate[\s].*(?:via|through|using)[\s].*experiment  |
        annotation[\s]of[\s]data  |  annotated[\s]data  |  labeled[\s]data  |
        testbed  |  test-bed  |
        channel[\s]measurement  |  channel[\s]sounding  |
        rf[\s]measurement  |  spectrum[\s]measurement  |
        city[\s-]scale[\s]measurement  |  large[\s-]scale[\s]measurement  |
        (?:million|thousand)[\s]users  |
        design[,\s]+implementation[,\s]+and[\s]evaluation
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def abstract_dataset_tier(abstract: str) -> str:
    """Classify the abstract into one of three tiers:

    - ``"named"``      — explicit dataset name present → full LLM extraction
    - ``"collection"`` — describes data collection but no dataset name →
                         skip LLM (retry when PDF available)
    - ``"none"``       — no data signal at all → skip

    The distinction prevents the model from inventing dataset names for papers
    that only say "we evaluated using data from an operator" (no public name).
    """
    if not abstract:
        return "none"
    if _ABSTRACT_NAMED_DATASET_RE.search(abstract):
        return "named"
    if _ABSTRACT_COLLECTION_RE.search(abstract):
        return "collection"
    return "none"


RELATIONSHIP_TYPES = {"introduced", "reused", "extended", "compared_against", "unclear"}


CONFIDENCE_LEVELS = ("high", "medium", "low")


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
    confidence: str  # "high", "medium", or "low"
    evidence_text: str
    grounded: bool | None = None  # None = not checked, True/False = checked


@dataclass
class DroppedRecord:
    """A dataset record that was filtered out, with reason."""
    name: str
    reason: str  # e.g. "low_confidence", "garbage_name", "dedup_merged"
    raw: dict


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
    model_version: str = ""
    dropped: list[DroppedRecord] | None = None


_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _title_words_in_text(title: str, text: str, threshold: float = 0.6) -> bool:
    """True if enough distinctive title words appear in the text.

    Word-level matching (rather than exact substring) tolerates hyphenation,
    line breaks, and ligature artifacts in PDF-extracted text.
    """
    norm = re.compile(r"[^a-z0-9]+")
    text_words = set(norm.sub(" ", text.lower()).split())
    title_words = [w for w in norm.sub(" ", title.lower()).split() if len(w) > 3]
    if not title_words:
        return True
    hits = sum(1 for w in title_words if w in text_words)
    return hits / len(title_words) >= threshold


@contextlib.contextmanager
def _quiet_pypdf():
    """Suppress pypdf's chatty stderr warnings about malformed PDFs.

    pypdf writes directly to stderr (not via Python's warnings module) when it
    encounters common PDF quirks from IEEE/ACM submission systems — duplicate
    font tables, wrong cross-reference offsets, etc.  The warnings are harmless;
    pypdf still extracts text correctly.  We redirect stderr to /dev/null only
    for the duration of PdfReader construction so the pipeline output stays clean.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)  # save real stderr fd
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)  # restore stderr
        os.close(saved_fd)
        os.close(devnull_fd)


def _pdf_matches_title(pdf_bytes: bytes, expected_title: str) -> bool:
    """Verify fetched PDF is the expected paper by checking its first pages.

    Guards against wrong-paper matches from title-search OA providers. Errs on
    the side of acceptance: scanned PDFs (no text layer) or extraction failures
    pass, since a false rejection loses full text for the whole paper.
    """
    try:
        from pypdf import PdfReader

        with _quiet_pypdf():
            reader = PdfReader(_io.BytesIO(pdf_bytes))
            text = " ".join((page.extract_text() or "") for page in reader.pages[:2])
    except Exception:
        return True
    if len(text.strip()) < 200:
        return True  # no usable text layer — can't verify, don't reject
    return _title_words_in_text(expected_title, text)


# 15 MB raw ≈ 20 MB base64 — fits Gemini's inline-data limit (20 MB) with the
# prompt sent separately in the request. Larger files are rejected rather than
# truncated.
_MAX_PDF_BYTES = 1024 * 1024 * 15


def _fetch_pdf_bytes(
    pdf_url: str,
    max_bytes: int = _MAX_PDF_BYTES,
    expected_title: str | None = None,
    attempts: int = 3,
) -> bytes | None:
    """Download a PDF and return raw bytes for native LLM attachment.

    Robustness guarantees:
    - rewrites known landing-page URLs (HAL, etc.) to direct PDF links;
    - retries transient network errors with backoff;
    - rejects files larger than ``max_bytes`` instead of returning silently
      truncated (corrupt) bytes;
    - when ``expected_title`` is given, rejects PDFs whose first pages don't
      contain the title (wrong-paper guard for title-search OA providers).
    - hard wall-clock deadline of 30s total per attempt — prevents silent stalls
      where a server accepts the TCP connection but never (or very slowly) sends
      data, which defeats socket-level timeouts.
    """
    # Rewrite landing-page URLs to direct PDF links where possible.
    pdf_url = _rewrite_landing_page_url(pdf_url) or ""
    if not pdf_url:
        return None

    # Wall-clock deadline per attempt — prevents silent stalls where a server
    # accepts the TCP connection but never (or very slowly) sends data, which
    # defeats socket-level timeouts.  Uses a daemon thread + Future so it works
    # correctly from both the main thread and worker threads (SIGALRM is
    # main-thread-only and would silently break parallel prefetch workers).
    import concurrent.futures as _cf

    _TOTAL_TIMEOUT = 30  # seconds per attempt

    def _do_download(url: str, mb: int) -> bytes:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": _BROWSER_UA, "Accept": "application/pdf,*/*"},
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read(mb + 1)

    for attempt in range(attempts):
        try:
            # Do NOT use the executor as a context manager — its __exit__
            # blocks until all submitted threads finish, so a hung r.read()
            # would make the timeout meaningless.  Instead use a module-level
            # executor and just abandon the future; the daemon thread will
            # eventually die on its own (or when the process exits).
            _ex = _cf.ThreadPoolExecutor(max_workers=1)
            fut = _ex.submit(_do_download, pdf_url, max_bytes)
            _ex.shutdown(wait=False)  # don't block; let daemon thread run free
            try:
                raw = fut.result(timeout=_TOTAL_TIMEOUT)
            except _cf.TimeoutError:
                fut.cancel()
                raise TimeoutError("PDF download timed out")
        except urllib.error.HTTPError as exc:
            if exc.code in (408, 429, 500, 502, 503, 504) and attempt < attempts - 1:
                time.sleep(1.5 * (2 ** attempt))
                continue
            return None
        except Exception:
            if attempt < attempts - 1:
                time.sleep(1.5 * (2 ** attempt))
                continue
            return None
        if raw[:4] != b"%PDF":
            return None
        if len(raw) > max_bytes:
            return None  # oversized — a truncated PDF is corrupt, don't pass it on
        if expected_title and not _pdf_matches_title(raw, expected_title):
            return None
        return raw
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


def _is_acm_blocked(url: str) -> bool:
    """Return True for URLs that resolve to ACM's paywalled PDFs.

    OpenAlex sometimes reports ``https://doi.org/10.1145/...`` as an OA PDF
    URL, but those DOIs redirect to dl.acm.org, which blocks programmatic
    downloads. Treating them as blocked avoids wasting time fetching HTML
    landing pages and keeps the extraction_source honest.
    """
    return (
        "dl.acm.org" in url
        or "doi.org/10.1145" in url
        or "doi.org/10.1109/" in url
        or "ieeexplore.ieee.org" in url
    )


def _rewrite_landing_page_url(url: str) -> str | None:
    """Attempt to convert a known landing-page URL to a direct PDF URL.

    Several OA providers store record/landing page URLs rather than direct
    PDF links. This function recognises the patterns and rewrites them.
    Returns the rewritten URL, or None if the URL should be skipped entirely
    (e.g. IEEE paywalled DOIs).

    Patterns handled:
    - hal.science/hal-XXXXX           → hal.science/hal-XXXXX/document
      (HAL serves the PDF when Accept: application/pdf is sent; the /document
      path also triggers content-negotiation more reliably)
    - *.handle.net/handle/...         → try /bitstream/... discovery via HTML
      DSpace handles are landing pages; we skip them (too complex to reliably
      extract bitstream URLs without scraping).
    - figshare.com/articles/*/NNN     → figshare API download URL
    - doi.org/10.1109/...             → ieeexplore.ieee.org (paywalled) → skip
    - doi.org/10.1145/...             → already caught by _is_acm_blocked
    """
    import re as _re
    # HAL: https://hal.science/hal-XXXXXXX  (no extension → landing page)
    hal_m = _re.match(r"^(https://hal\.science/hal-\d+)/?$", url)
    if hal_m:
        return hal_m.group(1) + "/document"
    # HAL archives-ouvertes.fr (legacy)
    hal_old = _re.match(r"^(https://(?:hal|tel|halshs|medihal)\.archives-ouvertes\.fr/[^/]+)/?$", url)
    if hal_old:
        return hal_old.group(1) + "/document"
    # doi.org resolving to IEEE (paywalled) — skip
    if "doi.org/10.1109/" in url or "ieeexplore.ieee.org" in url:
        return None
    # Handle.net resolver and DSpace repos — always serve HTML landing pages
    # and are extremely slow; no reliable way to extract the bitstream URL
    # without scraping. Skip entirely to avoid 90s stall per paper.
    if "handle.net" in url or "/handle/" in url:
        return None
    # Bare doi.org links (not ACM/IEEE) — let the downloader try; it may
    # redirect to an open repo.  Return unchanged.
    return url


def _fetch_crossref_bibtex(doi: str, attempts: int = 3) -> str | None:
    """Retrieve BibTeX from CrossRef for a given DOI.

    Retries with exponential backoff on rate-limit/transient errors so that
    concurrent workers degrade gracefully instead of silently falling back to
    minimal BibTeX when doi.org throttles.
    """
    if not doi:
        return None
    url = f"https://doi.org/{urllib.parse.quote(doi, safe='/')}"
    for attempt in range(attempts):
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/x-bibtex", "User-Agent": "wireless-taxonomy/0.1"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                text = r.read().decode("utf-8", errors="replace")
            if text.strip().startswith("@"):
                return text.strip()
            return None
        except urllib.error.HTTPError as exc:
            if exc.code in (408, 429, 500, 502, 503, 504) and attempt < attempts - 1:
                time.sleep(1.5 * (2 ** attempt))
                continue
            return None
        except Exception:
            if attempt < attempts - 1:
                time.sleep(1.5 * (2 ** attempt))
                continue
            return None
    return None


def has_cached_pdf(conn, paper_id: int, pdf_url: str) -> bool:
    """Check if a PDF is cached without loading the full blob."""
    if conn is None:
        return False
    try:
        row = conn.execute(
            "SELECT 1 FROM paper_text_artifacts "
            "WHERE paper_id = ? AND source_url = ? AND fetch_status = 'ok' LIMIT 1",
            (paper_id, pdf_url),
        ).fetchone()
        return row is not None
    except Exception:
        return False


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


def load_cached_pdf_text(conn, paper_id: int, pdf_url: str) -> str | None:
    """Return pre-extracted PDF text from paper_text_artifacts if cached.

    Cheaper than load_cached_pdf: stores plain text (~50KB) instead of
    base64-encoded PDF bytes (~10MB), and avoids a second pypdf extraction
    pass at classification/extraction time.
    """
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT content_text, source_type FROM paper_text_artifacts "
            "WHERE paper_id = ? AND source_url = ? AND fetch_status = 'ok' LIMIT 1",
            (paper_id, pdf_url),
        ).fetchone()
        if row and row["content_text"] and row["source_type"] == "pdf_text":
            return row["content_text"]
    except Exception:
        pass
    return None


def store_cached_pdf_text(conn, paper_id: int, pdf_url: str, pdf_text: str) -> None:
    """Persist pre-extracted PDF plain text into paper_text_artifacts.

    Preferred over store_cached_pdf for the prefetch stage: stores ~50KB of
    text instead of ~10MB of base64 bytes, and makes re-runs instant because
    text is ready without re-downloading or re-extracting the PDF.
    """
    if conn is None:
        return
    if not pdf_text or not pdf_text.strip():
        return
    try:
        sha = hashlib.sha256(pdf_text.encode("utf-8", errors="replace")).hexdigest()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT OR REPLACE INTO paper_text_artifacts
              (paper_id, source_type, source_url, fetch_status,
               content_text, content_sha256, fetched_at, created_at)
            VALUES (?, 'pdf_text', ?, 'ok', ?, ?, ?, ?)
            """,
            (paper_id, pdf_url, pdf_text, sha, now, now),
        )
    except Exception as exc:
        import sys
        print(f"  [!] failed to cache PDF text for paper {paper_id}: {exc}", file=sys.stderr)


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
    except Exception as exc:
        import sys
        print(f"  [!] failed to cache PDF for paper {paper_id}: {exc}", file=sys.stderr)


def store_cached_pdf_failure(conn, paper_id: int, pdf_url: str, error_message: str = "") -> None:
    """Record a download failure in paper_text_artifacts (fetch_status='failed').

    On re-runs, papers with a recorded failure are skipped automatically unless
    --retry-failed is passed. This means a fixed download bug can be surfaced
    selectively without re-attempting every paper in the corpus.
    """
    if conn is None:
        return
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT OR REPLACE INTO paper_text_artifacts
              (paper_id, source_type, source_url, fetch_status,
               content_text, content_sha256, error_message, fetched_at, created_at)
            VALUES (?, 'pdf_b64', ?, 'failed', '', '', ?, ?, ?)
            """,
            (paper_id, pdf_url, error_message[:500], now, now),
        )
    except Exception as exc:
        import sys
        print(f"  [!] failed to record PDF failure for paper {paper_id}: {exc}", file=sys.stderr)


def load_cached_pdf_failed(conn, paper_id: int, pdf_url: str) -> bool:
    """Return True if a previous download attempt for this paper+url was recorded as failed.

    Used by the prefetch stage to skip known-bad URLs unless --retry-failed
    clears the failure record first.
    """
    if conn is None:
        return False
    try:
        row = conn.execute(
            "SELECT 1 FROM paper_text_artifacts "
            "WHERE paper_id = ? AND source_url = ? AND fetch_status = 'failed' LIMIT 1",
            (paper_id, pdf_url),
        ).fetchone()
        return row is not None
    except Exception:
        return False


def clear_cached_pdf_failures(conn, paper_id: int, pdf_url: str) -> None:
    """Remove failure records for a paper+url so the next prefetch retries it.

    Called when --retry-failed is passed, scoped to papers whose download
    previously failed. Does not touch successful downloads or OA resolution.
    """
    if conn is None:
        return
    try:
        conn.execute(
            "DELETE FROM paper_text_artifacts WHERE paper_id = ? AND source_url = ? AND fetch_status = 'failed'",
            (paper_id, pdf_url),
        )
    except Exception as exc:
        import sys
        print(f"  [!] failed to clear PDF failure for paper {paper_id}: {exc}", file=sys.stderr)


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
      "confidence": "high",
      "evidence_text": "We collected 5G NR traces across 12 routes in NYC over 3 months."
    }},
    {{
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
      "confidence": "high",
      "evidence_text": "We evaluate our model on the CRAWDAD dartmouth/campus WiFi trace."
    }}
  ]
}}

If the paper uses NO datasets and introduces none, return {{"datasets": []}}.
Prefer quality over quantity — only extract datasets you are reasonably sure about.

Return ONLY valid JSON — no markdown, no explanation outside the JSON.
"""


def _extraction_cache_key(paper_id: int, text_hash: str, model_identity: str = "") -> str:
    """Content-addressed extraction key, scoped to the model that produced it.

    Two components ensure cache correctness:
    - ``text_hash``      — the paper content (PDF bytes or abstract text)
    - ``model_identity`` — provider/model chain; switching models forces
      a fresh extraction rather than serving a different model's output.

    Prompt changes do NOT automatically invalidate the cache — this is
    intentional.  Wholesale prompt invalidation would re-extract every
    paper ($1-3+ per prompt edit).  Instead, use the admin purge-cache
    command to surgically clear entries containing specific dataset names
    when a prompt change is known to affect a small subset of papers.

    Passing model_identity="" yields the legacy v1 key for cache migration.
    """
    if model_identity:
        digest = hashlib.sha256(
            f"dataset_extract:v2:{model_identity}:{paper_id}:{text_hash}".encode()
        ).hexdigest()
        return f"de:v2:{digest}"
    digest = hashlib.sha256(f"dataset_extract:v1:{paper_id}:{text_hash}".encode()).hexdigest()
    return f"de:v1:{digest}"


class DatasetExtractor:
    """Extract dataset records from a paper using its PDF text and an LLM."""

    def __init__(self, router: Any, cache: Any | None = None, conn: Any | None = None) -> None:
        self.router = router
        self.cache = cache
        self.conn = conn

    def _model_identity(self) -> str:
        """Stable identifier for the primary provider/model only.

        Part of the extraction cache key so results are reused only while the
        model that produced them is unchanged; swapping the primary model
        re-runs extraction (clean control experiments). Fallbacks are a
        resilience mechanism and do NOT change the experiment identity.
        """
        try:
            provider = self.router.select_provider()
            return f"{provider.provider}/{provider.model}"
        except Exception:
            return ""

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
        pdf_bytes: bytes | None = None,
        pdf_text: str | None = None,
        refresh: bool = False,
    ) -> DatasetExtractionResult:
        """Extract dataset records for one paper.

        ``pdf_bytes`` may be supplied by the caller (e.g. pre-fetched by the
        pipeline) to send the PDF natively to the LLM (best quality — tables
        and figures are preserved).  ``pdf_text`` may be supplied instead when
        only pre-extracted plain text is available (good quality — prose and
        captions intact, but tables/figures lost).  If neither is provided the
        extractor falls back to the DB cache, then the network, then abstract.
        ``refresh`` skips the cache read (but still writes the fresh result).
        """
        from wireless_taxonomy.llm import LlmRequest

        bibtex_key = _make_bibtex_key(authors, year, title)
        crossref_bibtex = self._cached_crossref_bibtex(doi)
        if crossref_bibtex:
            bibtex = re.sub(r"(@\w+\{)[^,]+,", rf"\g<1>{bibtex_key},", crossref_bibtex, count=1)
        else:
            bibtex = _make_minimal_bibtex(bibtex_key, title, authors, year, venue, doi)

        extraction_source = "abstract"
        text_section = ""

        if pdf_bytes:
            extraction_source = "pdf"
        elif pdf_text:
            # Pre-extracted plain text — good quality but no tables/figures.
            extraction_source = "pdf_text"
        elif pdf_url and not _is_acm_blocked(pdf_url):
            # Check DB cache before hitting the network — prefer raw bytes
            # (native LLM attachment) but accept pre-extracted text if that's
            # what was stored during prefetch.
            pdf_bytes = self._load_cached_pdf(paper_id, pdf_url)
            if not pdf_bytes:
                pdf_text = self._load_cached_pdf_text(paper_id, pdf_url)
            if not pdf_bytes and not pdf_text:
                pdf_bytes = _fetch_pdf_bytes(pdf_url, expected_title=title)
                if pdf_bytes:
                    self._store_cached_pdf(paper_id, pdf_url, pdf_bytes)
            if pdf_bytes:
                extraction_source = "pdf"
            elif pdf_text:
                extraction_source = "pdf_text"

        if not pdf_bytes and not pdf_text and not (abstract or "").strip():
            # No full text and no abstract: extraction would be pure guesswork.
            return DatasetExtractionResult(
                paper_id=paper_id, title=title, authors=authors, venue=venue, year=year,
                doi=doi, bibtex_key=bibtex_key, bibtex=bibtex,
                datasets=[], extraction_source="skipped_no_text",
            )

        if not pdf_bytes and not pdf_text:
            # ── Abstract-only two-tier pre-filter ─────────────────────────
            # Tier "none": no dataset language at all → skip, no LLM call.
            # Tier "collection": describes data collection but no named
            #   dataset → the data is private/unnamed; skip and wait for PDF.
            # Tier "named": abstract explicitly names a dataset/corpus →
            #   proceed with a strict LLM call (name + availability only).
            #
            # This eliminates hallucinated dataset names for papers that only
            # say "we evaluated using data from an operator" — the dominant
            # source of fabricated entries in abstract-only extraction.
            tier = abstract_dataset_tier(abstract or "")
            if tier == "none":
                return DatasetExtractionResult(
                    paper_id=paper_id, title=title, authors=authors, venue=venue, year=year,
                    doi=doi, bibtex_key=bibtex_key, bibtex=bibtex,
                    datasets=[], extraction_source="skipped_abstract_no_dataset",
                )
            if tier == "collection":
                # Data collection is described but no dataset is named.
                # Skip the LLM — unnamed/proprietary data cannot be reliably
                # extracted from an abstract without inventing a name.
                # The paper will be retried automatically if a PDF becomes available.
                return DatasetExtractionResult(
                    paper_id=paper_id, title=title, authors=authors, venue=venue, year=year,
                    doi=doi, bibtex_key=bibtex_key, bibtex=bibtex,
                    datasets=[], extraction_source="skipped_abstract_collection_only",
                )
            # tier == "named": abstract explicitly names a dataset.
            extraction_source = "abstract"
            text_section = (
                f"\nPaper text (abstract only — full text unavailable):\n---\n{abstract[:8000]}\n---\n"
                "\nSTRICT ABSTRACT MODE: you only have the abstract, not the full paper.\n"
                "Rules:\n"
                "1. Extract a dataset ONLY if the abstract contains its EXPLICIT NAME — "
                "a proper noun like 'DeepSense 6G', 'MNIST', 'Ookla Speedtest', etc.\n"
                "2. If the abstract only says 'we collected data', 'we measured', or "
                "'we used data from an operator/provider' WITHOUT naming the dataset, "
                "return {\"datasets\": []}. Do NOT invent a name.\n"
                "3. Do NOT infer datasets from paper topic or domain knowledge. Only "
                "extract what is explicitly named in the abstract text above.\n"
                "4. For each named dataset: fill in availability (Y/N + URL if stated). "
                "Leave OSI layers, modalities, and collection environment blank — they "
                "will be filled from the full PDF if available.\n"
                "5. If in doubt, return {\"datasets\": []}.\n"
            )
        elif pdf_text and not pdf_bytes:
            # Pre-extracted text: prose and captions intact; tables/figures lost.
            text_section = f"\nPaper text (extracted from PDF):\n---\n{pdf_text[:120_000]}\n---\n"

        content_hash = hashlib.sha256(
            pdf_bytes or (pdf_text or abstract or title).encode()
        ).hexdigest()[:16]
        model_identity = self._model_identity()
        cache_key = _extraction_cache_key(paper_id, content_hash, model_identity)

        if self.cache is not None and not refresh:
            cached = self.cache.get_llm(cache_key)
            if cached is None and model_identity:
                # Migration 1: pre-model-scoped (v1) entries had no model in
                # the key at all — adopt them under the new key once.
                legacy = self.cache.get_llm(_extraction_cache_key(paper_id, content_hash))
                if legacy is not None:
                    self.cache.set_llm(cache_key, legacy)
                    cached = legacy
            if cached is None and model_identity:
                # Migration 2: old full-chain format included every provider
                # with an API key (e.g. "google/flash,openai/...,anthropic/...").
                # Reconstruct that chain from ALL configured keys (not the
                # fallback setting, which may have changed) and adopt the
                # legacy entry under the new primary-only key.
                try:
                    all_providers = self.router.all_configured_providers()
                    chain_id = ",".join(f"{p.provider}/{p.model}" for p in all_providers)
                    if chain_id and chain_id != model_identity:
                        chain_key = _extraction_cache_key(paper_id, content_hash, chain_id)
                        legacy2 = self.cache.get_llm(chain_key)
                        if legacy2 is not None:
                            self.cache.set_llm(cache_key, legacy2)
                            cached = legacy2
                except Exception:
                    pass
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
            # When PDF-as-bytes fails (e.g. Gemini HTTP 400 content policy),
            # fall back to pypdf text extraction and retry without the PDF
            # attachment. Degraded but much better than returning nothing.
            if pdf_bytes:
                from wireless_taxonomy.llm import _pdf_bytes_to_text
                pdf_text = _pdf_bytes_to_text(pdf_bytes)
                if pdf_text and len(pdf_text) > 200:
                    text_section = f"\nPaper text (extracted from PDF):\n---\n{pdf_text[:120_000]}\n---\n"
                    text_prompt = _EXTRACTION_PROMPT_TMPL.format(
                        title=title, authors=authors, venue=venue, year=year,
                        doi=doi or "unknown", text_section=text_section,
                    )
                    try:
                        response = self.router.complete(
                            LlmRequest(
                                task="dataset_extraction",
                                schema_name="DatasetExtraction",
                                prompt=text_prompt,
                                metadata={"paper_id": paper_id, "title": title},
                            )
                        )
                        parsed = response.parsed
                        if isinstance(parsed, dict):
                            extraction_source = "pdf_text_fallback"
                        else:
                            raise ValueError(f"LLM returned non-dict on text fallback: {response.content[:200]}")
                    except Exception as text_exc:
                        return DatasetExtractionResult(
                            paper_id=paper_id, title=title, authors=authors, venue=venue, year=year,
                            doi=doi, bibtex_key=bibtex_key, bibtex=bibtex,
                            datasets=[], extraction_source=extraction_source,
                            error=f"PDF failed: {exc} | text fallback failed: {text_exc}",
                        )
                else:
                    return DatasetExtractionResult(
                        paper_id=paper_id, title=title, authors=authors, venue=venue, year=year,
                        doi=doi, bibtex_key=bibtex_key, bibtex=bibtex,
                        datasets=[], extraction_source=extraction_source,
                        error=f"PDF failed: {exc} | pypdf extraction too short ({len(pdf_text)} chars)",
                    )
            else:
                return DatasetExtractionResult(
                    paper_id=paper_id, title=title, authors=authors, venue=venue, year=year,
                    doi=doi, bibtex_key=bibtex_key, bibtex=bibtex,
                    datasets=[], extraction_source=extraction_source, error=str(exc),
                )

        datasets, dropped = _parse_dataset_records(parsed.get("datasets") or [])

        # Evidence grounding: verify evidence_text has word overlap with source.
        source_text_for_grounding = ""
        if pdf_bytes:
            try:
                from pypdf import PdfReader
                with _quiet_pypdf():
                    reader = PdfReader(_io.BytesIO(pdf_bytes))
                    source_text_for_grounding = " ".join(
                        (page.extract_text() or "") for page in reader.pages[:15]
                    )[:60000]
            except Exception:
                pass
        if not source_text_for_grounding and abstract:
            source_text_for_grounding = abstract
        if source_text_for_grounding:
            _ground_evidence(datasets, source_text_for_grounding)

        # Verify availability URLs the LLM found in the paper text via live HTTP check.
        # Paper-stated availability is ground truth; live check upgrades null -> bool.
        for ds in datasets:
            if ds.availability_url:
                ds.availability = self._cached_url_live(ds.availability_url)
            elif ds.availability is None and ds.availability_notes:
                url_match = re.search(r'https?://\S+', ds.availability_notes)
                if url_match:
                    ds.availability_url = url_match.group(0).rstrip('.,)')
                    ds.availability = self._cached_url_live(ds.availability_url)

        model_version = f"{response.provider}:{response.model}"
        if self.cache is not None:
            self.cache.set_llm(cache_key, {
                "datasets": [_record_to_dict(d) for d in datasets],
                "source": extraction_source,
                "model_version": model_version,
            })

        return DatasetExtractionResult(
            paper_id=paper_id, title=title, authors=authors, venue=venue, year=year,
            doi=doi, bibtex_key=bibtex_key, bibtex=bibtex,
            datasets=datasets, extraction_source=extraction_source,
            model_version=model_version,
            dropped=dropped or None,
        )

    def _cached_url_live(self, url: str) -> bool:
        """URL liveness check with a 7-day disk cache.

        Availability URLs rarely flip status mid-corpus; caching avoids
        re-hitting the same GitHub/website URL for every paper that shares a
        dataset, and makes re-runs network-free.
        """
        if not url:
            return False
        cache_key = f"urllive:{url.strip().lower()}"
        if self.cache is not None:
            cached = self.cache.get_llm(cache_key)
            if cached is not None and "live" in cached:
                try:
                    checked = datetime.fromisoformat(cached.get("checked_at", ""))
                    if datetime.now(timezone.utc) - checked < timedelta(days=7):
                        return bool(cached["live"])
                except (ValueError, TypeError):
                    pass
        live = _check_url_live(url)
        if self.cache is not None:
            self.cache.set_llm(cache_key, {
                "live": live,
                "checked_at": datetime.now(timezone.utc).isoformat(),
            })
        return live

    def _cached_crossref_bibtex(self, doi: str) -> str | None:
        """CrossRef BibTeX lookup with disk cache (keyed by DOI).

        Successful lookups are cached forever (BibTeX for a DOI is immutable);
        failures are NOT cached so transient doi.org errors stay retryable.
        """
        if not doi:
            return None
        cache_key = f"bibtex:{doi.strip().lower()}"
        if self.cache is not None:
            cached = self.cache.get_llm(cache_key)
            if cached is not None and cached.get("bibtex"):
                return str(cached["bibtex"])
        bibtex = _fetch_crossref_bibtex(doi)
        if bibtex and self.cache is not None:
            self.cache.set_llm(cache_key, {"bibtex": bibtex})
        return bibtex

    def _load_cached_pdf(self, paper_id: int, pdf_url: str) -> bytes | None:
        return load_cached_pdf(self.conn, paper_id, pdf_url)

    def _load_cached_pdf_text(self, paper_id: int, pdf_url: str) -> str | None:
        return load_cached_pdf_text(self.conn, paper_id, pdf_url)

    def _store_cached_pdf(self, paper_id: int, pdf_url: str, pdf_bytes: bytes) -> None:
        store_cached_pdf(self.conn, paper_id, pdf_url, pdf_bytes)

    def _from_cache(
        self, cached: dict, paper_id: int, title: str, authors: str,
        venue: str, year: int, doi: str, bibtex_key: str, bibtex: str, extraction_source: str,
    ) -> DatasetExtractionResult:
        datasets, dropped = _parse_dataset_records(cached.get("datasets") or [])
        return DatasetExtractionResult(
            paper_id=paper_id, title=title, authors=authors, venue=venue, year=year,
            doi=doi, bibtex_key=bibtex_key, bibtex=bibtex,
            datasets=datasets, extraction_source=cached.get("source", extraction_source),
            model_version=cached.get("model_version", "unknown-cached"),
            dropped=dropped or None,
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
# Minimum accepted confidence levels for filtering.
# Both introduced and reused datasets require at least "medium".
_PASS_CONFIDENCE = {"high", "medium"}


def _parse_confidence(raw: Any) -> str:
    """Parse confidence from LLM output — handles both categorical and numeric.

    New extractions produce "high"/"medium"/"low". Old cached results may
    have numeric floats which are mapped: >=0.60 → "medium", >=0.85 → "high".
    """
    if raw is None:
        return "low"
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in CONFIDENCE_LEVELS:
            return s
        # Try parsing as a numeric string from old cache
        try:
            return _numeric_to_categorical(float(s))
        except (TypeError, ValueError):
            return "low"
    if isinstance(raw, (int, float)):
        return _numeric_to_categorical(float(raw))
    return "low"


def _numeric_to_categorical(score: float) -> str:
    """Map legacy numeric confidence to categorical for old cached results."""
    if score >= 0.85:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


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


def _looks_like_authored_name(name: str) -> bool:
    """True if the name follows the AuthorYear-description convention we asked for."""
    return bool(re.match(r"^[A-Z][a-z]+\d{4}-", name))


def _ground_evidence(datasets: list[DatasetRecord], source_text: str) -> None:
    """Check each dataset's evidence_text against the source document.

    Sets ``ds.grounded = True`` if >40% of distinctive evidence words appear
    in the source text, ``False`` otherwise. Skips entries with no evidence.
    This is a cheap word-overlap heuristic (no LLM call, no cost).
    """
    _STOP = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
             "for", "and", "or", "that", "this", "with", "from", "by", "on", "it"}
    source_words = set(re.sub(r"[^a-z0-9]+", " ", source_text.lower()).split())
    for ds in datasets:
        if not ds.evidence_text:
            ds.grounded = None
            continue
        ev_words = set(re.sub(r"[^a-z0-9]+", " ", ds.evidence_text.lower()).split())
        ev_words -= _STOP
        if len(ev_words) < 3:
            ds.grounded = None  # too short to check meaningfully
            continue
        hits = sum(1 for w in ev_words if w in source_words)
        ds.grounded = (hits / len(ev_words)) >= 0.4


def _name_tokens(name: str) -> set[str]:
    """Normalized word tokens for name similarity comparison."""
    return set(re.sub(r"[^a-z0-9]+", " ", name.lower()).split()) - {
        "dataset", "data", "traces", "measurements", "set", "the", "a", "of", "for",
    }


def _name_similarity(a: str, b: str) -> float:
    """Jaccard similarity over normalized name tokens."""
    ta, tb = _name_tokens(a), _name_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _dedup_within_paper(records: list[DatasetRecord]) -> tuple[list[DatasetRecord], list[DroppedRecord]]:
    """Merge near-duplicate datasets extracted from the same paper.

    Two records are merged if they have the same relationship_type and
    >0.7 name similarity. The longer-named (more descriptive) record wins;
    the shorter is logged as a dropped duplicate.
    """
    if len(records) <= 1:
        return records, []
    kept: list[DatasetRecord] = []
    dropped: list[DroppedRecord] = []
    used = [False] * len(records)
    for i, ri in enumerate(records):
        if used[i]:
            continue
        best = ri
        for j in range(i + 1, len(records)):
            if used[j]:
                continue
            rj = records[j]
            if ri.relationship_type != rj.relationship_type:
                continue
            if _name_similarity(ri.name, rj.name) > 0.7:
                used[j] = True
                # Keep the longer/more descriptive name
                if len(rj.name) > len(best.name):
                    dropped.append(DroppedRecord(
                        name=best.name, reason="dedup_merged",
                        raw={"merged_into": rj.name},
                    ))
                    best = rj
                else:
                    dropped.append(DroppedRecord(
                        name=rj.name, reason="dedup_merged",
                        raw={"merged_into": best.name},
                    ))
        kept.append(best)
    return kept, dropped


def _parse_dataset_records(raw: list[Any]) -> tuple[list[DatasetRecord], list[DroppedRecord]]:
    """Parse and validate LLM extraction output into DatasetRecords.

    Returns (accepted_records, dropped_records) so callers can audit what
    was filtered and why.
    """
    records: list[DatasetRecord] = []
    dropped: list[DroppedRecord] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        confidence = _parse_confidence(item.get("confidence"))
        rel = str(item.get("relationship_type") or "unclear").lower()
        if rel not in RELATIONSHIP_TYPES:
            rel = "unclear"

        # Filtering: reject low-confidence datasets. Both introduced and
        # reused require at least "medium" confidence.
        if confidence not in _PASS_CONFIDENCE:
            dropped.append(DroppedRecord(name=name, reason="low_confidence", raw=item))
            continue

        # Name quality filtering depends on relationship type:
        # - "introduced": lenient — paper's own data. Only reject the most
        #   egregious garbage (figure refs, table refs).
        # - everything else ("reused", "extended", etc.): strict — must be a
        #   proper searchable name that can be cross-referenced across papers.
        if rel == "introduced":
            if not _looks_like_authored_name(name):
                lower = name.lower().strip()
                if lower.startswith(("figure ", "fig.", "fig ", "table ")):
                    dropped.append(DroppedRecord(name=name, reason="garbage_name_figure_ref", raw=item))
                    continue
                if lower in _GARBAGE_EXACT:
                    dropped.append(DroppedRecord(name=name, reason="garbage_name_exact", raw=item))
                    continue
        else:
            if _is_garbage_name(name):
                dropped.append(DroppedRecord(name=name, reason="garbage_name", raw=item))
                continue

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

        # Audit flag for possible non-wireless auxiliary datasets.
        # The prompt instructs the LLM to skip these, but as a lightweight
        # backstop we tag suspicious records for human review.  We never
        # hard-drop here — a downstream human reviews the audit log.
        if _looks_non_wireless(name, osi):
            dropped.append(DroppedRecord(
                name=name, reason="possible_non_wireless", raw={**item, "_note": (
                    "L7-only OSI layers and no wireless keyword in dataset name. "
                    "Review audit log to confirm or discard."
                )},
            ))
            # Fall through — record is still accepted; audit entry is advisory only.

        records.append(DatasetRecord(
            name=name, relationship_type=rel, modalities=modalities, osi_layers=osi,
            availability=availability, availability_notes=avail_notes,
            availability_url=avail_url, collection_environment=env,
            known_users=known_users, confidence=confidence, evidence_text=evidence,
        ))

    # Within-paper deduplication
    records, dedup_drops = _dedup_within_paper(records)
    dropped.extend(dedup_drops)

    return records, dropped


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
