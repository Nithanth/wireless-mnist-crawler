from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

from wireless_taxonomy.ingest.base import IngestAdapter
from wireless_taxonomy.models import PaperSeed

FetchJson = Callable[[str], dict[str, Any]]

# venue label (normalized) -> DBLP publication-stream prefix (without the year)
_DBLP_STREAMS = {
    "sigcomm": "conf/sigcomm/sigcomm",
    "imc": "conf/imc/imc",
    "nsdi": "conf/nsdi/nsdi",
    "mobicom": "conf/mobicom/mobicom",
    "mobisys": "conf/mobisys/mobisys",
    "sensys": "conf/sensys/sensys",
    "ipsn": "conf/ipsn/ipsn",
    "infocom": "conf/infocom/infocom",
    "conext": "conf/conext/conext",
    "co-next": "conf/conext/conext",
    "hotnets": "conf/hotnets/hotnets",
}

_DOI_RE = re.compile(r"10\.\d{4,9}/\S+")
_DISAMBIGUATION_RE = re.compile(r"\s+\d{4}$")

# Non-main-track entries DBLP lists alongside full papers. These are short
# poster/demo/workshop/keynote records that aren't peer-reviewed main-track
# papers, so they pollute the proceedings "universe" and inflate false
# positives. They're almost always flagged by a leading title token.
_NON_PAPER_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"poster|demo|demonstration|work[\s-]?in[\s-]?progress|wip|"
    r"tutorial|keynote|panel|invited\s+(?:talk|paper|speaker)|"
    r"extended\s+abstract|abstract|short\s+paper|"
    r"birds[\s-]of[\s-]a[\s-]feather|bof|"
    r"workshop\s+(?:summary|report)|session\s+details|"
    r"front\s+matter|table\s+of\s+contents|proceedings\s+of"
    r")\b\s*[:\-\u2013\u2014]",
    re.IGNORECASE,
)


def is_non_paper_title(title: str) -> bool:
    """True for posters/demos/workshop/keynote records (not main-track papers).

    DBLP TOCs interleave these short non-paper entries with full papers; they
    carry a leading marker like ``Poster:`` / ``Demo:`` / ``Keynote:``. Dropping
    them at ingest keeps the proceedings universe to peer-reviewed main-track
    papers so the eval isn't penalised by non-paper false positives.
    """
    return bool(_NON_PAPER_PREFIX_RE.match(title or ""))


def resolve_stream(venue: str) -> str | None:
    """Return the DBLP stream for a venue, or None if it has no mapping."""
    key = re.sub(r"[^a-z0-9-]", "", venue.strip().lower())
    return _DBLP_STREAMS.get(key)


def stream_for_venue(venue: str) -> str:
    key = re.sub(r"[^a-z0-9-]", "", venue.strip().lower())
    stream = _DBLP_STREAMS.get(key)
    if stream is None:
        raise ValueError(
            f"No DBLP stream mapping for venue {venue!r}. Known venues: "
            f"{', '.join(sorted(_DBLP_STREAMS))}. Pass --bibtex/--csv to ingest it directly."
        )
    return stream


class DblpIngestAdapter(IngestAdapter):
    """Fetches a venue/year main-track paper list from the DBLP publication API.

    DBLP is the authoritative, unblocked enumeration of accepted papers
    (title/authors/DOI). The TOC is paginated (100 hits/page) and throttled with
    a short sleep between pages to stay within DBLP's rate limits.
    """

    source_method = "dblp"

    def __init__(
        self,
        venue: str,
        year: int,
        fetch_json: FetchJson | None = None,
        stream: str | None = None,
        sleep_seconds: float = 3.0,
    ) -> None:
        self.venue = venue
        self.year = year
        self.stream = stream or stream_for_venue(venue)
        self.sleep_seconds = sleep_seconds
        if fetch_json is None:
            from wireless_taxonomy.analyze.abstracts import _default_fetch_json

            fetch_json = _default_fetch_json
        self.fetch_json = fetch_json

    def fetch(self) -> list[PaperSeed]:
        seeds: list[PaperSeed] = []
        for hit in self._fetch_toc():
            info = hit.get("info", {}) if isinstance(hit, dict) else {}
            if not isinstance(info, dict):
                continue
            if str(info.get("type", "")).lower().startswith("editor"):
                continue  # proceedings/front-matter record, not a paper
            title = (info.get("title") or "").rstrip(".").strip()
            if not title:
                continue
            if is_non_paper_title(title):
                continue  # poster/demo/workshop/keynote record, not main-track
            authors = _authors(info)
            doi = _doi(info)
            ee = info.get("ee") if isinstance(info.get("ee"), str) else ""
            seeds.append(
                PaperSeed(
                    title=title,
                    authors=authors,
                    venue=self.venue,
                    year=int(info.get("year") or self.year),
                    source_url=ee or f"https://dblp.org/db/{self.stream}{self.year}.html",
                    abstract=None,
                    doi=doi or None,
                    source_method=self.source_method,
                    source_confidence=0.97 if title and authors else 0.70,
                    evidence_text=f"DBLP {self.stream}{self.year}: {title}",
                )
            )
        return seeds

    def _fetch_toc(self) -> list[dict[str, Any]]:
        key = f"{self.stream}{self.year}.bht"
        hits: list[dict[str, Any]] = []
        offset = 0
        while True:
            url = (
                "https://dblp.org/search/publ/api?"
                f"q=toc%3Adb/{key}%3A&h=100&f={offset}&format=json"
            )
            payload = self.fetch_json(url)
            result = payload.get("result", {}) if isinstance(payload, dict) else {}
            hit_block = result.get("hits", {}) if isinstance(result, dict) else {}
            total = int(hit_block.get("@total", 0) or 0)
            page = hit_block.get("hit", [])
            if isinstance(page, dict):
                page = [page]
            if not isinstance(page, list):
                page = []
            hits.extend(h for h in page if isinstance(h, dict))
            offset += 100
            if offset >= total or not page:
                break
            time.sleep(self.sleep_seconds)
        return hits


def _authors(info: dict[str, Any]) -> list[str]:
    authors = info.get("authors", {})
    raw = authors.get("author", []) if isinstance(authors, dict) else []
    if isinstance(raw, dict):
        raw = [raw]
    names: list[str] = []
    for entry in raw if isinstance(raw, list) else []:
        text = entry.get("text", "") if isinstance(entry, dict) else str(entry)
        cleaned = _DISAMBIGUATION_RE.sub("", text).strip()
        if cleaned:
            names.append(cleaned)
    return names


def _doi(info: dict[str, Any]) -> str:
    doi = info.get("doi")
    if isinstance(doi, str) and doi.strip():
        return doi.strip().lower()
    ee = info.get("ee")
    if isinstance(ee, str):
        match = _DOI_RE.search(ee)
        if match:
            return match.group(0).lower()
    return ""
