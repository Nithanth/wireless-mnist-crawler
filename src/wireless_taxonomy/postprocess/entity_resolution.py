"""Dataset entity resolution: detect duplicate/same datasets across papers.

Two strategies, run in order:
1. URL/DOI dedup — near-certain matches from shared availability URLs or DOIs.
2. Candidate flagging — normalized name + modality + OSI similarity scoring
   to surface probable-same pairs for human review.

Future: LLM confirmation step behind the same interface.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Protocol
from urllib.parse import urlparse


@dataclass
class DatasetRecord:
    """A single dataset row (from merged CSV or raw JSON)."""

    name: str
    bibtex_keys: list[str] = field(default_factory=list)
    modalities: str = ""
    osi_layers: str = ""
    environment: str = ""
    availability_url: str = ""
    availability_notes: str = ""
    doi: str = ""

    @property
    def all_urls(self) -> list[str]:
        """Extract all URLs from availability fields."""
        text = f"{self.availability_url} {self.availability_notes}"
        return re.findall(r"https?://[^\s,;)\"']+", text)


@dataclass
class Match:
    """A pair of datasets flagged as potentially the same."""

    a: DatasetRecord
    b: DatasetRecord
    confidence: float  # 0.0–1.0
    reason: str
    method: str  # "url_dedup" | "similarity" | "llm" (future)


class Resolver(Protocol):
    """Interface for entity resolution strategies."""

    def resolve(self, datasets: list[DatasetRecord]) -> list[Match]: ...


# ─── Strategy 1: URL/DOI dedup ────────────────────────────────────────────────


def _normalize_url(url: str) -> str:
    """Strip trailing slashes, fragments, tracking params for comparison."""
    parsed = urlparse(url.strip().rstrip("/"))
    # Drop fragment and common tracking params
    path = parsed.path.rstrip("/")
    # Normalize github URLs: strip .git suffix, tree/main etc
    if "github.com" in parsed.netloc:
        path = re.sub(r"\.git$", "", path)
        path = re.sub(r"/tree/(main|master)(/.*)?$", "", path)
    return f"{parsed.netloc}{path}".lower()


class URLDedup:
    """Group datasets that share a normalized URL or DOI."""

    def resolve(self, datasets: list[DatasetRecord]) -> list[Match]:
        url_to_records: dict[str, list[DatasetRecord]] = defaultdict(list)

        for ds in datasets:
            for url in ds.all_urls:
                norm = _normalize_url(url)
                if norm:
                    url_to_records[norm].append(ds)
            if ds.doi:
                url_to_records[f"doi:{ds.doi.lower().strip()}"].append(ds)

        matches: list[Match] = []
        seen_pairs: set[tuple[str, str]] = set()

        for _url, records in url_to_records.items():
            if len(records) < 2:
                continue
            for i, a in enumerate(records):
                for b in records[i + 1 :]:
                    # Skip same-paper datasets
                    if set(a.bibtex_keys) == set(b.bibtex_keys):
                        continue
                    pair_key = tuple(sorted([a.name, b.name]))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    matches.append(Match(
                        a=a, b=b,
                        confidence=0.95,
                        reason=f"shared URL: {_url}",
                        method="url_dedup",
                    ))

        return matches


# ─── Strategy 2: Similarity-based candidate flagging ──────────────────────────


def _normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    n = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    # Strip generic suffixes that inflate false matches
    n = re.sub(r"\b(dataset|traces|measurements|data|performance)\b", "", n)
    return re.sub(r"\s+", " ", n).strip()


def _normalize_modality(mod: str) -> str:
    n = re.sub(r"[^a-z0-9 ,]", " ", mod.lower())
    return re.sub(r"\s+", " ", n).strip()


class SimilarityFlagger:
    """Flag dataset pairs with high combined similarity."""

    def __init__(self, name_threshold: float = 0.75, combined_threshold: float = 0.70):
        self.name_threshold = name_threshold
        self.combined_threshold = combined_threshold

    def _signature(self, ds: DatasetRecord) -> str:
        return f"{_normalize_name(ds.name)} {_normalize_modality(ds.modalities)} {ds.osi_layers.lower()}"

    def resolve(self, datasets: list[DatasetRecord]) -> list[Match]:
        matches: list[Match] = []

        for i, a in enumerate(datasets):
            for b in datasets[i + 1 :]:
                # Skip same-paper
                if set(a.bibtex_keys) == set(b.bibtex_keys):
                    continue

                name_ratio = SequenceMatcher(
                    None, _normalize_name(a.name), _normalize_name(b.name)
                ).ratio()
                combined_ratio = SequenceMatcher(
                    None, self._signature(a), self._signature(b)
                ).ratio()

                if name_ratio >= self.name_threshold or combined_ratio >= self.combined_threshold:
                    confidence = max(name_ratio, combined_ratio) * 0.8  # cap below url_dedup
                    matches.append(Match(
                        a=a, b=b,
                        confidence=round(confidence, 3),
                        reason=f"name={name_ratio:.2f} combined={combined_ratio:.2f}",
                        method="similarity",
                    ))

        return matches


# ─── Strategy 3: LLM confirmation (stub) ─────────────────────────────────────


class LLMConfirmer:
    """Future: take candidate pairs and ask an LLM 'are these the same dataset?'

    Not implemented — stub maintains the interface so it can be dropped in
    when corpus scale justifies it.
    """

    def resolve(self, datasets: list[DatasetRecord]) -> list[Match]:
        raise NotImplementedError(
            "LLM entity resolution not yet implemented. "
            "Use URLDedup + SimilarityFlagger for now."
        )


# ─── Orchestrator ─────────────────────────────────────────────────────────────


def reconcile(
    datasets: list[DatasetRecord],
    *,
    url_dedup: bool = True,
    similarity: bool = True,
    similarity_name_threshold: float = 0.75,
    similarity_combined_threshold: float = 0.70,
) -> list[Match]:
    """Run all enabled resolution strategies and return deduplicated matches.

    Matches from url_dedup are high-confidence (0.95).
    Matches from similarity are lower (capped at 0.8).
    Results are sorted by confidence descending.
    """
    all_matches: list[Match] = []
    seen_pairs: set[tuple[str, str]] = set()

    if url_dedup:
        for m in URLDedup().resolve(datasets):
            pair_key = tuple(sorted([m.a.name, m.b.name]))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                all_matches.append(m)

    if similarity:
        flagger = SimilarityFlagger(
            name_threshold=similarity_name_threshold,
            combined_threshold=similarity_combined_threshold,
        )
        for m in flagger.resolve(datasets):
            pair_key = tuple(sorted([m.a.name, m.b.name]))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                all_matches.append(m)

    all_matches.sort(key=lambda m: -m.confidence)
    return all_matches
