"""Dataset entity resolution: detect duplicate/same datasets across papers.

Three strategies, run in order:
1. URL/DOI dedup — near-certain matches from shared availability URLs or DOIs.
2. Candidate flagging — normalized name + modality + OSI similarity scoring.
3. LLM confirmation — takes similarity candidates and asks the LLM whether
   two datasets are genuinely the same artifact. Returns yes/no/unsure.

The pipeline flow is:
  merge-results → reconcile-datasets [--llm-confirm]
                         ↓
              consolidated_datasets.csv  (canonical, deduplicated)
"""

import json as _json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Protocol
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


# ─── Strategy 3: LLM confirmation ─────────────────────────────────────────────

_LLM_CONFIRM_PROMPT = """\
You are a research dataset deduplication assistant. Given two dataset descriptions \
extracted from different academic papers, determine whether they refer to the SAME \
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

Respond with JSON: {{"verdict": "yes" | "no" | "unsure", "reason": "<brief explanation>"}}
"""


class LLMConfirmer:
    """Ask an LLM to confirm/reject similarity candidates.

    Takes pre-filtered candidate pairs (from SimilarityFlagger) and confirms
    whether they are genuinely the same dataset. This acts as a precision
    filter on the high-recall similarity stage.

    Verdicts:
      - "yes"    → auto-merge (confidence 0.90)
      - "unsure" → flag for human review (confidence 0.70)
      - "no"     → drop (not returned)
    """

    def __init__(self, llm_complete: "Any | None" = None):
        """Accept a callable(prompt: str) -> str for LLM completion.

        If None, will lazily initialize from the project's LlmRouter.
        """
        self._complete = llm_complete

    def _get_complete(self) -> "Any":
        if self._complete is None:
            from wireless_taxonomy.config import load_dotenv, load_llm_settings
            from wireless_taxonomy.llm import LlmRequest, LlmRouter
            load_dotenv()
            router = LlmRouter(load_llm_settings())

            def _complete(prompt: str) -> str:
                resp = router.complete(LlmRequest(task="entity_resolution", prompt=prompt))
                return resp.content

            self._complete = _complete
        return self._complete

    def confirm_pairs(self, candidates: list[Match]) -> list[Match]:
        """Filter candidate matches through LLM confirmation.

        Returns only confirmed ("yes") and uncertain ("unsure") pairs.
        "no" verdicts are dropped.
        """
        complete = self._get_complete()
        confirmed: list[Match] = []

        for candidate in candidates:
            prompt = _LLM_CONFIRM_PROMPT.format(
                a_name=candidate.a.name,
                a_keys=", ".join(candidate.a.bibtex_keys),
                a_mod=candidate.a.modalities,
                a_osi=candidate.a.osi_layers,
                a_env=candidate.a.environment,
                a_url=candidate.a.availability_url,
                b_name=candidate.b.name,
                b_keys=", ".join(candidate.b.bibtex_keys),
                b_mod=candidate.b.modalities,
                b_osi=candidate.b.osi_layers,
                b_env=candidate.b.environment,
                b_url=candidate.b.availability_url,
            )

            try:
                raw = complete(prompt)
                parsed = _parse_verdict(raw)
                verdict = parsed.get("verdict", "unsure").lower().strip()
                reason = parsed.get("reason", "")
            except Exception:
                verdict = "unsure"
                reason = "LLM call failed; flagging for human review"

            if verdict == "no":
                continue  # drop false positive

            confidence = 0.90 if verdict == "yes" else 0.70
            confirmed.append(Match(
                a=candidate.a,
                b=candidate.b,
                confidence=confidence,
                reason=f"LLM {verdict}: {reason}",
                method="llm_confirmed" if verdict == "yes" else "llm_unsure",
            ))

        return confirmed

    def resolve(self, datasets: list[DatasetRecord]) -> list[Match]:
        """Full resolve: run similarity first, then confirm with LLM."""
        flagger = SimilarityFlagger(name_threshold=0.60, combined_threshold=0.55)
        candidates = flagger.resolve(datasets)
        return self.confirm_pairs(candidates)


def _parse_verdict(raw: str) -> dict[str, str]:
    """Extract verdict JSON from LLM response, tolerant of markdown fences."""
    raw = raw.strip()
    # Strip markdown code fences
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return _json.loads(raw)
    except _json.JSONDecodeError:
        # Try to find JSON object in the response
        match = re.search(r"\{[^}]+\}", raw)
        if match:
            try:
                return _json.loads(match.group())
            except _json.JSONDecodeError:
                pass
    return {"verdict": "unsure", "reason": "Could not parse LLM response"}


# ─── Orchestrator ─────────────────────────────────────────────────────────────


def reconcile(
    datasets: list[DatasetRecord],
    *,
    url_dedup: bool = True,
    similarity: bool = True,
    llm_confirm: bool = False,
    similarity_name_threshold: float = 0.75,
    similarity_combined_threshold: float = 0.70,
) -> list[Match]:
    """Run all enabled resolution strategies and return deduplicated matches.

    Matches from url_dedup are high-confidence (0.95).
    Matches from similarity are lower (capped at 0.8).
    When llm_confirm=True, similarity candidates are passed through the LLM
    confirmer — "no" verdicts are dropped, "yes" get confidence 0.90,
    "unsure" get 0.70.
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
        sim_candidates = flagger.resolve(datasets)

        # Filter out candidates already found by URL dedup
        sim_filtered = []
        for m in sim_candidates:
            pair_key = tuple(sorted([m.a.name, m.b.name]))
            if pair_key not in seen_pairs:
                sim_filtered.append(m)

        if llm_confirm and sim_filtered:
            # LLM precision filter: confirm or reject similarity candidates
            confirmer = LLMConfirmer()
            confirmed = confirmer.confirm_pairs(sim_filtered)
            for m in confirmed:
                pair_key = tuple(sorted([m.a.name, m.b.name]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    all_matches.append(m)
        else:
            # No LLM: surface all similarity candidates for human review
            for m in sim_filtered:
                pair_key = tuple(sorted([m.a.name, m.b.name]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    all_matches.append(m)

    all_matches.sort(key=lambda m: -m.confidence)
    return all_matches


# ─── Consolidation: merge duplicates into canonical dataset list ──────────────


@dataclass
class CanonicalDataset:
    """A deduplicated dataset with merged metadata from all instances."""

    canonical_name: str
    all_names: list[str] = field(default_factory=list)
    bibtex_keys: list[str] = field(default_factory=list)
    modalities: str = ""
    osi_layers: str = ""
    environments: list[str] = field(default_factory=list)
    availability_url: str = ""
    is_open: bool | None = None
    reuse_count: int = 1  # number of distinct papers using this dataset
    merge_reason: str = ""


def consolidate(
    datasets: list[DatasetRecord],
    matches: list[Match],
    *,
    auto_merge_threshold: float = 0.85,
) -> list[CanonicalDataset]:
    """Merge confirmed duplicates into a canonical dataset list.

    Datasets linked by matches above ``auto_merge_threshold`` are merged.
    The result is a deduplicated list with proper reuse counts.

    Each CanonicalDataset represents one real-world dataset, with:
    - canonical_name: longest/most descriptive name from the group
    - all_names: all variant names encountered
    - bibtex_keys: all papers that use this dataset
    - reuse_count: len(bibtex_keys) — the number of papers
    """
    # Build union-find for merging
    parent: dict[int, int] = {}  # index -> root index

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    # Index datasets by name for lookup
    name_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, ds in enumerate(datasets):
        name_to_indices[ds.name].append(i)
        parent[i] = i

    # Apply merges from confirmed matches
    for m in matches:
        if m.confidence < auto_merge_threshold:
            continue  # only auto-merge high-confidence matches
        # Find indices for each dataset in the match
        a_indices = name_to_indices.get(m.a.name, [])
        b_indices = name_to_indices.get(m.b.name, [])
        if a_indices and b_indices:
            union(a_indices[0], b_indices[0])

    # Also merge exact-name datasets (same name, different papers)
    for name, indices in name_to_indices.items():
        for idx in indices[1:]:
            union(indices[0], idx)

    # Group by root
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(datasets)):
        groups[find(i)].append(i)

    # Build canonical entries
    canonical_list: list[CanonicalDataset] = []
    for root, indices in sorted(groups.items()):
        group = [datasets[i] for i in indices]

        # Canonical name: longest name (most descriptive)
        all_names = sorted(set(ds.name for ds in group), key=len, reverse=True)
        canonical_name = all_names[0]

        # Merge bibtex keys (deduplicated, preserving order)
        seen_keys: dict[str, None] = {}
        for ds in group:
            for k in ds.bibtex_keys:
                seen_keys.setdefault(k, None)
        all_keys = list(seen_keys)

        # Merge modalities: take the longest/most detailed
        modalities = max((ds.modalities for ds in group), key=len, default="")

        # Merge OSI layers: union
        osi_set: set[str] = set()
        for ds in group:
            for layer in re.split(r"[;,\s]+", ds.osi_layers):
                layer = layer.strip()
                if layer:
                    osi_set.add(layer)
        osi_layers = ", ".join(sorted(osi_set))

        # Environments
        envs = sorted(set(ds.environment for ds in group if ds.environment))

        # Availability URL: first non-empty
        url = next((ds.availability_url for ds in group if ds.availability_url), "")

        # Merge reason
        reason = ""
        if len(all_names) > 1:
            reason = f"merged {len(all_names)} name variants"

        canonical_list.append(CanonicalDataset(
            canonical_name=canonical_name,
            all_names=all_names,
            bibtex_keys=all_keys,
            modalities=modalities,
            osi_layers=osi_layers,
            environments=envs,
            availability_url=url,
            reuse_count=len(all_keys),
            merge_reason=reason,
        ))

    # Sort by reuse count descending, then name
    canonical_list.sort(key=lambda d: (-d.reuse_count, d.canonical_name.lower()))
    return canonical_list
