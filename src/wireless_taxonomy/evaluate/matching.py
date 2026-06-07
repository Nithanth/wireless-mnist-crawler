from __future__ import annotations

import difflib
from dataclasses import dataclass

from wireless_taxonomy.analyze.text_match import author_names, last_name, normalize_title

# Title similarity at/above this accepts a match on its own.
DEFAULT_TITLE_THRESHOLD = 0.92
# Below the strict threshold, a match still counts if title similarity clears this
# floor AND enough authors overlap (the "author boost").
AUTHOR_BOOST_TITLE_THRESHOLD = 0.80
AUTHOR_BOOST_MIN_OVERLAP = 0.5


@dataclass(frozen=True)
class PaperRecord:
    key: str  # normalized title, used as the primary comparison key
    title: str  # original title (for display)
    authors: str  # raw authors string
    surnames: frozenset[str]  # normalized author surnames for overlap scoring


def make_record(title: str, authors: str | None = None) -> PaperRecord:
    surnames = frozenset(name for name in (last_name(n) for n in author_names(authors)) if name)
    return PaperRecord(key=normalize_title(title), title=title, authors=authors or "", surnames=surnames)


@dataclass(frozen=True)
class MatchPair:
    manual_title: str
    automated_title: str
    title_similarity: float
    author_overlap: float
    shared_authors: list[str]
    method: str  # "exact" or "fuzzy"


@dataclass(frozen=True)
class MatchResult:
    matched: list[MatchPair]
    missed_by_cli: list[str]  # manual titles with no automated match
    extra_from_cli: list[str]  # automated titles with no manual match


def author_overlap(left: frozenset[str], right: frozenset[str]) -> tuple[float, list[str]]:
    if not left or not right:
        return 0.0, []
    shared = left & right
    denom = min(len(left), len(right))
    return (len(shared) / denom if denom else 0.0), sorted(shared)


def _dedup_by_key(records: list[PaperRecord]) -> dict[str, PaperRecord]:
    by_key: dict[str, PaperRecord] = {}
    for record in records:
        if record.key:
            by_key.setdefault(record.key, record)
    return by_key


def match_papers(
    manual: list[PaperRecord],
    automated: list[PaperRecord],
    *,
    fuzzy: bool = True,
    title_threshold: float = DEFAULT_TITLE_THRESHOLD,
    author_boost_title_threshold: float = AUTHOR_BOOST_TITLE_THRESHOLD,
    author_boost_min_overlap: float = AUTHOR_BOOST_MIN_OVERLAP,
) -> MatchResult:
    """Match manual papers to automated papers, one-to-one.

    Exact normalized-title matches are taken first. When `fuzzy` is on, remaining
    papers are matched by title similarity (difflib ratio) with an author-overlap
    boost: a sub-threshold title can still match if enough author surnames overlap.
    Candidate pairs are assigned greedily by descending score so each paper is used once.
    """

    manual_by_key = _dedup_by_key(manual)
    auto_by_key = _dedup_by_key(automated)

    matched: list[MatchPair] = []
    used_manual: set[str] = set()
    used_auto: set[str] = set()

    for key, mrec in manual_by_key.items():
        arec = auto_by_key.get(key)
        if arec is None:
            continue
        overlap, shared = author_overlap(mrec.surnames, arec.surnames)
        matched.append(MatchPair(mrec.title, arec.title, 1.0, overlap, shared, "exact"))
        used_manual.add(key)
        used_auto.add(key)

    if fuzzy:
        candidates: list[tuple[float, float, float, list[str], PaperRecord, PaperRecord]] = []
        for mrec in (r for k, r in manual_by_key.items() if k not in used_manual):
            for arec in (r for k, r in auto_by_key.items() if k not in used_auto):
                similarity = difflib.SequenceMatcher(None, mrec.key, arec.key).ratio()
                overlap, shared = author_overlap(mrec.surnames, arec.surnames)
                accepted = similarity >= title_threshold or (
                    similarity >= author_boost_title_threshold and overlap >= author_boost_min_overlap
                )
                if accepted:
                    rank = similarity + 0.05 * overlap
                    candidates.append((rank, similarity, overlap, shared, mrec, arec))
        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, similarity, overlap, shared, mrec, arec in candidates:
            if mrec.key in used_manual or arec.key in used_auto:
                continue
            matched.append(MatchPair(mrec.title, arec.title, similarity, overlap, shared, "fuzzy"))
            used_manual.add(mrec.key)
            used_auto.add(arec.key)

    matched.sort(key=lambda pair: (pair.method != "exact", -pair.title_similarity, pair.manual_title.lower()))
    missed_by_cli = sorted(r.title for k, r in manual_by_key.items() if k not in used_manual)
    extra_from_cli = sorted(r.title for k, r in auto_by_key.items() if k not in used_auto)
    return MatchResult(matched=matched, missed_by_cli=missed_by_cli, extra_from_cli=extra_from_cli)
