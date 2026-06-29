
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any

from wireless_taxonomy.textnorm import normalize_doi, normalize_title


@dataclass(frozen=True)
class PaperRef:
    """A paper identity used for set matching."""

    key: str
    title: str
    normalized_title: str
    normalized_doi: str

    @classmethod
    def build(cls, key: str, title: str | None, doi: str | None = None) -> "PaperRef":
        return cls(
            key=key,
            title=title or "",
            normalized_title=normalize_title(title),
            normalized_doi=normalize_doi(doi),
        )


@dataclass
class MatchResult:
    matched: list[tuple[PaperRef, PaperRef]] = field(default_factory=list)
    unmatched_a: list[PaperRef] = field(default_factory=list)
    unmatched_b: list[PaperRef] = field(default_factory=list)


def match(a_refs: list[PaperRef], b_refs: list[PaperRef], fuzzy_threshold: float = 0.92) -> MatchResult:
    """Match two sets of papers: DOI -> normalized title -> fuzzy title.

    Each B item is consumed at most once. ``fuzzy_threshold`` of 1.0 disables
    fuzzy matching and requires an exact normalized-title (or DOI) match.
    """
    result = MatchResult()
    remaining = list(b_refs)
    doi_index: dict[str, PaperRef] = {ref.normalized_doi: ref for ref in remaining if ref.normalized_doi}
    title_index: dict[str, PaperRef] = {}
    for ref in remaining:
        if ref.normalized_title and ref.normalized_title not in title_index:
            title_index[ref.normalized_title] = ref
    consumed: set[int] = set()

    for a in a_refs:
        chosen: PaperRef | None = None
        if a.normalized_doi and a.normalized_doi in doi_index:
            candidate = doi_index[a.normalized_doi]
            if id(candidate) not in consumed:
                chosen = candidate
        if chosen is None and a.normalized_title and a.normalized_title in title_index:
            candidate = title_index[a.normalized_title]
            if id(candidate) not in consumed:
                chosen = candidate
        if chosen is None and fuzzy_threshold < 1.0 and a.normalized_title:
            chosen = _best_fuzzy(a, remaining, consumed, fuzzy_threshold)
        if chosen is None:
            result.unmatched_a.append(a)
        else:
            consumed.add(id(chosen))
            result.matched.append((a, chosen))

    result.unmatched_b = [ref for ref in remaining if id(ref) not in consumed]
    return result


def _best_fuzzy(a: PaperRef, candidates: list[PaperRef], consumed: set[int], threshold: float) -> PaperRef | None:
    best: PaperRef | None = None
    best_score = threshold
    for b in candidates:
        if id(b) in consumed or not b.normalized_title:
            continue
        score = SequenceMatcher(None, a.normalized_title, b.normalized_title).ratio()
        if score >= best_score:
            best_score = score
            best = b
    return best


@dataclass(frozen=True)
class Metrics:
    tp: int
    fp: int
    fn: int

    @property
    def jaccard(self) -> float:
        denom = self.tp + self.fp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "jaccard": round(self.jaccard, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def jaccard(pred_keys: set[str], gold_keys: set[str]) -> float:
    union = pred_keys | gold_keys
    return len(pred_keys & gold_keys) / len(union) if union else 0.0


def aggregate(instance_rows: list[dict[str, Any]], *, scope_to_universe: bool = False) -> dict[str, Any]:
    """Aggregate per-(venue, year) count rows into per-venue and overall metrics.

    ``instance_rows`` each carry: venue, year, tp, fp, fn, fn_missed,
    fn_missing_from_universe. When ``scope_to_universe`` is set, gold papers that
    are not in the ingested main-proceedings universe (``fn_missing_from_universe``
    — i.e. co-located workshop papers) are dropped from the denominator, so the
    effective FN is ``fn_missed`` only.
    """
    count_cols = ["tp", "fp", "fn", "fn_missed", "fn_missing_from_universe"]

    if not instance_rows:
        empty = _row_with_metrics({"tp": 0, "fp": 0, "fn": 0}, scope_to_universe=scope_to_universe)
        return {"per_conference_year": [], "per_conference": [], "overall": empty}

    per_year = [_row_with_metrics(row, scope_to_universe=scope_to_universe) for row in instance_rows]

    grouped: dict[str, dict[str, int]] = {}
    for row in instance_rows:
        venue = row["venue"]
        bucket = grouped.setdefault(venue, {"venue": venue, **{c: 0 for c in count_cols}})
        for col in count_cols:
            bucket[col] += int(row.get(col, 0) or 0)
    per_conf = [
        _row_with_metrics(grouped[venue], scope_to_universe=scope_to_universe)
        for venue in sorted(grouped)
    ]

    totals = {col: sum(int(row.get(col, 0) or 0) for row in instance_rows) for col in count_cols}
    overall = _row_with_metrics(totals, scope_to_universe=scope_to_universe)

    return {
        "per_conference_year": per_year,
        "per_conference": per_conf,
        "overall": overall,
    }


def _row_with_metrics(row: dict[str, Any], *, scope_to_universe: bool = False) -> dict[str, Any]:
    fn_missed = int(row.get("fn_missed", 0))
    fn_missing = int(row.get("fn_missing_from_universe", 0))
    effective_fn = fn_missed if scope_to_universe else int(row["fn"])
    metrics = Metrics(int(row["tp"]), int(row["fp"]), effective_fn)
    out: dict[str, Any] = {}
    for key in ("venue", "year"):
        if key in row and row[key] is not None:
            out[key] = row[key]
    out.update(metrics.to_dict())
    out["fn_missed"] = fn_missed
    out["fn_missing_from_universe"] = fn_missing
    out["scoped_to_universe"] = scope_to_universe
    return out


def to_markdown(report: dict[str, Any]) -> str:
    """Render an `evaluate_overlap` report as a readable Markdown document."""

    scoped = report.get("scope_to_universe", False)
    lines: list[str] = []
    lines.append("# Wireless classification vs. manual sheet")
    lines.append("")
    lines.append(
        f"- **Classifier:** `{report.get('classifier')}`  |  **Pass:** `{report.get('pass_mode')}`  |  "
        f"**Fuzzy threshold:** `{report.get('fuzzy_threshold')}`"
    )
    lines.append(
        f"- **Scope:** {'main proceedings only — workshop papers dropped' if scoped else 'all gold papers (workshops included)'}"
    )
    lines.append(
        "- **Matching:** DOI → exact title → fuzzy title. Jaccard (IoU) = TP / (TP+FP+FN); "
        "precision = TP/(TP+FP); recall = TP/(TP+FN)."
    )
    lines.append("")

    overall = report.get("overall", {})
    lines.append("## Overall")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append(f"| Jaccard (IoU) | **{overall.get('jaccard')}** |")
    lines.append(f"| precision | {overall.get('precision')} |")
    lines.append(f"| recall | {overall.get('recall')} |")
    lines.append(f"| F1 | {overall.get('f1')} |")
    lines.append(f"| TP / FP / FN | {overall.get('tp')} / {overall.get('fp')} / {overall.get('fn')} |")
    if scoped:
        lines.append(f"| dropped workshop papers | {overall.get('fn_missing_from_universe')} |")
    lines.append("")

    def _table(rows: list[dict[str, Any]], year: bool) -> None:
        header = "| venue | year | jaccard | precision | recall | f1 | tp | fp | fn | fn_miss | dropped |"
        if not year:
            header = "| venue | jaccard | precision | recall | f1 | tp | fp | fn | fn_miss | dropped |"
        lines.append(header)
        lines.append("| " + " | ".join(["---"] * (header.count("|") - 1)) + " |")
        for r in rows:
            cells = [str(r.get("venue", ""))]
            if year:
                cells.append(str(r.get("year", "")))
            cells += [
                str(r.get("jaccard")), str(r.get("precision")), str(r.get("recall")), str(r.get("f1")),
                str(r.get("tp")), str(r.get("fp")), str(r.get("fn")),
                str(r.get("fn_missed")), str(r.get("fn_missing_from_universe")),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    if report.get("instances"):
        lines.append("## Per conference-year")
        lines.append("")
        _table(report["instances"], year=True)

    if report.get("per_conference"):
        lines.append("## Per venue (all years)")
        lines.append("")
        _table(report["per_conference"], year=False)

    under = report.get("under_curated_instances") or []
    if under:
        lines.append("## Under-curated / excluded venue-years (not in headline)")
        lines.append("")
        lines.append(
            "These venue-years are reported separately and excluded from the overall "
            "metrics (e.g. thinly- or stale-curated rows whose gold set is too small "
            "to score fairly). Their would-be metrics are shown for reference."
        )
        lines.append("")
        lines.append("| venue | year | gold | reason | precision | recall | f1 | tp | fp | fn |")
        lines.append("| " + " | ".join(["---"] * 10) + " |")
        for r in under:
            lines.append(
                "| "
                + " | ".join(
                    str(x)
                    for x in (
                        r.get("venue", ""),
                        r.get("year", ""),
                        r.get("gold_papers", ""),
                        r.get("reason", ""),
                        r.get("precision"),
                        r.get("recall"),
                        r.get("f1"),
                        r.get("tp"),
                        r.get("fp"),
                        r.get("fn"),
                    )
                )
                + " |"
            )
        lines.append("")

    mismatches = report.get("mismatches") or []
    if mismatches:
        lines.append("## Discrepancies")
        lines.append("")
        lines.append(
            "`classifier_miss` = paper is in the main proceedings but the classifier didn't flag it. "
            "`missing_from_universe` = paper in your sheet but not in the main proceedings (workshop / out of scope)."
        )
        lines.append("")
        for m in mismatches:
            fp = m.get("false_positives") or []
            miss = m.get("false_negatives_classifier_miss") or []
            outside = m.get("false_negatives_missing_from_universe") or []
            lines.append(f"### {m.get('venue')} {m.get('year')}")
            lines.append(f"- false positives ({len(fp)}): {', '.join(fp) if fp else '—'}")
            lines.append(f"- classifier misses ({len(miss)}): {', '.join(miss) if miss else '—'}")
            label = "dropped workshop papers" if scoped else "missing from proceedings"
            lines.append(f"- {label} ({len(outside)}): {', '.join(outside) if outside else '—'}")
            lines.append("")

    return "\n".join(lines)
