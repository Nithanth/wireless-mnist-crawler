from __future__ import annotations

import csv
import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wireless_taxonomy.analyze.text_match import normalize_title
from wireless_taxonomy.eval.overlap import Metrics
from wireless_taxonomy.evaluate.matching import author_overlap, make_record, match_papers
from wireless_taxonomy.textnorm import normalize_doi

# Per-paper diff CSV header.
DIFF_COLUMNS = [
    "status",
    "title_a",
    "title_b",
    "doi",
    "match_type",
    "title_similarity",
    "author_overlap",
    "shared_authors",
    "abstract_a",
    "abstract_b",
    "wireless_label_a",
    "wireless_label_b",
]


def load_paper_set(path: str | Path) -> list[dict[str, Any]]:
    """Load a `paper-set` export (csv or json) into a list of row dicts.

    Accepts either format the `paper-set` command emits. utf-8-sig tolerates the
    BOM that spreadsheet exports prepend. Rows missing a title are dropped.
    """

    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = data if isinstance(data, list) else []
    else:
        with path.open("r", newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
    return [row for row in rows if isinstance(row, dict) and str(row.get("title") or "").strip()]


@dataclass(frozen=True)
class DiffSummary:
    """Overlap between two automated paper sets (e.g. URL+LLM vs DBLP+OpenAlex).

    Counts are over deduplicated papers, so `jaccard` is the IoU of the two sets.
    `abstracts_a`/`abstracts_b` count how many papers in each set carry an
    abstract. When `reference` ("a" or "b") names the ground-truth side, the other
    side is scored against it as predictions (precision/recall/F1).
    """

    label_a: str
    label_b: str
    count_a: int
    count_b: int
    shared: int
    only_in_a: int
    only_in_b: int
    doi_count: int
    fuzzy_count: int
    abstracts_a: int
    abstracts_b: int
    reference: str | None = None

    @property
    def union(self) -> int:
        return self.shared + self.only_in_a + self.only_in_b

    @property
    def jaccard(self) -> float:
        return self.shared / self.union if self.union else 1.0

    def metrics(self) -> Metrics | None:
        """Precision/recall/F1 of the non-reference side vs the reference side.

        `reference="b"` treats B as ground truth: FP = papers only in A (extra),
        FN = papers only in B (missed). `reference="a"` is the mirror image.
        Returns None when no reference side is chosen.
        """
        if self.reference == "b":
            return Metrics(tp=self.shared, fp=self.only_in_a, fn=self.only_in_b)
        if self.reference == "a":
            return Metrics(tp=self.shared, fp=self.only_in_b, fn=self.only_in_a)
        return None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "jaccard_index": round(self.jaccard, 4),
            "counts": {
                "a": self.count_a,
                "b": self.count_b,
                "shared": self.shared,
                "union": self.union,
                "only_in_a": self.only_in_a,
                "only_in_b": self.only_in_b,
                "doi_matches": self.doi_count,
                "fuzzy": self.fuzzy_count,
            },
            "abstract_coverage": {
                "a": self.abstracts_a,
                "b": self.abstracts_b,
            },
        }
        metrics = self.metrics()
        if metrics is not None:
            payload["reference"] = self.reference
            payload["precision_recall"] = metrics.to_dict()
        return payload


def _has_abstract(row: dict[str, Any]) -> bool:
    return bool(str(row.get("abstract") or "").strip())


def _index_by_key(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """First-wins map of normalized title -> row (deduplicates by match key)."""
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = normalize_title(str(row.get("title") or ""))
        if key:
            by_key.setdefault(key, row)
    return by_key


def _shared_row(a_row: dict[str, Any], b_row: dict[str, Any], match_type: str, similarity: float) -> dict[str, Any]:
    a_rec = make_record(str(a_row.get("title") or ""), a_row.get("authors"))
    b_rec = make_record(str(b_row.get("title") or ""), b_row.get("authors"))
    overlap, shared = author_overlap(a_rec.surnames, b_rec.surnames)
    return {
        "status": "shared",
        "title_a": a_row.get("title") or "",
        "title_b": b_row.get("title") or "",
        "doi": str(a_row.get("doi") or b_row.get("doi") or ""),
        "match_type": match_type,
        "title_similarity": round(similarity, 4),
        "author_overlap": round(overlap, 4),
        "shared_authors": "; ".join(shared),
        "abstract_a": "yes" if _has_abstract(a_row) else "no",
        "abstract_b": "yes" if _has_abstract(b_row) else "no",
        "wireless_label_a": str(a_row.get("wireless_label") or ""),
        "wireless_label_b": str(b_row.get("wireless_label") or ""),
    }


def diff_paper_sets(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    *,
    fuzzy: bool = True,
    label_a: str = "A",
    label_b: str = "B",
    reference: str | None = None,
) -> tuple[DiffSummary, list[dict[str, Any]]]:
    """Diff two paper sets: shared / only-in-A / only-in-B, plus a per-paper view.

    Matching is DOI-first (exact normalized DOI), then exact normalized title,
    then optional fuzzy title similarity with an author-overlap boost — each paper
    used once. `A` is the left side, `B` the right side. When `reference` is "a" or
    "b", precision/recall/F1 of the other side are computed against it.
    """

    if reference not in (None, "a", "b"):
        raise ValueError("reference must be 'a', 'b', or None")

    a_by_key = _index_by_key(rows_a)
    b_by_key = _index_by_key(rows_b)

    diff_rows: list[dict[str, Any]] = []
    used_a: set[str] = set()
    used_b: set[str] = set()
    doi_count = 0

    # 1) DOI-first: an exact normalized-DOI match beats any title logic.
    b_by_doi: dict[str, str] = {}
    for key, row in b_by_key.items():
        doi = normalize_doi(row.get("doi"))
        if doi:
            b_by_doi.setdefault(doi, key)
    doi_pairs: list[dict[str, Any]] = []
    for a_key, a_row in a_by_key.items():
        doi = normalize_doi(a_row.get("doi"))
        if not doi:
            continue
        b_key = b_by_doi.get(doi)
        if b_key is None or b_key in used_b:
            continue
        b_row = b_by_key[b_key]
        similarity = difflib.SequenceMatcher(None, a_key, b_key).ratio()
        doi_pairs.append(_shared_row(a_row, b_row, "doi", similarity))
        used_a.add(a_key)
        used_b.add(b_key)
        doi_count += 1

    # 2) Title (exact then fuzzy) over the papers not already DOI-matched.
    rem_a = [make_record(r["title"], r.get("authors")) for k, r in a_by_key.items() if k not in used_a]
    rem_b = [make_record(r["title"], r.get("authors")) for k, r in b_by_key.items() if k not in used_b]
    result = match_papers(rem_a, rem_b, fuzzy=fuzzy)
    fuzzy_count = sum(1 for pair in result.matched if pair.method == "fuzzy")

    title_pairs: list[dict[str, Any]] = []
    for pair in result.matched:
        a_row = a_by_key.get(normalize_title(pair.manual_title), {})
        b_row = b_by_key.get(normalize_title(pair.automated_title), {})
        title_pairs.append(_shared_row(a_row, b_row, pair.method, pair.title_similarity))

    diff_rows.extend(doi_pairs)
    diff_rows.extend(title_pairs)

    for title in result.missed_by_cli:
        a_row = a_by_key.get(normalize_title(title), {})
        diff_rows.append(
            {
                "status": "only_in_a",
                "title_a": title,
                "title_b": "",
                "doi": str(a_row.get("doi") or ""),
                "match_type": "",
                "title_similarity": "",
                "author_overlap": "",
                "shared_authors": "",
                "abstract_a": "yes" if _has_abstract(a_row) else "no",
                "abstract_b": "",
                "wireless_label_a": str(a_row.get("wireless_label") or ""),
                "wireless_label_b": "",
            }
        )
    for title in result.extra_from_cli:
        b_row = b_by_key.get(normalize_title(title), {})
        diff_rows.append(
            {
                "status": "only_in_b",
                "title_a": "",
                "title_b": title,
                "doi": str(b_row.get("doi") or ""),
                "match_type": "",
                "title_similarity": "",
                "author_overlap": "",
                "shared_authors": "",
                "abstract_a": "",
                "abstract_b": "yes" if _has_abstract(b_row) else "no",
                "wireless_label_a": "",
                "wireless_label_b": str(b_row.get("wireless_label") or ""),
            }
        )

    summary = DiffSummary(
        label_a=label_a,
        label_b=label_b,
        count_a=len(a_by_key),
        count_b=len(b_by_key),
        shared=doi_count + len(result.matched),
        only_in_a=len(result.missed_by_cli),
        only_in_b=len(result.extra_from_cli),
        doi_count=doi_count,
        fuzzy_count=fuzzy_count,
        abstracts_a=sum(1 for row in a_by_key.values() if _has_abstract(row)),
        abstracts_b=sum(1 for row in b_by_key.values() if _has_abstract(row)),
        reference=reference,
    )
    return summary, diff_rows


def write_diff_csv(rows: list[dict[str, Any]], output: str | Path) -> Path:
    output = Path(output)
    if output.suffix.lower() != ".csv":
        output = output.with_suffix(".csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DIFF_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_diff_report(summary: DiffSummary, rows: list[dict[str, Any]], output: str | Path) -> Path:
    output = Path(output)
    if output.suffix.lower() != ".json":
        output = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {**summary.to_dict(), "papers": rows}
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def _coverage(count: int, total: int) -> str:
    pct = (count / total * 100) if total else 0.0
    return f"{count}/{total} ({pct:.0f}%)"


def format_diff_summary(summary: DiffSummary) -> str:
    """Readable multi-line summary of a two-set diff."""

    lines = [
        f"{summary.label_a} vs {summary.label_b}  —  Jaccard (IoU) = {summary.jaccard:.4f}",
        f"  shared (intersection)  : {summary.shared:>4}   (doi: {summary.doi_count}, fuzzy: {summary.fuzzy_count})",
        f"  union                  : {summary.union:>4}",
        f"  {summary.label_a} / {summary.label_b} (papers)   : {summary.count_a:>4} / {summary.count_b}",
        f"  only in {summary.label_a:<14}: {summary.only_in_a:>4}",
        f"  only in {summary.label_b:<14}: {summary.only_in_b:>4}",
        f"  abstracts in {summary.label_a:<9}: {_coverage(summary.abstracts_a, summary.count_a)}",
        f"  abstracts in {summary.label_b:<9}: {_coverage(summary.abstracts_b, summary.count_b)}",
    ]
    metrics = summary.metrics()
    if metrics is not None:
        gold = summary.label_b if summary.reference == "b" else summary.label_a
        pred = summary.label_a if summary.reference == "b" else summary.label_b
        lines.append(f"  vs ground truth ({gold}); scoring {pred}:")
        lines.append(
            f"    precision = {metrics.precision:.4f}   recall = {metrics.recall:.4f}   f1 = {metrics.f1:.4f}"
        )
    return "\n".join(lines)
