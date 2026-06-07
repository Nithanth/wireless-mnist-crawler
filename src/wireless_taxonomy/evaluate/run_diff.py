from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wireless_taxonomy.analyze.text_match import normalize_title
from wireless_taxonomy.evaluate.matching import make_record, match_papers

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

    Counts are over deduplicated normalized-title keys, so `jaccard` is the IoU of
    the two sets. `abstracts_a`/`abstracts_b` count how many papers in each set
    carry an abstract — the lever the URL+LLM pass is meant to improve.
    """

    label_a: str
    label_b: str
    count_a: int
    count_b: int
    shared: int
    only_in_a: int
    only_in_b: int
    fuzzy_count: int
    abstracts_a: int
    abstracts_b: int

    @property
    def union(self) -> int:
        return self.shared + self.only_in_a + self.only_in_b

    @property
    def jaccard(self) -> float:
        return self.shared / self.union if self.union else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
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
                "fuzzy": self.fuzzy_count,
            },
            "abstract_coverage": {
                "a": self.abstracts_a,
                "b": self.abstracts_b,
            },
        }


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


def diff_paper_sets(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    *,
    fuzzy: bool = True,
    label_a: str = "A",
    label_b: str = "B",
) -> tuple[DiffSummary, list[dict[str, Any]]]:
    """Diff two paper sets: shared / only-in-A / only-in-B, plus a per-paper view.

    Matching reuses the shared matcher (exact normalized title, then optional
    fuzzy title similarity with an author-overlap boost), one-to-one. `A` is
    treated as the left side, `B` as the right side.
    """

    a_by_key = _index_by_key(rows_a)
    b_by_key = _index_by_key(rows_b)

    records_a = [make_record(row["title"], row.get("authors")) for row in a_by_key.values()]
    records_b = [make_record(row["title"], row.get("authors")) for row in b_by_key.values()]
    result = match_papers(records_a, records_b, fuzzy=fuzzy)

    fuzzy_count = sum(1 for pair in result.matched if pair.method == "fuzzy")
    summary = DiffSummary(
        label_a=label_a,
        label_b=label_b,
        count_a=len(a_by_key),
        count_b=len(b_by_key),
        shared=len(result.matched),
        only_in_a=len(result.missed_by_cli),
        only_in_b=len(result.extra_from_cli),
        fuzzy_count=fuzzy_count,
        abstracts_a=sum(1 for row in a_by_key.values() if _has_abstract(row)),
        abstracts_b=sum(1 for row in b_by_key.values() if _has_abstract(row)),
    )

    def _a(title: str) -> dict[str, Any]:
        return a_by_key.get(normalize_title(title), {})

    def _b(title: str) -> dict[str, Any]:
        return b_by_key.get(normalize_title(title), {})

    diff_rows: list[dict[str, Any]] = []
    for pair in result.matched:
        a_row, b_row = _a(pair.manual_title), _b(pair.automated_title)
        diff_rows.append(
            {
                "status": "shared",
                "title_a": pair.manual_title,
                "title_b": pair.automated_title,
                "doi": str(a_row.get("doi") or b_row.get("doi") or ""),
                "match_type": pair.method,
                "title_similarity": round(pair.title_similarity, 4),
                "author_overlap": round(pair.author_overlap, 4),
                "shared_authors": "; ".join(pair.shared_authors),
                "abstract_a": "yes" if _has_abstract(a_row) else "no",
                "abstract_b": "yes" if _has_abstract(b_row) else "no",
                "wireless_label_a": str(a_row.get("wireless_label") or ""),
                "wireless_label_b": str(b_row.get("wireless_label") or ""),
            }
        )
    for title in result.missed_by_cli:
        a_row = _a(title)
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
        b_row = _b(title)
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
        f"  shared (intersection)  : {summary.shared:>4}   (fuzzy: {summary.fuzzy_count})",
        f"  union                  : {summary.union:>4}",
        f"  {summary.label_a} / {summary.label_b} (papers)   : {summary.count_a:>4} / {summary.count_b}",
        f"  only in {summary.label_a:<14}: {summary.only_in_a:>4}",
        f"  only in {summary.label_b:<14}: {summary.only_in_b:>4}",
        f"  abstracts in {summary.label_a:<9}: {_coverage(summary.abstracts_a, summary.count_a)}",
        f"  abstracts in {summary.label_b:<9}: {_coverage(summary.abstracts_b, summary.count_b)}",
    ]
    return "\n".join(lines)
