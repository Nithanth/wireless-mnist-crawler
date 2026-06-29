"""DB-free, snapshot eval: score a classified CSV against a gold sheet.

The overlap scoring is a pure, point-in-time computation — it needs neither the
SQLite DB nor the network. This module wires the existing matcher
(:func:`wireless_taxonomy.eval.overlap.match`) and metrics directly to two files:

* a **classified CSV** — the full labelled set emitted by ``classify --csv``
  (columns ``title``, ``doi``, ``venue``, ``year``, ``label`` are used). Because
  the CSV carries *every* paper with its yes/maybe/no ``label``, this single file
  gives both the predicted-positive set (rows whose label passes ``pass_mode``)
  **and** the proceedings universe (all rows);
* a **gold sheet** — the curated CSV/XLSX, read with the same schema-tolerant
  :class:`~wireless_taxonomy.ingest.gold.GoldSheetReader` the CLI uses.

Both sides are grouped by ``(venue, year)`` and matched DOI → exact title →
fuzzy title. Only conferences present in the classified CSV(s) are scored; gold
rows for venue-years that were never classified are ignored (and counted), so an
unrun venue is not penalised as a wall of false negatives.

``drop_workshops`` works here too, purely from the files: a curated paper that
matches **no** row in the classified universe is treated as a co-located
workshop paper (absent from the main proceedings) and dropped from the
denominator rather than counted as a classifier miss. This requires the full
labelled universe, so it is only allowed when the CSV carries a ``label`` column.
"""

import csv as _csv
from pathlib import Path
from typing import Any

from wireless_taxonomy.eval.overlap import PaperRef, _row_with_metrics, aggregate, match, to_markdown
from wireless_taxonomy.ingest.gold import GoldSheetReader

_PRED_TITLE_KEYS = ("title", "paper title", "name", "paper")
_PRED_DOI_KEYS = ("doi", "doi version of key", "doi url")
_PRED_VENUE_KEYS = ("venue", "conference", "conf")
_PRED_YEAR_KEYS = ("year", "yr")
_PRED_LABEL_KEYS = ("label", "wireless_label", "prediction")

_PASS_LABELS = {"high": {"yes"}, "low": {"yes", "maybe"}}


def _pick(lower_row: dict[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in lower_row and (lower_row[key] or "").strip():
            return lower_row[key].strip()
    return ""


def _vy_key(venue: str, year: Any) -> tuple[str, str]:
    return (str(venue).strip().lower(), str(year).strip())


class _ClassifiedRow:
    __slots__ = ("ref", "label")

    def __init__(self, ref: PaperRef, label: str | None) -> None:
        self.ref = ref
        self.label = label


def read_classified_csv(
    path: str,
) -> tuple[dict[tuple[str, str], list[_ClassifiedRow]], dict[tuple[str, str], tuple[str, str]], bool]:
    """Read a classified CSV into per-(venue, year) rows.

    Returns ``(rows_by_key, display_labels, has_label_column)``. Each row keeps
    its :class:`PaperRef` and its raw ``label`` (or ``None`` when the file has no
    label column). ``display_labels`` preserves the original venue/year strings
    for reporting.
    """
    rows: dict[tuple[str, str], list[_ClassifiedRow]] = {}
    display: dict[tuple[str, str], tuple[str, str]] = {}
    has_label = False
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        reader = _csv.DictReader(fh)
        header_lower = {(c or "").strip().lower() for c in (reader.fieldnames or [])}
        has_label = any(k in header_lower for k in _PRED_LABEL_KEYS)
        for i, row in enumerate(reader):
            lower = {str(k).strip().lower(): (v or "") for k, v in row.items() if k is not None}
            title = _pick(lower, _PRED_TITLE_KEYS)
            if not title:
                continue
            venue = _pick(lower, _PRED_VENUE_KEYS)
            year = _pick(lower, _PRED_YEAR_KEYS)
            doi = _pick(lower, _PRED_DOI_KEYS)
            label = _pick(lower, _PRED_LABEL_KEYS).lower() or None
            ref = PaperRef.build(key=f"{path}:{i}", title=title, doi=doi or None)
            key = _vy_key(venue, year)
            rows.setdefault(key, []).append(_ClassifiedRow(ref, label))
            display.setdefault(key, (venue, year))
    return rows, display, has_label


def read_gold(
    paths: list[str],
    default_venue: str | None = None,
    default_year: int | None = None,
) -> tuple[dict[tuple[str, str], list[PaperRef]], dict[tuple[str, str], tuple[str, str]]]:
    """Group curated gold papers from sheet(s) by (venue, year)."""
    groups: dict[tuple[str, str], list[PaperRef]] = {}
    display: dict[tuple[str, str], tuple[str, str]] = {}
    for path in paths:
        for rec in GoldSheetReader(path, default_venue, default_year).read():
            ref = PaperRef.build(
                key=f"{rec.venue}:{rec.year}:{rec.normalized_title}",
                title=rec.title,
                doi=rec.doi,
            )
            key = _vy_key(rec.venue, rec.year)
            groups.setdefault(key, []).append(ref)
            display.setdefault(key, (rec.venue, str(rec.year)))
    return groups, display


def eval_files(
    classified_csv_paths: list[str],
    gold_paths: list[str],
    *,
    pass_mode: str = "high",
    drop_workshops: bool = False,
    fuzzy_threshold: float = 0.92,
    gold_default_venue: str | None = None,
    gold_default_year: int | None = None,
    exclude: list[tuple[str, Any]] | None = None,
    min_gold: int = 0,
) -> dict[str, Any]:
    """Score classified CSV(s) against gold sheet(s) with no DB or network.

    ``pass_mode`` selects the predicted-positive set from the ``label`` column
    (``high`` = ``yes`` only; ``low`` = ``yes``|``maybe``). ``drop_workshops``
    drops curated papers absent from the classified universe from the
    denominator. Returns a report dict shaped like the old ``eval-overlap``
    output so it renders with the shared
    :func:`~wireless_taxonomy.eval.overlap.to_markdown`.

    ``exclude`` is a list of ``(venue, year)`` pairs to pull out of the headline,
    and ``min_gold`` pulls out any venue-year whose curated gold set is smaller
    than ``min_gold`` papers. Both go to a separate ``under_curated`` bucket
    (with their would-be metrics, for visibility) instead of dragging the
    aggregate — the intended use is thinly- or stale-curated venue-years (e.g. a
    conference recorded before its papers were released) that punish precision
    despite the classifier being correct.
    """
    if pass_mode not in _PASS_LABELS:
        raise ValueError("pass_mode must be 'high' or 'low'")
    if min_gold < 0:
        raise ValueError("min_gold must be >= 0")
    positive_labels = _PASS_LABELS[pass_mode]
    exclude_set = {_vy_key(v, y) for v, y in (exclude or [])}

    universe: dict[tuple[str, str], list[PaperRef]] = {}
    predicted: dict[tuple[str, str], list[PaperRef]] = {}
    display: dict[tuple[str, str], tuple[str, str]] = {}
    files_have_labels = True

    for path in classified_csv_paths:
        rows_by_key, file_display, has_label = read_classified_csv(path)
        files_have_labels = files_have_labels and has_label
        for key, rows in rows_by_key.items():
            display.setdefault(key, file_display[key])
            for crow in rows:
                universe.setdefault(key, []).append(crow.ref)
                is_positive = crow.label in positive_labels if crow.label is not None else True
                if is_positive:
                    predicted.setdefault(key, []).append(crow.ref)

    if drop_workshops and not files_have_labels:
        raise ValueError(
            "drop_workshops needs the full classified universe; the classified CSV "
            "has no 'label' column. Re-export with `classify --csv` (which writes "
            "every paper + label), or score with drop_workshops=False."
        )

    gold, gold_display = read_gold(gold_paths, gold_default_venue, gold_default_year)
    for key, label in gold_display.items():
        display.setdefault(key, label)

    instance_rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    ignored_gold_instances: list[dict[str, Any]] = []
    under_curated_instances: list[dict[str, Any]] = []

    for key in sorted(universe, key=lambda k: (k[0], k[1])):
        venue, year = display.get(key, (key[0], key[1]))
        pred_refs = predicted.get(key, [])
        gold_refs = gold.get(key, [])
        universe_refs = universe.get(key, [])

        result = match(pred_refs, gold_refs, fuzzy_threshold=fuzzy_threshold)
        # Split missed gold papers into classifier misses vs proceedings gaps.
        in_universe = match(result.unmatched_b, universe_refs, fuzzy_threshold=fuzzy_threshold)
        fn_missed = len(in_universe.matched)
        fn_missing_from_universe = len(in_universe.unmatched_a)

        counts = {
            "venue": venue,
            "year": year,
            "tp": len(result.matched),
            "fp": len(result.unmatched_a),
            "fn": len(result.unmatched_b),
            "fn_missed": fn_missed,
            "fn_missing_from_universe": fn_missing_from_universe,
        }
        # Discrepancy detail is always kept (even for excluded venue-years) so
        # the per-conference FP/FN lists stay visible for inspection.
        mismatches.append(
            {
                "venue": venue,
                "year": year,
                "false_positives": [r.title for r in result.unmatched_a],
                "false_negatives_classifier_miss": [b.title for _, b in in_universe.matched],
                "false_negatives_missing_from_universe": [r.title for r in in_universe.unmatched_a],
            }
        )

        reason = _under_curated_reason(key, len(gold_refs), exclude_set, min_gold)
        if reason is not None:
            scored = _row_with_metrics(counts, scope_to_universe=drop_workshops)
            scored["gold_papers"] = len(gold_refs)
            scored["reason"] = reason
            under_curated_instances.append(scored)
            continue
        instance_rows.append(counts)

    # Gold venue-years that were never classified: report, don't penalise.
    for key in sorted(gold):
        if key not in universe:
            venue, year = display.get(key, (key[0], key[1]))
            ignored_gold_instances.append(
                {"venue": venue, "year": year, "gold_papers": len(gold[key])}
            )

    agg = aggregate(instance_rows, scope_to_universe=drop_workshops)
    report: dict[str, Any] = {
        "classifier": "file",
        "pass_mode": pass_mode,
        "fuzzy_threshold": fuzzy_threshold,
        "scope_to_universe": drop_workshops,
        "overall": agg["overall"],
        "per_conference": agg["per_conference"],
        "instances": agg["per_conference_year"],
        "mismatches": mismatches,
        "ignored_gold_instances": ignored_gold_instances,
        "under_curated_instances": under_curated_instances,
        "min_gold": min_gold,
    }
    return report


def _under_curated_reason(
    key: tuple[str, str],
    gold_count: int,
    exclude_set: set[tuple[str, str]],
    min_gold: int,
) -> str | None:
    """Why a classified venue-year is pulled from the headline, or None to keep it."""
    if key in exclude_set:
        return "excluded"
    if min_gold > 0 and gold_count < min_gold:
        return f"under-curated (<{min_gold} gold papers)"
    return None


def eval_files_to_outputs(
    classified_csv_paths: list[str],
    gold_paths: list[str],
    *,
    json_out: str | None = None,
    md_out: str | None = None,
    pass_mode: str = "high",
    drop_workshops: bool = False,
    fuzzy_threshold: float = 0.92,
    gold_default_venue: str | None = None,
    gold_default_year: int | None = None,
    exclude: list[tuple[str, Any]] | None = None,
    min_gold: int = 0,
) -> dict[str, Any]:
    """Run :func:`eval_files` and optionally write JSON and/or Markdown reports."""
    import json as _json

    report = eval_files(
        classified_csv_paths,
        gold_paths,
        pass_mode=pass_mode,
        drop_workshops=drop_workshops,
        fuzzy_threshold=fuzzy_threshold,
        gold_default_venue=gold_default_venue,
        gold_default_year=gold_default_year,
        exclude=exclude,
        min_gold=min_gold,
    )
    if json_out:
        path = Path(json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if md_out:
        path = Path(md_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(to_markdown(report), encoding="utf-8")
    return report
