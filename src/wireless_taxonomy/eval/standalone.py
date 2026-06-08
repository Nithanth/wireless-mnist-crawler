"""DB-free, snapshot eval: score a classified CSV against a gold sheet.

The overlap scoring is a pure, point-in-time computation — it needs neither the
SQLite DB nor the network. This module wires the existing matcher
(:func:`wireless_taxonomy.eval.overlap.match`) and metrics directly to two files:

* a **classified CSV** — the wireless list emitted by ``classify-conference --csv``
  (columns ``title``, ``doi``, ``venue``, ``year`` are used);
* a **gold sheet** — the curated CSV/XLSX, read with the same schema-tolerant
  :class:`~wireless_taxonomy.ingest.gold.GoldSheetReader` the CLI uses.

Both sides are grouped by ``(venue, year)`` and matched DOI → exact title →
fuzzy title. Only conferences present in the classified CSV(s) are scored; gold
rows for venue-years that were never classified are ignored (and counted), so an
unrun venue is not penalised as a wall of false negatives.

Unlike ``eval-overlap``, this has **no** ``--drop-workshops`` option: deciding a
curated paper is a co-located workshop requires the full ingested proceedings
universe, which only the DB-backed flow has. Here every unmatched gold paper is a
false negative.
"""
from __future__ import annotations

import csv as _csv
from pathlib import Path
from typing import Any

from wireless_taxonomy.eval.overlap import PaperRef, aggregate, match, to_markdown
from wireless_taxonomy.ingest.gold import GoldSheetReader

_PRED_TITLE_KEYS = ("title", "paper title", "name", "paper")
_PRED_DOI_KEYS = ("doi", "doi version of key", "doi url")
_PRED_VENUE_KEYS = ("venue", "conference", "conf")
_PRED_YEAR_KEYS = ("year", "yr")


def _pick(lower_row: dict[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in lower_row and (lower_row[key] or "").strip():
            return lower_row[key].strip()
    return ""


def _vy_key(venue: str, year: Any) -> tuple[str, str]:
    return (str(venue).strip().lower(), str(year).strip())


def read_classified_csv(path: str) -> dict[tuple[str, str], list[PaperRef]]:
    """Group predicted wireless papers from a classified CSV by (venue, year)."""
    groups: dict[tuple[str, str], list[PaperRef]] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        reader = _csv.DictReader(fh)
        for i, row in enumerate(reader):
            lower = {str(k).strip().lower(): (v or "") for k, v in row.items() if k is not None}
            title = _pick(lower, _PRED_TITLE_KEYS)
            if not title:
                continue
            venue = _pick(lower, _PRED_VENUE_KEYS)
            year = _pick(lower, _PRED_YEAR_KEYS)
            doi = _pick(lower, _PRED_DOI_KEYS)
            ref = PaperRef.build(key=f"{path}:{i}", title=title, doi=doi or None)
            groups.setdefault(_vy_key(venue, year), []).append(ref)
    return groups


def read_gold(
    paths: list[str],
    default_venue: str | None = None,
    default_year: int | None = None,
) -> dict[tuple[str, str], list[PaperRef]]:
    """Group curated gold papers from sheet(s) by (venue, year)."""
    groups: dict[tuple[str, str], list[PaperRef]] = {}
    for path in paths:
        for rec in GoldSheetReader(path, default_venue, default_year).read():
            ref = PaperRef.build(key=f"{rec.venue}:{rec.year}:{rec.normalized_title}",
                                 title=rec.title, doi=rec.doi)
            groups.setdefault(_vy_key(rec.venue, rec.year), []).append(ref)
    return groups


def eval_files(
    classified_csv_paths: list[str],
    gold_paths: list[str],
    *,
    fuzzy_threshold: float = 0.92,
    gold_default_venue: str | None = None,
    gold_default_year: int | None = None,
) -> dict[str, Any]:
    """Score classified CSV(s) against gold sheet(s) with no DB or network.

    Returns a report dict shaped like the ``eval-overlap`` output (``overall``,
    ``per_conference``, ``instances``, ``mismatches``) so it renders with the
    shared :func:`~wireless_taxonomy.eval.overlap.to_markdown`.
    """
    predicted: dict[tuple[str, str], list[PaperRef]] = {}
    venue_year_labels: dict[tuple[str, str], tuple[str, str]] = {}
    for path in classified_csv_paths:
        for key, refs in read_classified_csv(path).items():
            predicted.setdefault(key, []).extend(refs)

    gold = read_gold(gold_paths, gold_default_venue, gold_default_year)

    # Preserve original venue/year strings for display from whichever side has them.
    for path in classified_csv_paths:
        with Path(path).open(newline="", encoding="utf-8-sig") as fh:
            for row in _csv.DictReader(fh):
                lower = {str(k).strip().lower(): (v or "") for k, v in row.items() if k is not None}
                venue = _pick(lower, _PRED_VENUE_KEYS)
                year = _pick(lower, _PRED_YEAR_KEYS)
                venue_year_labels.setdefault(_vy_key(venue, year), (venue, year))
    for gpath in gold_paths:
        for rec in GoldSheetReader(gpath, gold_default_venue, gold_default_year).read():
            venue_year_labels.setdefault(_vy_key(rec.venue, rec.year), (rec.venue, str(rec.year)))

    instance_rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    ignored_gold_instances: list[dict[str, Any]] = []

    for key in sorted(predicted, key=lambda k: (k[0], k[1])):
        venue, year = venue_year_labels.get(key, (key[0], key[1]))
        pred_refs = predicted.get(key, [])
        gold_refs = gold.get(key, [])
        result = match(pred_refs, gold_refs, fuzzy_threshold=fuzzy_threshold)
        tp, fp, fn = len(result.matched), len(result.unmatched_a), len(result.unmatched_b)
        instance_rows.append(
            {"venue": venue, "year": year, "tp": tp, "fp": fp, "fn": fn,
             "fn_missed": fn, "fn_missing_from_universe": 0}
        )
        mismatches.append(
            {
                "venue": venue,
                "year": year,
                "false_positives": [r.title for r in result.unmatched_a],
                "false_negatives_classifier_miss": [r.title for r in result.unmatched_b],
                "false_negatives_missing_from_universe": [],
            }
        )

    # Gold venue-years that were never classified: report, don't penalise.
    for key in sorted(gold):
        if key not in predicted:
            venue, year = venue_year_labels.get(key, (key[0], key[1]))
            ignored_gold_instances.append(
                {"venue": venue, "year": year, "gold_papers": len(gold[key])}
            )

    agg = aggregate(instance_rows, scope_to_universe=False)
    report: dict[str, Any] = {
        "classifier": "file",
        "pass_mode": "n/a",
        "fuzzy_threshold": fuzzy_threshold,
        "scope_to_universe": False,
        "overall": agg["overall"],
        "per_conference": agg["per_conference"],
        "instances": agg["per_conference_year"],
        "mismatches": mismatches,
        "ignored_gold_instances": ignored_gold_instances,
    }
    return report


def eval_files_to_outputs(
    classified_csv_paths: list[str],
    gold_paths: list[str],
    *,
    json_out: str | None = None,
    md_out: str | None = None,
    fuzzy_threshold: float = 0.92,
    gold_default_venue: str | None = None,
    gold_default_year: int | None = None,
) -> dict[str, Any]:
    """Run :func:`eval_files` and optionally write JSON and/or Markdown reports."""
    import json as _json

    report = eval_files(
        classified_csv_paths,
        gold_paths,
        fuzzy_threshold=fuzzy_threshold,
        gold_default_venue=gold_default_venue,
        gold_default_year=gold_default_year,
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
