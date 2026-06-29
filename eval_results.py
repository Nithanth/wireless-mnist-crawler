#!/usr/bin/env python3
"""Evaluate LLM pipeline results against manual curation.

Compares papers and datasets from the pipeline's raw JSON output against
a manually curated Google Sheets CSV export. Produces paper-level recall/precision,
dataset extraction recall, and an over-retrieval analysis.

Usage:
    python eval_results.py --manual /path/to/manual_papers.csv --results ./src/results
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def norm_title(t: str) -> str:
    """Normalize a title for fuzzy matching."""
    t = t.lower().strip()
    t = re.sub(r"[^a-z0-9 ]", "", t)
    return re.sub(r"\s+", " ", t).strip()[:60]


def load_manual(path: str) -> dict[str, dict]:
    """Load manual CSV. Returns {norm_title -> paper_info}."""
    papers = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            venue = (row.get("Conference") or "").strip().upper()
            year = (row.get("Year") or "").strip()
            ws = (row.get("Workshop") or "").strip().upper()
            title = (row.get("Paper Title") or "").strip()
            key = (row.get("Bibtex Citation Key") or "").strip()
            datasets = (row.get("Datasets") or "").strip()

            if venue not in ("SIGCOMM", "IMC", "NSDI"):
                continue
            if year not in ("2022", "2023", "2024"):
                continue

            ds_list = [d.strip() for d in datasets.split(",") if d.strip()] if datasets else []
            papers[norm_title(title)] = {
                "key": key,
                "title": title,
                "venue": venue,
                "year": year,
                "datasets": ds_list,
                "workshop": ws == "Y",
            }
    return papers


def load_llm(results_dir: str) -> dict[str, dict]:
    """Load LLM pipeline results from raw JSON. Returns {norm_title -> paper_info}."""
    papers = {}
    for f in sorted(Path(results_dir).glob("*_raw.json")):
        if "master" in f.name:
            continue
        data = json.loads(f.read_text())
        for run in data.get("runs", []):
            venue = run.get("venue", "")
            year = run.get("year", "")
            for paper in run.get("papers", []):
                title = paper.get("title", "")
                papers[norm_title(title)] = {
                    "key": paper.get("bibtex_key", ""),
                    "title": title,
                    "venue": venue,
                    "year": str(year),
                    "datasets": [d["name"] for d in paper.get("datasets", []) if d.get("name")],
                    "source": paper.get("extraction_source", ""),
                    "error": paper.get("error", ""),
                }
    return papers


def evaluate(manual: dict, llm: dict) -> dict:
    """Run the full evaluation, returns structured results."""
    # Separate workshop from main-track
    manual_ws = {k: v for k, v in manual.items() if v["workshop"]}
    manual_main = {k: v for k, v in manual.items() if not v["workshop"]}

    m_titles = set(manual_main.keys())
    l_titles = set(llm.keys())
    matched = m_titles & l_titles
    manual_only = m_titles - l_titles
    llm_only = l_titles - m_titles

    # Paper-level metrics
    precision = len(matched) / len(l_titles) if l_titles else 0
    recall = len(matched) / len(m_titles) if m_titles else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    jaccard = len(matched) / len(m_titles | l_titles) if (m_titles | l_titles) else 0

    # Dataset-level (on matched papers)
    both_ds = manual_has_only = llm_has_only = neither = 0
    ds_misses = []
    for t in matched:
        m_ds = manual_main[t]["datasets"]
        l_ds = llm[t]["datasets"]
        if m_ds and l_ds:
            both_ds += 1
        elif m_ds and not l_ds:
            manual_has_only += 1
            ds_misses.append({"manual": manual_main[t], "llm": llm[t]})
        elif l_ds and not m_ds:
            llm_has_only += 1
        else:
            neither += 1

    total_with_manual = both_ds + manual_has_only
    ds_precision = both_ds / (both_ds + llm_has_only) if (both_ds + llm_has_only) else 0
    ds_recall = both_ds / total_with_manual if total_with_manual else 0
    ds_f1 = 2 * ds_precision * ds_recall / (ds_precision + ds_recall) if (ds_precision + ds_recall) else 0

    # By extraction source
    by_source = defaultdict(lambda: {"total": 0, "manual_has": 0, "both": 0})
    for t in matched:
        src = llm[t]["source"]
        m_ds = manual_main[t]["datasets"]
        l_ds = llm[t]["datasets"]
        by_source[src]["total"] += 1
        if m_ds:
            by_source[src]["manual_has"] += 1
        if m_ds and l_ds:
            by_source[src]["both"] += 1

    # Over-retrieval analysis: categorize LLM-only papers
    llm_only_by_venue = Counter()
    for t in llm_only:
        llm_only_by_venue[f"{llm[t]['venue']} {llm[t]['year']}"] += 1

    # Manual-only analysis: why were they missed?
    manual_only_list = []
    for t in sorted(manual_only):
        m = manual_main[t]
        manual_only_list.append(m)

    return {
        "paper_level": {
            "manual_total": len(manual),
            "manual_workshop": len(manual_ws),
            "manual_main_track": len(manual_main),
            "llm_total": len(llm),
            "matched": len(matched),
            "manual_only": len(manual_only),
            "llm_only": len(llm_only),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "jaccard": jaccard,
        },
        "dataset_level": {
            "both_have_datasets": both_ds,
            "manual_has_llm_empty": manual_has_only,
            "llm_has_manual_empty": llm_has_only,
            "neither": neither,
            "precision": ds_precision,
            "recall": ds_recall,
            "f1": ds_f1,
            "misses": ds_misses,
        },
        "by_source": dict(by_source),
        "over_retrieval": {
            "count": len(llm_only),
            "by_venue": dict(llm_only_by_venue),
            "papers": [llm[t] for t in sorted(llm_only)],
        },
        "manual_misses": manual_only_list,
    }


def print_report(results: dict) -> None:
    """Print a formatted evaluation report."""
    p = results["paper_level"]
    d = results["dataset_level"]
    s = results["by_source"]
    o = results["over_retrieval"]

    print("=" * 70)
    print("  WIRELESS TAXONOMY PIPELINE EVALUATION")
    print("=" * 70)

    print(f"\n  Manual set: {p['manual_total']} papers")
    print(f"    Workshop (excluded):  {p['manual_workshop']}")
    print(f"    Main-track (eval'd):  {p['manual_main_track']}")
    print(f"  LLM pipeline output:    {p['llm_total']} papers")

    print(f"\n{'─' * 70}")
    print(f"  PAPER RECALL (does the pipeline find manually curated papers?)")
    print(f"{'─' * 70}")
    print(f"\n  Matched:      {p['matched']}/{p['manual_main_track']}")
    print(f"  Recall:       {p['recall']:.1%}")
    print(f"  F1:           {p['f1']:.3f}")
    print(f"  Jaccard:      {p['jaccard']:.3f}")

    if results["manual_misses"]:
        print(f"\n  Missed papers ({p['manual_only']}):")
        for m in results["manual_misses"]:
            print(f"    [{m['venue']} {m['year']}] {m['key']}: {m['title'][:52]}")

    print(f"\n{'─' * 70}")
    print(f"  DATASET EXTRACTION (on {p['matched']} matched papers)")
    print(f"{'─' * 70}")
    print(f"\n  Papers where both found datasets:  {d['both_have_datasets']}")
    print(f"  Manual has datasets, LLM empty:    {d['manual_has_llm_empty']}")
    print(f"  LLM found datasets, manual empty:  {d['llm_has_manual_empty']}")
    print(f"\n  Precision: {d['precision']:.1%}")
    print(f"  Recall:    {d['recall']:.1%}")
    print(f"  F1:        {d['f1']:.3f}")

    if d["misses"]:
        print(f"\n  Extraction misses ({d['manual_has_llm_empty']}):")
        for miss in d["misses"]:
            m, l = miss["manual"], miss["llm"]
            print(f"    {m['key']} ({l['source']}): {m['title'][:48]}")
            print(f"      Expected: {', '.join(m['datasets'])[:65]}")

    print(f"\n{'─' * 70}")
    print(f"  RECALL BY EXTRACTION SOURCE")
    print(f"{'─' * 70}")
    print(f"\n  {'Source':<20} {'Papers':<8} {'w/ Manual DS':<14} {'LLM Found':<11} {'Recall'}")
    for src in sorted(s.keys()):
        v = s[src]
        r = v["both"] / v["manual_has"] if v["manual_has"] else 0
        print(f"  {src:<20} {v['total']:<8} {v['manual_has']:<14} {v['both']:<11} {r:.0%}")

    print(f"\n{'─' * 70}")
    print(f"  OVER-RETRIEVAL ANALYSIS")
    print(f"{'─' * 70}")
    print(f"\n  LLM found {o['count']} papers not in manual set")
    print(f"  These are NOT false positives — they're genuinely wireless papers")
    print(f"  that the manual curation hasn't covered yet.")
    print(f"\n  By venue:")
    for vy, count in sorted(o["by_venue"].items()):
        print(f"    {vy}: {count}")
    if o["papers"]:
        print(f"\n  Papers ({min(15, len(o['papers']))} shown):")
        for p in o["papers"][:15]:
            ds_str = f" [{len(p['datasets'])} ds]" if p["datasets"] else ""
            print(f"    [{p['venue']} {p['year']}] {p['title'][:50]}{ds_str}")

    print(f"\n{'─' * 70}")
    print(f"  SUMMARY")
    print(f"{'─' * 70}")
    pr = results["paper_level"]
    dr = results["dataset_level"]
    print(f"""
  ┌────────────────────────────────┬──────────┐
  │ Paper Recall                   │ {pr['recall']:>6.1%}   │
  │ Paper F1                       │ {pr['f1']:>6.3f}   │
  │ Dataset Extraction Precision   │ {dr['precision']:>6.1%}   │
  │ Dataset Extraction Recall      │ {dr['recall']:>6.1%}   │
  │ Dataset Extraction F1          │ {dr['f1']:>6.3f}   │
  │ PDF Source Recall              │ {s.get('pdf',{}).get('both',0)}/{s.get('pdf',{}).get('manual_has',0):>3}    │
  │ Abstract Source Recall         │ {s.get('abstract',{}).get('both',0)}/{s.get('abstract',{}).get('manual_has',0):>3}    │
  └────────────────────────────────┴──────────┘
""")


def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline against manual curation")
    parser.add_argument("--manual", required=True, help="Path to manual curation CSV")
    parser.add_argument("--results", default="./src/results", help="Path to results directory")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of report")
    args = parser.parse_args()

    manual = load_manual(args.manual)
    llm = load_llm(args.results)
    results = evaluate(manual, llm)

    if args.json:
        # Strip non-serializable bits
        out = {
            "paper_level": results["paper_level"],
            "dataset_level": {k: v for k, v in results["dataset_level"].items() if k != "misses"},
            "by_source": results["by_source"],
            "over_retrieval": {"count": results["over_retrieval"]["count"], "by_venue": results["over_retrieval"]["by_venue"]},
        }
        json.dump(out, sys.stdout, indent=2)
    else:
        print_report(results)


if __name__ == "__main__":
    main()
