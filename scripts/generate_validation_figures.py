#!/usr/bin/env python3
"""Generate validation tables/plots comparing the manual set vs AI-assisted results.

Outputs (to corpora/wireless_v1/output/figures/):
  1. validation_table.tex      — paper & dataset precision/recall/Jaccard per venue
  2. papers_pie.png/pdf        — papers by venue pie chart + overall stats
  3. reuse_histogram.png/pdf   — dataset reuse distribution, manual vs AI val set
  4. openness_table.tex        — dataset openness across the three sets
  5. openness_vs_adoption.tex  — openness split by reuse (once vs >1)
  6. stats.txt                 — all fill-in numbers for the slide text

Definitions:
  - Manual Validation Set: prof's curated sheet (SIGCOMM/IMC/NSDI), main track only
  - AI-Assisted Validation Set: pipeline results restricted to the same
    venue-years as the manual sheet
  - AI-Assisted Total Set: all pipeline results (all 5 venues, 2022-2025)

Manual reuse counts are recomputed by counting duplicate dataset names within
the sheet, NOT the sheet's "Number of Papers using Dataset" column (which
includes papers outside the corpus).
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "corpora/wireless_v1/results"
OUTPUT = REPO / "corpora/wireless_v1/output"
MANUAL = REPO / "corpora/wireless_v1/manual"
FIGDIR = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else OUTPUT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

VAL_VENUES = ("SIGCOMM", "IMC", "NSDI")


# ── normalization helpers ────────────────────────────────────────────────
def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def norm_ds_name(n: str) -> str:
    n = (n or "").lower()
    n = re.sub(r"\b(dataset|data|traces?|measurements?|database)\b", " ", n)
    return re.sub(r"[^a-z0-9]+", " ", n).strip()


def name_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_ds_name(a), norm_ds_name(b)).ratio()


# ── load manual sheets ───────────────────────────────────────────────────
def load_manual():
    with open(MANUAL / "manual_papers.csv", encoding="utf-8-sig") as f:
        papers = [
            p for p in csv.DictReader(f)
            if (p.get("Workshop") or "").strip().upper() != "Y"
            and (p.get("Paper Title") or "").strip()
        ]
    with open(MANUAL / "manual_datasets.csv", encoding="utf-8-sig") as f:
        datasets = []
        for d in csv.DictReader(f):
            name = (d.get("dataset name") or "").strip()
            avail = (d.get("Availaibility (open?)") or "").strip().lower()
            if not name or avail in ("33/132", "0.25"):  # summary rows
                continue
            datasets.append(d)
    return papers, datasets


# ── load AI results ──────────────────────────────────────────────────────
def load_ai_raw():
    """Per-paper AI results from raw JSONs: venue, year, title, key, datasets."""
    out = []
    for f in sorted(RESULTS.glob("*_raw.json")):
        if f.name.startswith(("master_", "consolidated_")):
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        for run in data.get("runs") or []:
            venue, year = run.get("venue", ""), run.get("year")
            for p in run.get("papers") or []:
                out.append({
                    "venue": venue, "year": int(year),
                    "title": p.get("title", ""),
                    "key": p.get("bibtex_key", ""),
                    "datasets": [
                        {
                            "name": ds.get("name", ""),
                            "availability": ds.get("availability"),
                        }
                        for ds in p.get("datasets") or []
                    ],
                })
    return out


def load_ai_consolidated():
    # Use the pdf-only consolidated CSV: dataset extraction is only run on
    # papers whose full PDF was available. The full set is still used for
    # paper classification (venue/year counts), but dataset-level stats
    # (reuse, openness, adoption) are restricted to pdf-available papers.
    with open(OUTPUT / "consolidated_datasets_pdf_only.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_db_availability():
    """canonical_name(lower) -> (availability_status, availability_url) from the corpus DB."""
    import sqlite3
    conn = sqlite3.connect(REPO / "corpora/wireless_v1/taxonomy.sqlite")
    rows = conn.execute(
        "SELECT canonical_name, availability_status, availability_url FROM datasets"
    ).fetchall()
    conn.close()
    return {r[0].lower().strip(): (r[1], r[2] or "") for r in rows}


# ── matching ─────────────────────────────────────────────────────────────
def match_papers(manual_titles: set[str], ai_titles: set[str]):
    inter = manual_titles & ai_titles
    return inter


def match_datasets(manual_ds: list[dict], ai_ds: list[dict], sim_threshold=0.75):
    """Greedy 1:1 matching between manual and AI dataset lists.

    Match if (a) share a bibtex key and name sim >= 0.45, or
             (b) name sim >= sim_threshold globally.
    Returns (matches, unmatched_manual, unmatched_ai).
    """
    candidates = []
    for i, m in enumerate(manual_ds):
        mkeys = {k.strip() for k in (m.get("bibtex citation key") or "").split(",") if k.strip()}
        for j, a in enumerate(ai_ds):
            s = name_sim(m["dataset name"], a["name"])
            shared_key = bool(mkeys & a["keys"])
            if (shared_key and s >= 0.45) or s >= sim_threshold:
                candidates.append((s + (0.5 if shared_key else 0), i, j))
    candidates.sort(reverse=True)
    used_m, used_a, matches = set(), set(), []
    for score, i, j in candidates:
        if i in used_m or j in used_a:
            continue
        used_m.add(i)
        used_a.add(j)
        matches.append((i, j, score))
    unmatched_m = [m for i, m in enumerate(manual_ds) if i not in used_m]
    unmatched_a = [a for j, a in enumerate(ai_ds) if j not in used_a]
    return matches, unmatched_m, unmatched_a


def prf(n_manual: int, n_ai: int, n_match: int):
    prec = n_match / n_ai if n_ai else 0.0
    rec = n_match / n_manual if n_manual else 0.0
    jac = n_match / (n_manual + n_ai - n_match) if (n_manual + n_ai - n_match) else 0.0
    return prec, rec, jac


def is_open(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v or "").strip().lower() in ("yes", "true", "open", "1")


def main():
    manual_papers, manual_datasets = load_manual()
    ai_raw = load_ai_raw()
    consolidated = load_ai_consolidated()
    db_avail = load_db_availability()
    with open(OUTPUT / "consolidated_papers.csv", encoding="utf-8") as f:
        all_papers = list(csv.DictReader(f))

    # venue-years present in manual sheet (defines the validation scope)
    # Exclude 2025: proceedings were not fully released at curation time.
    manual_papers = [p for p in manual_papers if int(p["Year"]) < 2025]
    manual_vy = {(p["Conference"].strip(), int(p["Year"])) for p in manual_papers}

    lines: list[str] = []
    say = lambda s="": (print(s), lines.append(s))

    say("=" * 70)
    say("VALIDATION SCOPE")
    say("=" * 70)
    say(f"Manual venue-years: {sorted(manual_vy)}")
    say(f"Manual papers (main track): {len(manual_papers)}")
    say(f"Manual datasets: {len(manual_datasets)}")

    # ══ 1. VALIDATION TABLE ══════════════════════════════════════════════
    table_rows = {"Papers": {}, "Datasets": {}}
    for venue in VAL_VENUES:
        vy_scope = {vy for vy in manual_vy if vy[0] == venue}

        m_papers = [p for p in manual_papers if (p["Conference"].strip(), int(p["Year"])) in vy_scope]
        m_titles = {norm_title(p["Paper Title"]) for p in m_papers}

        a_classified_papers = [
            p for p in all_papers
            if (p["Conference"].strip(), int(p["Year"])) in vy_scope
        ]
        a_titles = {norm_title(p["Paper Title"]) for p in a_classified_papers}

        inter = m_titles & a_titles
        prec, rec, jac = prf(len(m_titles), len(a_titles), len(inter))
        table_rows["Papers"][venue] = (len(m_titles), len(a_titles), prec, rec, jac)

        # datasets scoped to this venue
        m_keys = {(p.get("Bibtex Citation Key") or "").strip() for p in m_papers}
        m_ds = [
            d for d in manual_datasets
            if {k.strip() for k in (d.get("bibtex citation key") or "").split(",")} & m_keys
        ]
        a_papers = [p for p in ai_raw if (p["venue"], p["year"]) in vy_scope]
        a_ds = []
        for p in a_papers:
            for ds in p["datasets"]:
                a_ds.append({"name": ds["name"], "keys": {p["key"]}, "availability": ds["availability"]})
        matches, um, ua = match_datasets(m_ds, a_ds)
        prec, rec, jac = prf(len(m_ds), len(a_ds), len(matches))
        table_rows["Datasets"][venue] = (len(m_ds), len(a_ds), prec, rec, jac)

    # averages
    for level in table_rows:
        vals = list(table_rows[level].values())
        n_m = sum(v[0] for v in vals)
        n_a = sum(v[1] for v in vals)
        avg = tuple(sum(v[i] for v in vals) / len(vals) for i in (2, 3, 4))
        table_rows[level]["Average"] = (n_m, n_a, *avg)

    say("")
    say("=" * 70)
    say("1. VALIDATION TABLE")
    say("=" * 70)
    say(f"{'Level':<10}{'Venue':<10}{'Manual':>8}{'AI':>6}{'Prec':>8}{'Rec':>8}{'Jacc':>8}")
    for level, venues in table_rows.items():
        for venue, (nm, na, p, r, j) in venues.items():
            say(f"{level:<10}{venue:<10}{nm:>8}{na:>6}{p:>8.2f}{r:>8.2f}{j:>8.2f}")

    # LaTeX
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Validation of the AI-assisted pipeline against the manually curated set (SIGCOMM, IMC, NSDI).}",
        r"\label{tab:validation}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r" & & \textbf{Manual Validation} & \textbf{AI-Assisted} & \textbf{Precision} & \textbf{Recall} & \textbf{Jaccard} \\",
        r" & & \textbf{Set (\# Papers)} & \textbf{Validation Set (\# Papers)} & & & \textbf{Index} \\",
        r"\midrule",
    ]
    for level, venues in table_rows.items():
        first = True
        for venue, (nm, na, p, r, j) in venues.items():
            lvl = rf"\multirow{{4}}{{*}}{{{level}}}" if first else ""
            bold = r"\textbf" if venue == "Average" else lambda s: s
            vname = rf"\textbf{{{venue}}}" if venue == "Average" else venue
            tex.append(rf"{lvl} & {vname} & {nm} & {na} & {p:.2f} & {r:.2f} & {j:.2f} \\")
            first = False
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"
    tex += [r"\end{tabular}", r"\end{table}"]
    (FIGDIR / "validation_table.tex").write_text("\n".join(tex))

    # ══ 1b. OVERALL STATS + PIE ══════════════════════════════════════════
    venue_counts = Counter(p["Conference"] for p in all_papers)
    total_papers = len(all_papers)
    total_datasets = len(consolidated)

    say("")
    say("=" * 70)
    say("1b. OVERALL STATS")
    say("=" * 70)
    say(f"Total papers (wireless-classified, all 5 venues 2022-2025): {total_papers}")
    say(f"Total unique datasets found: {total_datasets}")
    for v, c in venue_counts.most_common():
        say(f"  {v}: {c} ({100*c/total_papers:.1f}%)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = ["SIGCOMM", "IMC", "NSDI", "ICC", "TWC"]
        sizes = [venue_counts.get(v, 0) for v in order]
        colors = ["#1f5c7a", "#ed7d31", "#2e7d32", "#29abe2", "#b939ad"]
        fig, ax = plt.subplots(figsize=(8, 7))

        import math
        # Only show percentage on slices > 3%, annotate others externally
        def autopct_fn(pct):
            return f"{pct:.0f}%" if pct >= 3 else ""

        wedges, texts, autotexts = ax.pie(
            sizes, labels=None, autopct=autopct_fn, startangle=140,
            colors=colors, pctdistance=0.75,
            textprops={"fontsize": 20},
        )
        for at in autotexts:
            at.set_fontsize(20)
        # For small slices, add spaced-out external annotations
        small_slices = [(i, w, s) for i, (w, s) in enumerate(zip(wedges, sizes))
                        if 100 * s / total_papers < 3]
        if small_slices:
            spread = 22  # degrees between annotations
            base_angle = sum((w.theta2 + w.theta1) / 2 for _, w, _ in small_slices) / len(small_slices)
            for idx, (i, wedge, size) in enumerate(small_slices):
                pct = 100 * size / total_papers
                mid = (wedge.theta2 + wedge.theta1) / 2
                offset = (idx - (len(small_slices) - 1) / 2) * spread
                ann_angle = base_angle + offset
                x = 1.6 * math.cos(math.radians(ann_angle))
                y = 1.6 * math.sin(math.radians(ann_angle))
                ax.annotate(
                    f"{order[i]}: {pct:.1f}%",
                    xy=(0.9 * math.cos(math.radians(mid)),
                        0.9 * math.sin(math.radians(mid))),
                    xytext=(x, y), fontsize=18,
                    ha="left" if x > 0 else "right", va="center",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.7),
                )
        ax.legend(wedges, [f"{v} ({venue_counts.get(v,0)})" for v in order],
                  loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=18)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIGDIR / f"papers_pie.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        say("  -> papers_pie.png/pdf")
    except ImportError:
        say("  [matplotlib unavailable; skipped pie]")

    # ══ 2. DATASET REUSE ═════════════════════════════════════════════════
    # Manual reuse: count duplicate dataset names within the sheet,
    # but ONLY datasets whose bibtex key matches a manual paper (exclude
    # datasets the curator included from papers outside the corpus).
    m_paper_keys = {(p.get("Bibtex Citation Key") or "").strip() for p in manual_papers}
    m_paper_keys.discard("")
    manual_datasets_scoped = [
        d for d in manual_datasets
        if {k.strip() for k in (d.get("bibtex citation key") or "").split(",") if k.strip()}
        & m_paper_keys
    ]
    m_name_counts = Counter(norm_ds_name(d["dataset name"]) for d in manual_datasets_scoped)
    manual_reuse = Counter(m_name_counts.values())

    # AI validation set reuse: consolidated datasets whose keys are all/any in
    # SIGCOMM/IMC/NSDI papers, counting only keys from those venues.
    # Use the papers CSV (which has disambiguated keys) rather than raw JSONs.
    val_keys = {
        p["Bibtex Citation Key"] for p in all_papers
        if (p["Conference"].strip(), int(p["Year"])) in manual_vy
    }
    ai_val_reuse_counts = []
    for d in consolidated:
        keys = [k.strip() for k in d["Bibtex Citation Keys"].split(",") if k.strip()]
        n = sum(1 for k in keys if k in val_keys)
        if n > 0:
            ai_val_reuse_counts.append(n)
    ai_val_reuse = Counter(ai_val_reuse_counts)

    # AI total set reuse
    ai_total_reuse = Counter(int(d["Reuse Count"]) for d in consolidated)

    say("")
    say("=" * 70)
    say("2. DATASET REUSE DISTRIBUTIONS")
    say("=" * 70)
    say(f"{'Times used':>10} | {'Manual':>7} | {'AI-val':>7} | {'AI-total':>8}")
    all_ns = sorted(set(manual_reuse) | set(ai_val_reuse) | set(ai_total_reuse))
    for n in all_ns:
        say(f"{n:>10} | {manual_reuse.get(n,0):>7} | {ai_val_reuse.get(n,0):>7} | {ai_total_reuse.get(n,0):>8}")

    def reuse_summary(counter, label):
        total = sum(counter.values())
        multi = sum(c for n, c in counter.items() if n > 1)
        say(f"  {label}: {total} datasets, {multi} ({100*multi/total:.1f}%) used more than once")
        return total, multi

    say("")
    say("2b. Total dataset reuse:")
    m_tot, m_multi = reuse_summary(manual_reuse, "Manual validation set")
    v_tot, v_multi = reuse_summary(ai_val_reuse, "AI-assisted validation set")
    t_tot, t_multi = reuse_summary(ai_total_reuse, "AI-assisted total set")

    say("")
    say("Most reused datasets (AI total set):")
    top = sorted(consolidated, key=lambda d: -int(d["Reuse Count"]))[:5]
    for d in top:
        say(f"  {d['Canonical Name']}: {d['Reuse Count']} papers")
    say("Most reused datasets (manual):")
    for nm, c in m_name_counts.most_common(5):
        if c > 1:
            say(f"  {nm}: {c} papers")

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # ── Validation histogram: manual vs AI-val (matching the mockup) ──
        max_n_val = max(max(manual_reuse.keys(), default=1), max(ai_val_reuse.keys(), default=1))
        ns = list(range(1, max_n_val + 1))
        x = np.arange(len(ns))
        w = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - w / 2, [manual_reuse.get(n, 0) for n in ns], w,
               label="Manual Validation Set", color="#1f5c7a")
        ax.bar(x + w / 2, [ai_val_reuse.get(n, 0) for n in ns], w,
               label="AI-Assisted Validation Set", color="#ed7d31")
        for i, n in enumerate(ns):
            for dx, cnt in ((-w / 2, manual_reuse.get(n, 0)), (w / 2, ai_val_reuse.get(n, 0))):
                if cnt:
                    ax.text(i + dx, cnt + 0.5, str(cnt), ha="center", fontsize=22, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(ns, fontsize=20)
        ax.set_xlabel("Number of Times Dataset Used", fontsize=24)
        ax.set_ylabel("Number of Datasets", fontsize=24)
        ax.tick_params(axis="y", labelsize=20)
        ax.legend(frameon=False, fontsize=20)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIGDIR / f"reuse_histogram_validation.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # ── Full corpus histogram: AI-total set with broken y-axis ──
        # Show ALL integers from 1 to max on x-axis (not just non-zero values)
        max_n_full = max(ai_total_reuse.keys(), default=1)
        ns_full = list(range(1, max_n_full + 1))
        counts_full = [ai_total_reuse.get(n, 0) for n in ns_full]

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, figsize=(10, 6),
                                              gridspec_kw={"height_ratios": [1, 2.5], "hspace": 0.08})
        bar_color = "#ed7d31"
        for ax in (ax_top, ax_bot):
            bars = ax.bar(range(len(ns_full)), counts_full, color=bar_color, edgecolor="white", linewidth=0.5)
        # Set limits for break
        max_count = max(counts_full)
        second_max = sorted(counts_full, reverse=True)[1] if len(counts_full) > 1 else max_count
        break_lo = min(second_max + 5, max_count - 10)
        break_hi = max_count - 5
        ax_top.set_ylim(break_hi, max_count + 15)
        ax_bot.set_ylim(0, break_lo)

        # Add count labels on all bars
        for ax in (ax_top, ax_bot):
            for i, (n, c) in enumerate(zip(ns_full, counts_full)):
                if c > 0:
                    # Only annotate on the axis that shows the bar
                    if c >= break_hi:
                        if ax is ax_top:
                            ax.text(i, c + 1, str(c), ha="center", fontsize=18, fontweight="bold")
                    elif c <= break_lo:
                        if ax is ax_bot:
                            ax.text(i, c + 0.5, str(c), ha="center", fontsize=18, fontweight="bold")

        # Diagonal break marks
        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.tick_params(bottom=False, labelsize=16)
        ax_bot.tick_params(labelsize=16)
        for sp in ["top", "right"]:
            ax_top.spines[sp].set_visible(False)
            ax_bot.spines[sp].set_visible(False)
        d = 0.01
        kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=0.8)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax_bot.transAxes)
        ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        ax_bot.set_xticks(range(len(ns_full)))
        ax_bot.set_xticklabels(ns_full, fontsize=16)
        ax_bot.set_xlabel("Number of Times Dataset Used", fontsize=18)
        fig.text(0.02, 0.5, "Number of Datasets", va="center", rotation="vertical", fontsize=24)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIGDIR / f"reuse_histogram.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        say("  -> reuse_histogram.png/pdf, reuse_histogram_validation.png/pdf")
    except ImportError:
        pass

    # ══ 3. DATASET OPENNESS ══════════════════════════════════════════════
    # Manual (scoped to papers in the corpus only)
    manual_open_by_name = defaultdict(bool)
    for d in manual_datasets_scoped:
        manual_open_by_name[norm_ds_name(d["dataset name"])] |= is_open(d.get("Availaibility (open?)"))
    m_open = sum(manual_open_by_name.values())

    # AI: availability from DB — strict mode: only count as open if the DB
    # has both status=open AND a non-empty availability_url (verified link).
    def ai_open_status(canonical_row):
        variants = [canonical_row["Canonical Name"]] + [
            v.strip() for v in (canonical_row["All Name Variants"] or "").split(";") if v.strip()
        ]
        for v in variants:
            entry = db_avail.get(v.lower().strip())
            if entry is None:
                continue
            status, url = entry
            if status == "open" and url:
                return True
            if status == "closed":
                return False
        return False  # unknown or open-without-URL counted as not-open

    # Raw LLM availability: what the LLM originally extracted (before verification)
    def ai_raw_open_status(canonical_row):
        """Check the raw JSON extraction for this dataset's availability."""
        name = canonical_row["Canonical Name"]
        variants = {name.lower().strip()} | {
            v.strip().lower() for v in (canonical_row["All Name Variants"] or "").split(";") if v.strip()
        }
        for p in ai_raw:
            for ds in p["datasets"]:
                if ds["name"].lower().strip() in variants:
                    avail = ds.get("availability")
                    if isinstance(avail, bool):
                        return avail
                    if str(avail or "").lower().strip() in ("open", "yes", "true"):
                        return True
        return False

    ai_val_rows = []
    for d in consolidated:
        keys = [k.strip() for k in d["Bibtex Citation Keys"].split(",") if k.strip()]
        n_val = sum(1 for k in keys if k in val_keys)
        if n_val > 0:
            ai_val_rows.append((d, n_val))

    v_open = sum(1 for d, _ in ai_val_rows if ai_open_status(d))
    t_open = sum(1 for d in consolidated if ai_open_status(d))

    # Raw LLM counts (before human verification)
    v_open_raw = sum(1 for d, _ in ai_val_rows if ai_raw_open_status(d))
    t_open_raw = sum(1 for d in consolidated if ai_raw_open_status(d))

    say("")
    say("=" * 70)
    say("3. DATASET OPENNESS")
    say("=" * 70)
    say(f"{'':<22}{'Manual':>10}{'AI-val':>10}{'AI-total':>10}")
    say(f"{'Total datasets':<22}{m_tot:>10}{len(ai_val_rows):>10}{t_tot:>10}")
    say(f"{'Publicly available':<22}{m_open:>10}{v_open:>10}{t_open:>10}")
    say(f"{'Percentage':<22}{100*m_open/m_tot:>9.1f}%{100*v_open/len(ai_val_rows):>9.1f}%{100*t_open/t_tot:>9.1f}%")
    say("")
    say("  Raw LLM (before human verification):")
    say(f"{'  Publicly available':<22}{'':<10}{v_open_raw:>10}{t_open_raw:>10}")
    say(f"{'  Percentage':<22}{'':<10}{100*v_open_raw/len(ai_val_rows):>9.1f}%{100*t_open_raw/t_tot:>9.1f}%")
    say(f"  LLM over-estimation: +{100*t_open_raw/t_tot - 100*t_open/t_tot:.1f} pp")
    say(f"  LLM accuracy (vs verified): {100*(t_tot - abs(t_open_raw - t_open))/t_tot:.1f}%")
    # Per-dataset accuracy: how many did the LLM get right?
    correct = 0
    for d in consolidated:
        llm_says_open = ai_raw_open_status(d)
        verified_open = ai_open_status(d)
        if llm_says_open == verified_open:
            correct += 1
    say(f"  Per-dataset agreement (LLM vs verified): {correct}/{t_tot} ({100*correct/t_tot:.1f}%)")

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Dataset openness across the manual and AI-assisted sets.}",
        r"\label{tab:openness}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \textbf{Manual} & \multicolumn{2}{c}{\textbf{AI-Assisted Validation Set}} & \textbf{AI-Assisted} \\",
        r"\cmidrule(lr){3-4}",
        r" & \textbf{Validation Set} & \textbf{Raw LLM} & \textbf{Verified} & \textbf{Total (Verified)} \\",
        r"\midrule",
        rf"Total datasets & {m_tot} & {len(ai_val_rows)} & {len(ai_val_rows)} & {t_tot} \\",
        rf"Publicly-available & {m_open} & {v_open_raw} & {v_open} & {t_open} \\",
        rf"Percentage & {100*m_open/m_tot:.0f}\% & {100*v_open_raw/len(ai_val_rows):.0f}\% & {100*v_open/len(ai_val_rows):.0f}\% & {100*t_open/t_tot:.0f}\% \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (FIGDIR / "openness_table.tex").write_text("\n".join(tex))

    # ══ 4. OPENNESS vs ADOPTION ══════════════════════════════════════════
    # manual: openness by reuse group (name-count within sheet)
    def manual_openness_by_reuse():
        groups = {"once": [0, 0], "multi": [0, 0]}  # [open, total]
        seen = set()
        for d in manual_datasets_scoped:
            nm = norm_ds_name(d["dataset name"])
            if nm in seen:
                continue
            seen.add(nm)
            g = "multi" if m_name_counts[nm] > 1 else "once"
            groups[g][1] += 1
            if is_open(d.get("Availaibility (open?)")):
                groups[g][0] += 1
        return groups

    def ai_openness_by_reuse(rows):
        groups = {"once": [0, 0], "multi": [0, 0]}
        for d, n in rows:
            g = "multi" if n > 1 else "once"
            groups[g][1] += 1
            if ai_open_status(d):
                groups[g][0] += 1
        return groups

    mg = manual_openness_by_reuse()
    vg = ai_openness_by_reuse(ai_val_rows)
    tg = ai_openness_by_reuse([(d, int(d["Reuse Count"])) for d in consolidated])

    pct = lambda o, t: f"{100*o/t:.0f}\\%" if t else "--"
    pct_plain = lambda o, t: f"{100*o/t:.0f}%" if t else "--"

    say("")
    say("=" * 70)
    say("4. OPENNESS vs ADOPTION (% publicly available)")
    say("=" * 70)
    say(f"{'':<24}{'Manual':>10}{'AI-val':>10}{'AI-total':>10}")
    say(f"{'Used once':<24}{pct_plain(*mg['once']):>10}{pct_plain(*vg['once']):>10}{pct_plain(*tg['once']):>10}")
    say(f"{'Used more than once':<24}{pct_plain(*mg['multi']):>10}{pct_plain(*vg['multi']):>10}{pct_plain(*tg['multi']):>10}")
    say(f"(counts: manual once={mg['once']}, multi={mg['multi']}; "
        f"AI-val once={vg['once']}, multi={vg['multi']}; "
        f"AI-total once={tg['once']}, multi={tg['multi']})  [open, total]")

    # conversely: % of open datasets that are multi-use
    def multi_share_of_open(groups):
        open_total = groups["once"][0] + groups["multi"][0]
        return 100 * groups["multi"][0] / open_total if open_total else 0.0

    say("")
    say(f"Of the publicly-available datasets, multi-use share:")
    say(f"  Manual: {multi_share_of_open(mg):.0f}%")
    say(f"  AI-val: {multi_share_of_open(vg):.0f}%")
    say(f"  AI-total: {multi_share_of_open(tg):.0f}%")

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Openness vs.\ adoption: percentage of datasets that were publicly available, by reuse.}",
        r"\label{tab:openness-adoption}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{\textbf{Percentage of Datasets that were Publicly Available}} \\",
        r"\cmidrule(lr){2-4}",
        r" & \textbf{Manual} & \textbf{AI-Assisted} & \textbf{AI-Assisted} \\",
        r" & \textbf{Validation Set} & \textbf{Validation Set} & \textbf{Total Set} \\",
        r"\midrule",
        rf"Datasets used once & {pct(*mg['once'])} & {pct(*vg['once'])} & {pct(*tg['once'])} \\",
        rf"Datasets used more than once & {pct(*mg['multi'])} & {pct(*vg['multi'])} & {pct(*tg['multi'])} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (FIGDIR / "openness_vs_adoption.tex").write_text("\n".join(tex))

    (FIGDIR / "stats.txt").write_text("\n".join(lines))
    say("")
    say(f"All outputs written to {FIGDIR}/")


if __name__ == "__main__":
    main()
