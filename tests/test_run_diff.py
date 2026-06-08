import json

from wireless_taxonomy.evaluate.run_diff import (
    DIFF_COLUMNS,
    diff_paper_sets,
    format_diff_summary,
    load_paper_set,
    write_diff_csv,
    write_diff_report,
)


def _row(title, *, authors="", abstract="", doi="", wireless_label=""):
    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "doi": doi,
        "wireless_label": wireless_label,
    }


def test_doi_match_beats_title_drift() -> None:
    # Same paper, very different titles, but a shared DOI -> still matched.
    a = [_row("Spectrum Sensing: A Field Study", doi="10.1145/x")]
    b = [_row("On Sensing the RF Spectrum in the Wild", doi="10.1145/X")]  # DOI case-insensitive
    summary, rows = diff_paper_sets(a, b)
    assert summary.shared == 1
    assert summary.doi_count == 1
    assert summary.only_in_a == 0 and summary.only_in_b == 0
    assert rows[0]["match_type"] == "doi"


def test_reference_precision_recall() -> None:
    # B is ground truth (3 papers); A predicts 2 correct + 1 extra.
    a = [_row("Alpha", doi="10.1/a"), _row("Beta", doi="10.1/b"), _row("Hallucinated")]
    b = [_row("Alpha", doi="10.1/a"), _row("Beta", doi="10.1/b"), _row("Gamma", doi="10.1/g")]
    summary, _ = diff_paper_sets(a, b, reference="b")
    metrics = summary.metrics()
    assert metrics is not None
    assert metrics.tp == 2 and metrics.fp == 1 and metrics.fn == 1
    assert metrics.precision == 2 / 3  # 2 of 3 predicted are real
    assert metrics.recall == 2 / 3  # 2 of 3 real papers caught
    # Jaccard is symmetric and unchanged by the reference choice.
    assert summary.jaccard == 0.5
    text = format_diff_summary(summary)
    assert "precision" in text and "recall" in text


def test_diff_counts_shared_and_unique() -> None:
    a = [_row("Deep Learning for Radio"), _row("MIMO Beamforming"), _row("Only In A")]
    b = [_row("deep learning for radio"), _row("MIMO Beamforming"), _row("Only In B")]
    summary, rows = diff_paper_sets(a, b)

    assert summary.shared == 2
    assert summary.only_in_a == 1
    assert summary.only_in_b == 1
    assert summary.union == 4
    assert summary.jaccard == 0.5

    statuses = sorted(row["status"] for row in rows)
    assert statuses == ["only_in_a", "only_in_b", "shared", "shared"]


def test_diff_fuzzy_with_author_boost_and_exact_toggle() -> None:
    a = [_row("Spectrum Sensing in 5G Networks", authors="Ada Lovelace, Grace Hopper")]
    b = [_row("Spectrum Sensing in 5G Network", authors="A. Lovelace, G. Hopper")]

    summary, _ = diff_paper_sets(a, b)
    assert summary.shared == 1
    assert summary.fuzzy_count == 1

    strict, _ = diff_paper_sets(a, b, fuzzy=False)
    assert strict.shared == 0
    assert strict.only_in_a == 1
    assert strict.only_in_b == 1


def test_abstract_coverage_tracks_each_side() -> None:
    a = [_row("Paper One", abstract="we measure CSI"), _row("Paper Two")]
    b = [_row("Paper One", abstract="we measure CSI"), _row("Paper Two", abstract="now with abstract")]
    summary, rows = diff_paper_sets(a, b, label_a="URL+LLM", label_b="DBLP")

    assert summary.abstracts_a == 1
    assert summary.abstracts_b == 2

    shared = [r for r in rows if r["status"] == "shared" and r["title_a"] == "Paper Two"][0]
    assert shared["abstract_a"] == "no"
    assert shared["abstract_b"] == "yes"

    text = format_diff_summary(summary)
    assert "URL+LLM vs DBLP" in text
    assert "Jaccard (IoU) = 1.0000" in text


def test_dedup_by_normalized_title() -> None:
    a = [_row("Same Title"), _row("same   title")]  # duplicate after normalization
    b = [_row("Same Title")]
    summary, _ = diff_paper_sets(a, b)
    assert summary.count_a == 1
    assert summary.shared == 1
    assert summary.union == 1


def test_load_paper_set_csv_and_json_roundtrip(tmp_path) -> None:
    a = [_row("Alpha", doi="10.1/a"), _row("Beta")]
    b = [_row("Alpha", doi="10.1/a")]
    summary, rows = diff_paper_sets(a, b)

    csv_path = write_diff_csv(rows, tmp_path / "diff.csv")
    assert csv_path.exists()
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(DIFF_COLUMNS)

    json_path = write_diff_report(summary, rows, tmp_path / "diff.json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["counts"]["shared"] == 1
    assert payload["counts"]["only_in_a"] == 1
    assert len(payload["papers"]) == 2

    # paper-set style files load back and diff identically.
    set_csv = tmp_path / "set_a.csv"
    import csv as _csv

    with set_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=["title", "authors", "abstract", "doi", "wireless_label"])
        writer.writeheader()
        writer.writerows(a)
    set_json = tmp_path / "set_b.json"
    set_json.write_text(json.dumps(b), encoding="utf-8")

    reloaded_a = load_paper_set(set_csv)
    reloaded_b = load_paper_set(set_json)
    summary2, _ = diff_paper_sets(reloaded_a, reloaded_b)
    assert summary2.shared == summary.shared
    assert summary2.union == summary.union
