import csv
import json
from pathlib import Path

import pytest

from wireless_taxonomy.config import load_settings
from wireless_taxonomy.evaluate.jaccard import (
    COMPARISON_COLUMNS,
    comparison_rows,
    comparison_rows_all,
    detect_title_column,
    evaluate_run,
    format_aggregate_summary,
    format_report_summary,
)
from wireless_taxonomy.export.paper_set import PAPER_SET_COLUMNS
from wireless_taxonomy.pipeline import Pipeline

FIXTURES = Path(__file__).parent / "fixtures"
MANUAL = FIXTURES / "manual_papers.csv"


def _ingest(tmp_path: Path) -> tuple[Pipeline, int]:
    pipeline = Pipeline(load_settings(tmp_path / "taxonomy.sqlite"))
    run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
    return pipeline, run_id


def test_paper_set_csv_has_match_key_and_abstract(tmp_path: Path) -> None:
    pipeline, run_id = _ingest(tmp_path)
    out = tmp_path / "papers.csv"
    try:
        path = pipeline.export_paper_set(run_id, str(out), "csv")
    finally:
        pipeline.close()

    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == PAPER_SET_COLUMNS
    wireless = next(row for row in rows if row["title"] == "Example Wireless Dataset Paper")
    assert wireless["match_key"] == "example wireless dataset paper"
    assert wireless["abstract"]
    assert wireless["venue"] == "SIGCOMM"


def test_paper_set_json_round_trips(tmp_path: Path) -> None:
    pipeline, run_id = _ingest(tmp_path)
    out = tmp_path / "papers.json"
    try:
        path = pipeline.export_paper_set(run_id, str(out), "json")
    finally:
        pipeline.close()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert {row["match_key"] for row in payload} == {
        "example wireless dataset paper",
        "example datacenter congestion paper",
    }


def test_paper_set_wireless_only_filters_to_classified(tmp_path: Path) -> None:
    pipeline, run_id = _ingest(tmp_path)
    try:
        pipeline.classify_wireless(run_id)
        path = pipeline.export_paper_set(run_id, str(tmp_path / "wireless.csv"), "csv", wireless_only=True)
    finally:
        pipeline.close()
    with path.open(encoding="utf-8") as fh:
        titles = [row["title"] for row in csv.DictReader(fh)]
    assert titles == ["Example Wireless Dataset Paper"]


def test_paper_set_wireless_only_requires_classification(tmp_path: Path) -> None:
    pipeline, run_id = _ingest(tmp_path)
    try:
        with pytest.raises(ValueError):
            pipeline.export_paper_set(run_id, str(tmp_path / "w.csv"), "csv", wireless_only=True)
    finally:
        pipeline.close()


def test_paper_set_rejects_unknown_format(tmp_path: Path) -> None:
    pipeline, run_id = _ingest(tmp_path)
    try:
        with pytest.raises(ValueError):
            pipeline.export_paper_set(run_id, str(tmp_path / "papers.txt"), "txt")
    finally:
        pipeline.close()


def test_jaccard_wireless_only_with_conference_filter(tmp_path: Path) -> None:
    pipeline, run_id = _ingest(tmp_path)
    out = tmp_path / "report.json"
    try:
        pipeline.classify_wireless(run_id)
        report = pipeline.jaccard(run_id, str(MANUAL), out=str(out))
    finally:
        pipeline.close()

    # Manual rows filtered to SIGCOMM/2025 -> 2 distinct papers (dup collapses, NSDI dropped).
    # Automated wireless set is just the wireless paper; the datacenter paper is excluded.
    assert report.venue == "SIGCOMM"
    assert report.year == 2025
    assert report.wireless_only is True
    assert report.conference_filtered is True
    assert report.title_column == "Paper Title"
    assert report.automated_count == 1
    assert report.manual_count == 2
    assert report.intersection_count == 1
    assert report.jaccard_index == pytest.approx(0.5)
    assert report.matched == ["Example Wireless Dataset Paper"]
    assert report.missed_by_cli == ["A Manually Curated Paper Not Fetched"]
    assert report.extra_from_cli == []

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["jaccard_index"] == pytest.approx(0.5)
    assert payload["counts"]["intersection"] == 1


def test_jaccard_all_papers(tmp_path: Path) -> None:
    pipeline, run_id = _ingest(tmp_path)
    try:
        report = pipeline.jaccard(run_id, str(MANUAL), wireless_only=False)
    finally:
        pipeline.close()
    # Full ingested list (2) vs SIGCOMM/2025 manual (2): the non-wireless paper is now an extra.
    assert report.automated_count == 2
    assert report.manual_count == 2
    assert report.jaccard_index == pytest.approx(1 / 3)
    assert report.extra_from_cli == ["Example Datacenter Congestion Paper"]


def test_jaccard_no_conference_filter_includes_other_venues(tmp_path: Path) -> None:
    pipeline, run_id = _ingest(tmp_path)
    try:
        report = pipeline.jaccard(run_id, str(MANUAL), wireless_only=False, conference_filter=False)
    finally:
        pipeline.close()
    assert report.conference_filtered is False
    assert report.manual_count == 3  # NSDI row no longer filtered out


def test_jaccard_explicit_title_column(tmp_path: Path) -> None:
    pipeline, run_id = _ingest(tmp_path)
    try:
        report = pipeline.jaccard(run_id, str(MANUAL), title_col="paper title", wireless_only=False)
    finally:
        pipeline.close()
    assert report.title_column == "Paper Title"
    assert report.intersection_count == 1


def _write_manual(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["Paper Title", "Authors", "Conference", "Year"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_jaccard_fuzzy_matches_near_duplicate_with_author_boost(tmp_path: Path) -> None:
    manual = tmp_path / "manual.csv"
    # Near-duplicate of the fetched "Example Wireless Dataset Paper" with shared authors.
    _write_manual(
        manual,
        [
            {
                "Paper Title": "Example Wireless Dataset Paper (Extended)",
                "Authors": "A. Lovelace, G. Hopper",
                "Conference": "SIGCOMM",
                "Year": "2025",
            }
        ],
    )
    pipeline, run_id = _ingest(tmp_path)
    try:
        pipeline.classify_wireless(run_id)
        fuzzy = pipeline.jaccard(run_id, str(manual))
        strict = pipeline.jaccard(run_id, str(manual), fuzzy=False)
    finally:
        pipeline.close()

    assert fuzzy.fuzzy is True
    assert fuzzy.intersection_count == 1
    assert fuzzy.jaccard_index == pytest.approx(1.0)
    assert len(fuzzy.fuzzy_matches) == 1
    pair = fuzzy.fuzzy_matches[0]
    assert pair.shared_authors == ["hopper", "lovelace"]

    # Exact mode does not collapse the near-duplicate.
    assert strict.intersection_count == 0
    assert strict.missed_by_cli == ["Example Wireless Dataset Paper (Extended)"]
    assert strict.extra_from_cli == ["Example Wireless Dataset Paper"]


def _ingest_two_conferences(tmp_path: Path) -> tuple[Pipeline, int, int]:
    pipeline = Pipeline(load_settings(tmp_path / "taxonomy.sqlite"))
    sigcomm = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
    nsdi_csv = tmp_path / "nsdi.csv"
    _write_manual(
        nsdi_csv,
        [{"Paper Title": "Some Other Conference Paper", "Authors": "", "Conference": "NSDI", "Year": "2024"}],
    )
    nsdi = pipeline.ingest("NSDI", 2024, "csv", str(nsdi_csv))
    return pipeline, sigcomm, nsdi


def test_jaccard_all_aggregates_micro_macro(tmp_path: Path) -> None:
    pipeline, _, _ = _ingest_two_conferences(tmp_path)
    out = tmp_path / "aggregate.json"
    try:
        aggregate = pipeline.jaccard_all(str(MANUAL), wireless_only=False, out=str(out))
    finally:
        pipeline.close()

    # Two conferences, sorted by venue name: NSDI 2024 then SIGCOMM 2025.
    assert [(r.venue, r.year) for r in aggregate.reports] == [("NSDI", 2024), ("SIGCOMM", 2025)]
    by_venue = {r.venue: r for r in aggregate.reports}
    assert by_venue["NSDI"].jaccard_index == pytest.approx(1.0)  # 1 matched / 1 union
    assert by_venue["SIGCOMM"].jaccard_index == pytest.approx(1 / 3)  # 1 matched / 3 union
    # micro pools papers: (1+1) / (1+3); macro averages indices: (1.0 + 1/3) / 2
    assert aggregate.micro_index == pytest.approx(0.5)
    assert aggregate.macro_index == pytest.approx((1.0 + 1 / 3) / 2)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["conferences"] == 2
    assert payload["micro_jaccard_index"] == pytest.approx(0.5)


def test_jaccard_all_skips_unclassified_when_auto_classify_off(tmp_path: Path) -> None:
    pipeline, sigcomm, _ = _ingest_two_conferences(tmp_path)
    try:
        pipeline.classify_wireless(sigcomm)  # only SIGCOMM gets a wireless classification run
        aggregate = pipeline.jaccard_all(str(MANUAL), wireless_only=True, auto_classify=False)
    finally:
        pipeline.close()

    assert [r.venue for r in aggregate.reports] == ["SIGCOMM"]
    assert [entry["venue"] for entry in aggregate.skipped] == ["NSDI"]
    assert "classify-wireless" in aggregate.skipped[0]["reason"]


def test_jaccard_all_auto_classifies_unclassified_conferences(tmp_path: Path) -> None:
    pipeline, _, _ = _ingest_two_conferences(tmp_path)
    try:
        # No conference classified up front; auto_classify (default) should classify both.
        aggregate = pipeline.jaccard_all(str(MANUAL), wireless_only=True)
    finally:
        pipeline.close()

    assert aggregate.skipped == []
    assert {r.venue for r in aggregate.reports} == {"NSDI", "SIGCOMM"}


def test_paper_set_surfaces_wireless_confidence(tmp_path: Path) -> None:
    pipeline, run_id = _ingest(tmp_path)
    try:
        pipeline.classify_wireless(run_id)
        path = pipeline.export_paper_set(run_id, str(tmp_path / "ps.csv"), "csv", wireless_only=True)
    finally:
        pipeline.close()
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert "wireless_label" in rows[0] and "wireless_confidence" in rows[0]
    wireless = rows[0]
    assert wireless["wireless_label"] == "yes"
    assert float(wireless["wireless_confidence"]) > 0


def test_comparison_rows_status_and_confidence(tmp_path: Path) -> None:
    pipeline, run_id = _ingest(tmp_path)
    try:
        pipeline.classify_wireless(run_id)
        evaluation = evaluate_run(pipeline.conn, run_id, str(MANUAL))
        rows = comparison_rows(evaluation)
    finally:
        pipeline.close()

    by_status: dict[str, list[dict]] = {}
    for row in rows:
        assert list(row.keys()) == COMPARISON_COLUMNS
        by_status.setdefault(row["status"], []).append(row)

    assert set(by_status) == {"matched", "missed_by_cli"}
    matched = by_status["matched"][0]
    assert matched["manual_title"] == "Example Wireless Dataset Paper"
    assert matched["automated_title"] == "Example Wireless Dataset Paper"
    assert matched["wireless_label"] == "yes"
    assert float(matched["wireless_confidence"]) > 0
    # The curated-but-not-fetched paper has no automated counterpart / confidence.
    missed = by_status["missed_by_cli"][0]
    assert missed["manual_title"] == "A Manually Curated Paper Not Fetched"
    assert missed["automated_title"] == ""
    assert missed["wireless_confidence"] == ""


def test_jaccard_writes_comparison_csv(tmp_path: Path) -> None:
    pipeline, run_id = _ingest(tmp_path)
    csv_out = tmp_path / "comparison.csv"
    try:
        pipeline.classify_wireless(run_id)
        pipeline.jaccard(run_id, str(MANUAL), csv_out=str(csv_out))
    finally:
        pipeline.close()
    assert csv_out.exists()
    with csv_out.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == COMPARISON_COLUMNS
        statuses = {row["status"] for row in reader}
    assert statuses == {"matched", "missed_by_cli"}


def test_comparison_rows_all_spans_conferences(tmp_path: Path) -> None:
    pipeline, sigcomm, nsdi = _ingest_two_conferences(tmp_path)
    try:
        pipeline.classify_wireless(sigcomm)
        pipeline.classify_wireless(nsdi)
        rows = comparison_rows_all(pipeline.conn, str(MANUAL), wireless_only=False)
    finally:
        pipeline.close()
    venues = {row["venue"] for row in rows}
    assert venues == {"SIGCOMM", "NSDI"}
    assert all(list(row.keys()) == COMPARISON_COLUMNS for row in rows)


def test_format_report_summary_is_readable(tmp_path: Path) -> None:
    pipeline, run_id = _ingest(tmp_path)
    try:
        pipeline.classify_wireless(run_id)
        report = pipeline.jaccard(run_id, str(MANUAL))
    finally:
        pipeline.close()
    text = format_report_summary(report)
    assert "SIGCOMM 2025" in text
    assert "Jaccard (IoU) = 0.5000" in text
    assert "mean wireless confidence" in text


def test_format_aggregate_summary_lists_conferences(tmp_path: Path) -> None:
    pipeline, _, _ = _ingest_two_conferences(tmp_path)
    try:
        aggregate = pipeline.jaccard_all(str(MANUAL), wireless_only=False)
    finally:
        pipeline.close()
    text = format_aggregate_summary(aggregate)
    assert "micro" in text and "macro" in text
    assert "NSDI 2024" in text and "SIGCOMM 2025" in text


def test_detect_title_column_errors_when_missing() -> None:
    with pytest.raises(ValueError):
        detect_title_column(["foo", "bar"])
    with pytest.raises(ValueError):
        detect_title_column(["Paper Title"], override="nonexistent")
