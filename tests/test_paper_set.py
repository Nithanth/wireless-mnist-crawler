import csv
import json
from pathlib import Path

import pytest

from wireless_taxonomy.config import load_settings
from wireless_taxonomy.evaluate.jaccard import detect_title_column
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


def test_detect_title_column_errors_when_missing() -> None:
    with pytest.raises(ValueError):
        detect_title_column(["foo", "bar"])
    with pytest.raises(ValueError):
        detect_title_column(["Paper Title"], override="nonexistent")
