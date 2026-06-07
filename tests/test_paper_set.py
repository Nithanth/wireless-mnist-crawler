import csv
import json
from pathlib import Path

import pytest

from wireless_taxonomy.config import load_settings
from wireless_taxonomy.evaluate.jaccard import detect_title_column
from wireless_taxonomy.export.paper_set import PAPER_SET_COLUMNS
from wireless_taxonomy.pipeline import Pipeline

FIXTURES = Path(__file__).parent / "fixtures"


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
    titles = {row["title"] for row in rows}
    assert "Example Wireless Dataset Paper" in titles
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


def test_paper_set_rejects_unknown_format(tmp_path: Path) -> None:
    pipeline, run_id = _ingest(tmp_path)
    try:
        with pytest.raises(ValueError):
            pipeline.export_paper_set(run_id, str(tmp_path / "papers.txt"), "txt")
    finally:
        pipeline.close()


def test_jaccard_index_and_diff(tmp_path: Path) -> None:
    pipeline, run_id = _ingest(tmp_path)
    out = tmp_path / "report.json"
    try:
        report = pipeline.jaccard(run_id, str(FIXTURES / "manual_papers.csv"), out=str(out))
    finally:
        pipeline.close()

    # Manual set has 2 distinct papers after normalization (the duplicate collapses);
    # one overlaps the fetched set, one is missed. The fetched set has one extra paper.
    assert report.title_column == "Paper Title"
    assert report.automated_count == 2
    assert report.manual_count == 2
    assert report.intersection_count == 1
    assert report.union_count == 3
    assert report.jaccard_index == pytest.approx(1 / 3)
    assert report.missed_by_cli == ["A Manually Curated Paper Not Fetched"]
    assert report.extra_from_cli == ["Example Datacenter Congestion Paper"]
    assert report.matched == ["Example Wireless Dataset Paper"]

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["counts"]["intersection"] == 1
    assert payload["jaccard_index"] == pytest.approx(1 / 3)


def test_jaccard_explicit_title_column(tmp_path: Path) -> None:
    pipeline, run_id = _ingest(tmp_path)
    try:
        report = pipeline.jaccard(run_id, str(FIXTURES / "manual_papers.csv"), title_col="paper title")
    finally:
        pipeline.close()
    assert report.title_column == "Paper Title"
    assert report.intersection_count == 1


def test_detect_title_column_errors_when_missing() -> None:
    with pytest.raises(ValueError):
        detect_title_column(["foo", "bar"])
    with pytest.raises(ValueError):
        detect_title_column(["Paper Title"], override="nonexistent")
