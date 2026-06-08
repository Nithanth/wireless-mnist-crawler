import csv
import json
from pathlib import Path

import pytest

from wireless_taxonomy.config import load_settings
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
