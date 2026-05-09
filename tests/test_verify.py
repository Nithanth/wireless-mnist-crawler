from pathlib import Path

from wireless_taxonomy.config import load_settings
from wireless_taxonomy.pipeline import Pipeline


FIXTURES = Path(__file__).parent / "fixtures"


def test_verify_paper_list_creates_report_and_review_items(tmp_path: Path) -> None:
    db = tmp_path / "taxonomy.sqlite"
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        verify_run_id = pipeline.verify_paper_list(run_id)
        report = pipeline.latest_paper_list_report(verify_run_id)
        review_count = pipeline.conn.execute("SELECT COUNT(*) AS count FROM review_items WHERE run_id = ?", (verify_run_id,)).fetchone()["count"]
    finally:
        pipeline.close()

    assert report is not None
    assert report["paper_count"] == 2
    assert report["missing_doi_count"] == 0
    assert report["final_confidence"] == 1.0
    assert review_count == 0


def test_verify_paper_list_flags_missing_metadata(tmp_path: Path) -> None:
    db = tmp_path / "taxonomy.sqlite"
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "csv", str(FIXTURES / "sample_papers.csv"))
        verify_run_id = pipeline.verify_paper_list(run_id)
        report = pipeline.latest_paper_list_report(verify_run_id)
        review_count = pipeline.conn.execute("SELECT COUNT(*) AS count FROM review_items WHERE run_id = ?", (verify_run_id,)).fetchone()["count"]
    finally:
        pipeline.close()

    assert report is not None
    assert report["missing_abstract_count"] == 1
    assert report["missing_doi_count"] == 1
    assert review_count >= 2
