from pathlib import Path
import json

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


def test_verify_paper_list_flags_structurally_malformed_records(tmp_path: Path) -> None:
    db = tmp_path / "taxonomy.sqlite"
    malformed_title = (
        "This paper introduces SpliDT, a scalable framework that reimagines DT deployment "
        "as a partitioned inference problem over a sliding window of packets. By dividing "
        "inference into sequential subtrees, SpliDT supports more stateful features without "
        "exceeding hardware limits and maintains line-rate processing."
    )
    contaminated_authors = (
        "Evaluations show that SpliDT supports up to 5x more features and outperforms baselines. "
        "Carbon- and Precedence-Aware Scheduling for Data Processing Clusters Ada Lovelace "
        "(Example University); Grace Hopper (Example Labs)"
    )
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        paper_id = pipeline.conn.execute(
            "SELECT id FROM papers WHERE title = ?",
            ("Example Wireless Dataset Paper",),
        ).fetchone()["id"]
        pipeline.conn.execute(
            "UPDATE papers SET title = ?, authors = ? WHERE id = ?",
            (malformed_title, contaminated_authors, paper_id),
        )
        pipeline.conn.commit()

        verify_run_id = pipeline.verify_paper_list(run_id)
        report = pipeline.latest_paper_list_report(verify_run_id)
        review = pipeline.conn.execute(
            "SELECT * FROM review_items WHERE run_id = ? AND field = 'Paper Structure'",
            (verify_run_id,),
        ).fetchone()
    finally:
        pipeline.close()

    assert report is not None
    payload = json.loads(report["report_json"])
    messages = [issue["message"] for issue in payload["issues"]]
    assert report["final_confidence"] < 1.0
    assert any("Paper title is unusually long" in message for message in messages)
    assert any("Paper title looks like abstract prose" in message for message in messages)
    assert review is not None
    assert review["review_reason"].startswith("Paper title is unusually long")
