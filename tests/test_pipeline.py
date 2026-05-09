from pathlib import Path

from wireless_taxonomy.config import load_settings
from wireless_taxonomy.pipeline import Pipeline


FIXTURES = Path(__file__).parent / "fixtures"


def test_tiny_end_to_end_fixture_pipeline(tmp_path: Path) -> None:
    db = tmp_path / "taxonomy.sqlite"
    out = tmp_path / "taxonomy.xlsx"
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.run("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"), str(out))
        runs = pipeline.status()
        pending_review = pipeline.conn.execute("SELECT COUNT(*) AS count FROM review_items").fetchone()["count"]
        papers = pipeline.conn.execute("SELECT COUNT(*) AS count FROM papers").fetchone()["count"]
    finally:
        pipeline.close()
    assert run_id == 1
    assert len(runs) == 6
    assert papers == 2
    assert pending_review >= 1
    assert out.exists()
