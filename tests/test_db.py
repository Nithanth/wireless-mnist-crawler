from pathlib import Path

from wireless_taxonomy.db import connect, migrate


def test_init_creates_expected_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "taxonomy.sqlite"
    migrate(db_path)
    conn = connect(db_path)
    try:
        tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        conn.close()
    assert "pipeline_runs" in tables
    assert "papers" in tables
    assert "review_items" in tables
    assert "evidence_claims" in tables
    assert "paper_text_artifacts" in tables
    assert "paper_text_links" in tables
    assert "paper_text_snippets" in tables
    assert "scope_assessments" in tables
    assert "paper_agentic_analyses" in tables
    assert "paper_analysis_dataset_claims" in tables
    assert "paper_analysis_reflections" in tables
    assert "paper_input_readiness" in tables
    assert "resolver_cache" in tables
