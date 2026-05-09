from pathlib import Path

from openpyxl import load_workbook

from wireless_taxonomy.config import load_settings
from wireless_taxonomy.export.schemas import BIBTEX_COLUMNS, BIBTEX_SHEET, DATASETS_COLUMNS, DATASETS_SHEET, PAPERS_COLUMNS, PAPERS_SHEET
from wireless_taxonomy.pipeline import Pipeline


FIXTURES = Path(__file__).parent / "fixtures"


def test_export_schema_columns_exact(tmp_path: Path) -> None:
    db = tmp_path / "taxonomy.sqlite"
    out = tmp_path / "taxonomy.xlsx"
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        pipeline.export(run_id, str(out), "xlsx")
    finally:
        pipeline.close()
    wb = load_workbook(out)
    assert [cell.value for cell in wb[PAPERS_SHEET][1]] == PAPERS_COLUMNS
    assert [cell.value for cell in wb[DATASETS_SHEET][1]] == DATASETS_COLUMNS
    assert [cell.value for cell in wb[BIBTEX_SHEET][1]] == BIBTEX_COLUMNS


def test_json_export_contains_nested_paper_records(tmp_path: Path) -> None:
    db = tmp_path / "taxonomy.sqlite"
    out = tmp_path / "taxonomy.json"
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        pipeline.export(run_id, str(out), "json")
    finally:
        pipeline.close()

    import json

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["run_id"] == run_id
    assert len(payload["papers"]) == 2
    assert "paper_sources" in payload["papers"][0]
    assert "paper_list_verification_reports" in payload
    assert payload["papers"][0]["title"] == "Example Wireless Dataset Paper"
