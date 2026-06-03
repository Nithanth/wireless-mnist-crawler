from pathlib import Path

from wireless_taxonomy.analyze.abstracts import AbstractEnricher, _openalex_abstract, _strip_jats
from wireless_taxonomy.config import load_settings
from wireless_taxonomy.pipeline import Pipeline

FIXTURES = Path(__file__).parent / "fixtures"


def test_openalex_inverted_index_reconstruction() -> None:
    payload = {"abstract_inverted_index": {"Wireless": [0], "sensing": [1, 3], "and": [2]}}
    assert _openalex_abstract(payload) == "Wireless sensing and sensing"


def test_strip_jats_tags() -> None:
    assert _strip_jats("<jats:p>Hello <jats:italic>world</jats:italic></jats:p>") == "Hello world"


def test_enricher_tries_providers_in_order() -> None:
    long_abstract = "A wireless sensing system using RF measurements and CSI features. " * 2

    def fake_fetch(url: str) -> dict:
        if "openalex" in url:
            return {}  # OpenAlex has nothing
        if "crossref" in url:
            return {"message": {"abstract": f"<jats:p>{long_abstract}</jats:p>"}}
        return {}

    enricher = AbstractEnricher(fetch_json=fake_fetch)
    result = enricher.fetch("Some Wireless Paper", "10.1/abc")
    assert result is not None
    assert result.provider == "crossref"
    assert "RF measurements" in result.abstract


def test_enricher_returns_none_when_no_provider_has_abstract() -> None:
    enricher = AbstractEnricher(fetch_json=lambda url: {})
    assert enricher.fetch("Title", "10.1/x") is None


def test_enrich_abstracts_pipeline_fills_missing(tmp_path: Path) -> None:
    db = tmp_path / "taxonomy.sqlite"

    class FakeEnricher:
        def fetch(self, title, doi):
            from wireless_taxonomy.analyze.abstracts import AbstractResult

            return AbstractResult("Backfilled wireless abstract about CSI.", "openalex", "http://x")

    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        pipeline.conn.execute("UPDATE papers SET abstract = NULL")
        pipeline.conn.commit()
        pipeline.enrich_abstracts(run_id, enricher=FakeEnricher())
        abstracts = [row["abstract"] for row in pipeline.conn.execute("SELECT abstract FROM papers")]
    finally:
        pipeline.close()
    assert all(a == "Backfilled wireless abstract about CSI." for a in abstracts)
