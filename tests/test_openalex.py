from __future__ import annotations

from pathlib import Path

from wireless_taxonomy.analyze.openalex import (
    abstract_for,
    fetch_work,
    reconstruct_abstract,
)
from wireless_taxonomy.config import load_settings
from wireless_taxonomy.pipeline import Pipeline


def test_reconstruct_abstract_orders_words_by_position() -> None:
    inverted = {"Wireless": [0, 3], "channel": [1], "estimation": [2]}
    assert reconstruct_abstract(inverted) == "Wireless channel estimation Wireless"


def test_reconstruct_abstract_handles_empty() -> None:
    assert reconstruct_abstract(None) is None
    assert reconstruct_abstract({}) is None


def test_fetch_work_prefers_doi_exact() -> None:
    calls: list[str] = []

    def fake(url: str) -> dict:
        calls.append(url)
        if "doi.org" in url:
            return {"id": "W1", "title": "By DOI", "abstract_inverted_index": {"a": [0]}}
        return {"results": []}

    work = fetch_work(fake, doi="10.1145/1234", title="Whatever")
    assert work is not None and work["id"] == "W1"
    assert any("doi.org" in url for url in calls)


def test_fetch_work_title_fallback_requires_match() -> None:
    def fake(url: str) -> dict:
        return {"results": [{"title": "Totally Different Paper", "abstract_inverted_index": {"x": [0]}}]}

    # Title returned by OpenAlex doesn't match the query → no match.
    assert fetch_work(fake, doi=None, title="Deep Learning for 5G Channel Estimation") is None


def test_abstract_for_title_match() -> None:
    def fake(url: str) -> dict:
        return {
            "results": [
                {
                    "title": "Deep Learning for 5G Channel Estimation",
                    "abstract_inverted_index": {"Neural": [0], "estimator": [1]},
                }
            ]
        }

    assert abstract_for(fake, doi=None, title="Deep Learning for 5G Channel Estimation") == "Neural estimator"


def test_abstract_for_network_error_is_graceful() -> None:
    def boom(url: str) -> dict:
        raise OSError("network down")

    assert abstract_for(boom, doi="10.1/x", title="Some Title") is None


def test_enrich_abstracts_fills_missing(tmp_path: Path) -> None:
    csv_path = tmp_path / "papers.csv"
    csv_path.write_text(
        "Paper Title,Authors,Conference,Year\n"
        "Deep Learning for 5G Channel Estimation,Alice Smith,SIGCOMM,2024\n",
        encoding="utf-8",
    )
    pipeline = Pipeline(load_settings(tmp_path / "taxonomy.sqlite"))

    def fake(url: str) -> dict:
        return {
            "results": [
                {
                    "title": "Deep Learning for 5G Channel Estimation",
                    "abstract_inverted_index": {"A": [0], "wireless": [1], "abstract": [2]},
                }
            ]
        }

    try:
        run_id = pipeline.ingest("SIGCOMM", 2024, "csv", str(csv_path))
        before = pipeline.conn.execute("SELECT abstract FROM papers").fetchone()["abstract"]
        pipeline.enrich_abstracts(run_id, fetch_json=fake)
        after = pipeline.conn.execute("SELECT abstract FROM papers").fetchone()["abstract"]
    finally:
        pipeline.close()

    assert not (before or "").strip()
    assert after == "A wireless abstract"
