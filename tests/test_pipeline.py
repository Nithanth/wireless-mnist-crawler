import json
from pathlib import Path

from wireless_taxonomy.analyze import acm_browser
from wireless_taxonomy.analyze import full_text
from wireless_taxonomy.models import PaperTextArtifact, PaperTextEnrichment, PaperTextLink
from wireless_taxonomy.config import load_settings
from wireless_taxonomy.llm import LlmResponse
from wireless_taxonomy.pipeline import Pipeline


FIXTURES = Path(__file__).parent / "fixtures"


def test_tiny_end_to_end_fixture_pipeline(tmp_path: Path) -> None:
    db = tmp_path / "taxonomy.sqlite"
    out = tmp_path / "taxonomy_csv"
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.run("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"), str(out))
        runs = pipeline.status()
        pending_review = pipeline.conn.execute("SELECT COUNT(*) AS count FROM review_items").fetchone()["count"]
        papers = pipeline.conn.execute("SELECT COUNT(*) AS count FROM papers").fetchone()["count"]
    finally:
        pipeline.close()
    assert run_id == 1
    assert len(runs) == 12
    assert papers == 2
    assert pending_review >= 1
    assert (out / "list_of_papers.csv").exists()


def test_classify_wireless_exports_title_abstract_json_cache(tmp_path: Path) -> None:
    db = tmp_path / "taxonomy.sqlite"
    out = tmp_path / "classification-cache.json"
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        classification_run_id = pipeline.classify_wireless(run_id)
        path = pipeline.export_classification_cache(run_id, classification_run_id, out)
        review_count = pipeline.conn.execute(
            "SELECT COUNT(*) AS count FROM review_items WHERE run_id = ?",
            (classification_run_id,),
        ).fetchone()["count"]
    finally:
        pipeline.close()

    payload = json.loads(path.read_text(encoding="utf-8"))
    papers = {paper["title"]: paper for paper in payload["papers"]}

    assert payload["schema"] == "wireless-taxonomy.paper-classification-cache.v1"
    assert payload["paper_count"] == 2
    assert payload["summary"]["category_counts"]["wireless"] == 1
    assert payload["summary"]["category_counts"]["networking_non_wireless"] == 1
    assert payload["summary"]["wireless_count"] == 1
    assert review_count == 0
    assert papers["Example Wireless Dataset Paper"]["classification"]["is_wireless"] == 1
    assert papers["Example Wireless Dataset Paper"]["classification"]["category"] == "wireless"
    assert papers["Example Datacenter Congestion Paper"]["classification"]["is_wireless"] == 0
    assert papers["Example Datacenter Congestion Paper"]["classification"]["category"] == "networking_non_wireless"


def test_enrich_paper_text_persists_artifacts_links_and_snippets(tmp_path: Path) -> None:
    db = tmp_path / "taxonomy.sqlite"
    paper_page = FIXTURES / "example_paper_page.html"
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        first_paper_id = pipeline.conn.execute(
            "SELECT id FROM papers WHERE title = ?",
            ("Example Wireless Dataset Paper",),
        ).fetchone()["id"]
        pipeline.conn.execute("UPDATE papers SET paper_url = ? WHERE id = ?", (str(paper_page), first_paper_id))
        pipeline.conn.commit()

        enrich_run_id = pipeline.enrich_paper_text(run_id)
        artifact_count = pipeline.conn.execute(
            "SELECT COUNT(*) AS count FROM paper_text_artifacts WHERE run_id = ?",
            (enrich_run_id,),
        ).fetchone()["count"]
        dataset_links = pipeline.conn.execute(
            "SELECT COUNT(*) AS count FROM paper_text_links WHERE run_id = ? AND link_type = 'dataset_or_artifact'",
            (enrich_run_id,),
        ).fetchone()["count"]
        snippets = pipeline.conn.execute(
            "SELECT snippet_text FROM paper_text_snippets WHERE run_id = ? ORDER BY id",
            (enrich_run_id,),
        ).fetchall()
    finally:
        pipeline.close()

    assert artifact_count >= 3
    assert dataset_links == 1
    assert any("RF measurements" in row["snippet_text"] for row in snippets)


def test_assess_paper_inputs_records_readiness_levels(tmp_path: Path) -> None:
    db = tmp_path / "taxonomy.sqlite"
    paper_page = FIXTURES / "example_paper_page.html"
    out = tmp_path / "readiness.json"
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        first_paper_id = pipeline.conn.execute(
            "SELECT id FROM papers WHERE title = ?",
            ("Example Wireless Dataset Paper",),
        ).fetchone()["id"]
        second_paper_id = pipeline.conn.execute(
            "SELECT id FROM papers WHERE title = ?",
            ("Example Datacenter Congestion Paper",),
        ).fetchone()["id"]
        pipeline.conn.execute("UPDATE papers SET paper_url = ? WHERE id = ?", (str(paper_page), first_paper_id))
        pipeline.conn.commit()
        pipeline.enrich_paper_text(run_id)

        readiness_run_id = pipeline.assess_paper_inputs(run_id)
        pipeline.export(readiness_run_id, str(out), "json", scope="exact")
        first = pipeline.conn.execute(
            "SELECT * FROM paper_input_readiness WHERE run_id = ? AND paper_id = ?",
            (readiness_run_id, first_paper_id),
        ).fetchone()
        second = pipeline.conn.execute(
            "SELECT * FROM paper_input_readiness WHERE run_id = ? AND paper_id = ?",
            (readiness_run_id, second_paper_id),
        ).fetchone()
        review_count = pipeline.conn.execute(
            "SELECT COUNT(*) AS count FROM review_items WHERE run_id = ?",
            (readiness_run_id,),
        ).fetchone()["count"]
    finally:
        pipeline.close()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert first is not None
    assert first["readiness_level"] == "full_text_plus_links"
    assert first["has_fetched_text"] == 1
    assert first["has_pdf_link"] == 1
    assert first["has_artifact_link"] == 1
    assert first["should_analyze"] == 1
    assert second is not None
    assert second["readiness_level"] == "abstract_only"
    assert second["has_fetched_text"] == 0
    assert second["has_artifact_link"] == 0
    assert second["should_analyze"] == 1
    assert review_count == 0
    assert len(payload["paper_input_readiness"]) == 2
    assert payload["paper_input_readiness"][0]["report"]["readiness_level"] in {"full_text_plus_links", "abstract_only"}


def test_discover_full_text_fetches_landing_pdf_and_artifact_links(tmp_path: Path) -> None:
    db = tmp_path / "taxonomy.sqlite"
    landing_page = FIXTURES / "full_text_landing.html"
    out = tmp_path / "full_text.json"
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        first_paper_id = pipeline.conn.execute(
            "SELECT id FROM papers WHERE title = ?",
            ("Example Wireless Dataset Paper",),
        ).fetchone()["id"]
        pipeline.conn.execute("UPDATE papers SET paper_url = ? WHERE id = ?", (str(landing_page), first_paper_id))
        pipeline.conn.commit()

        full_text_run_id = pipeline.discover_full_text(run_id)
        readiness_run_id = pipeline.assess_paper_inputs(run_id)
        pipeline.export(full_text_run_id, str(out), "json", scope="exact")
        artifact_types = {
            (row["source_type"], row["fetch_status"])
            for row in pipeline.conn.execute(
                "SELECT source_type, fetch_status FROM paper_text_artifacts WHERE run_id = ? AND paper_id = ?",
                (full_text_run_id, first_paper_id),
            ).fetchall()
        }
        link_types = {
            row["link_type"]
            for row in pipeline.conn.execute(
                "SELECT link_type FROM paper_text_links WHERE run_id = ? AND paper_id = ?",
                (full_text_run_id, first_paper_id),
            ).fetchall()
        }
        snippets = pipeline.conn.execute(
            "SELECT snippet_text FROM paper_text_snippets WHERE run_id = ? AND paper_id = ?",
            (full_text_run_id, first_paper_id),
        ).fetchall()
        readiness = pipeline.conn.execute(
            "SELECT * FROM paper_input_readiness WHERE run_id = ? AND paper_id = ?",
            (readiness_run_id, first_paper_id),
        ).fetchone()
    finally:
        pipeline.close()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert ("paper_landing_html", "fetched") in artifact_types
    assert ("pdf_text", "fetched") in artifact_types
    assert {"pdf", "repository", "dataset_or_artifact"} <= link_types
    assert any("Open5G Measurements" in row["snippet_text"] for row in snippets)
    assert readiness is not None
    assert readiness["readiness_level"] == "full_text_plus_links"
    assert readiness["has_fetched_text"] == 1
    assert readiness["has_pdf_link"] == 1
    assert readiness["has_artifact_link"] == 1
    assert payload["paper_text_artifacts"]


def test_discover_full_text_can_target_one_paper(tmp_path: Path) -> None:
    db = tmp_path / "taxonomy.sqlite"
    landing_page = FIXTURES / "full_text_landing.html"
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        first_paper_id = pipeline.conn.execute(
            "SELECT id FROM papers WHERE title = ?",
            ("Example Wireless Dataset Paper",),
        ).fetchone()["id"]
        pipeline.conn.execute("UPDATE papers SET paper_url = ? WHERE id = ?", (str(landing_page), first_paper_id))
        pipeline.conn.commit()

        full_text_run_id = pipeline.discover_full_text(run_id, paper_id=first_paper_id)
        artifact_papers = {
            row["paper_id"]
            for row in pipeline.conn.execute(
                "SELECT DISTINCT paper_id FROM paper_text_artifacts WHERE run_id = ?",
                (full_text_run_id,),
            ).fetchall()
        }
    finally:
        pipeline.close()

    assert artifact_papers == {first_paper_id}


def test_discover_full_text_uses_metadata_pdf_candidates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WIRELESS_TAXONOMY_ENABLE_WEB_SEARCH", "1")
    monkeypatch.setattr(
        "wireless_taxonomy.analyze.full_text._openalex_candidates",
        lambda doi, title=None: [
            ("openalex_pdf", str(FIXTURES / "example-fulltext.pdf")),
            ("openalex_artifact", "https://github.com/example/open5g-measurements"),
        ],
    )
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._crossref_candidates", lambda doi: [])
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._semantic_scholar_candidates", lambda doi, title, authors=None: [])
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._unpaywall_candidates", lambda doi: [])
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._arxiv_candidates", lambda title: [])
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._openreview_candidates", lambda title: [])
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._web_search_candidates", lambda title: [])
    db = tmp_path / "taxonomy.sqlite"
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        first_paper_id = pipeline.conn.execute(
            "SELECT id FROM papers WHERE title = ?",
            ("Example Wireless Dataset Paper",),
        ).fetchone()["id"]
        pipeline.conn.execute("UPDATE papers SET paper_url = ? WHERE id = ?", (str(FIXTURES / "example_paper_page.html"), first_paper_id))
        pipeline.conn.commit()

        full_text_run_id = pipeline.discover_full_text(run_id)
        pdf_artifact = pipeline.conn.execute(
            """
            SELECT * FROM paper_text_artifacts
            WHERE run_id = ? AND paper_id = ? AND source_type = 'pdf_text'
              AND source_url = ?
            """,
            (full_text_run_id, first_paper_id, str(FIXTURES / "example-fulltext.pdf")),
        ).fetchone()
        metadata_links = {
            row["link_type"]
            for row in pipeline.conn.execute(
                "SELECT link_type FROM paper_text_links WHERE run_id = ? AND paper_id = ?",
                (full_text_run_id, first_paper_id),
            ).fetchall()
        }
    finally:
        pipeline.close()

    assert pdf_artifact is not None
    assert pdf_artifact["fetch_status"] == "fetched"
    assert "Open5G Measurements" in pdf_artifact["content_text"]
    assert {"pdf", "repository"} <= metadata_links


def test_discover_full_text_uses_semantic_scholar_open_pdf_candidates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WIRELESS_TAXONOMY_ENABLE_WEB_SEARCH", "1")
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._openalex_candidates", lambda doi, title=None: [])
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._crossref_candidates", lambda doi: [])
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._unpaywall_candidates", lambda doi: [])
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._arxiv_candidates", lambda title: [])
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._openreview_candidates", lambda title: [])
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._web_search_candidates", lambda title: [])
    monkeypatch.setattr(
        "wireless_taxonomy.analyze.full_text._semantic_scholar_candidates",
        lambda doi, title, authors=None: [
            ("semantic_scholar_open_access_pdf", str(FIXTURES / "example-fulltext.pdf")),
            ("semantic_scholar_arxiv_landing", "https://arxiv.org/abs/2501.12345"),
        ],
    )
    db = tmp_path / "taxonomy.sqlite"
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        first_paper_id = pipeline.conn.execute(
            "SELECT id FROM papers WHERE title = ?",
            ("Example Wireless Dataset Paper",),
        ).fetchone()["id"]
        full_text_run_id = pipeline.discover_full_text(run_id, paper_id=first_paper_id)
        pdf_artifact = pipeline.conn.execute(
            """
            SELECT * FROM paper_text_artifacts
            WHERE run_id = ? AND paper_id = ? AND source_type = 'pdf_text'
              AND source_url = ?
            """,
            (full_text_run_id, first_paper_id, str(FIXTURES / "example-fulltext.pdf")),
        ).fetchone()
        links = {
            row["url"]: row["link_type"]
            for row in pipeline.conn.execute(
                "SELECT url, link_type FROM paper_text_links WHERE run_id = ? AND paper_id = ?",
                (full_text_run_id, first_paper_id),
            ).fetchall()
        }
    finally:
        pipeline.close()

    assert pdf_artifact is not None
    assert pdf_artifact["fetch_status"] == "fetched"
    assert links[str(FIXTURES / "example-fulltext.pdf")] == "pdf"
    assert links["https://arxiv.org/abs/2501.12345"] == "paper_landing"


def test_discover_full_text_uses_title_web_search_pdf_candidates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WIRELESS_TAXONOMY_ENABLE_WEB_SEARCH", "1")
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._openalex_candidates", lambda doi, title=None: [])
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._crossref_candidates", lambda doi: [])
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._semantic_scholar_candidates", lambda doi, title, authors=None: [])
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._unpaywall_candidates", lambda doi: [])
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._arxiv_candidates", lambda title: [])
    monkeypatch.setattr(
        "wireless_taxonomy.analyze.full_text._web_search_candidates",
        lambda title: [("web_search_result_pdf", "https://authors.example.edu/example-wireless-dataset-paper.pdf")],
    )

    def fake_fetch_bytes(url: str) -> bytes:
        assert url == "https://authors.example.edu/example-wireless-dataset-paper.pdf"
        return b"Example Wireless Dataset Paper\nWe use dataset named Open5G Measurements with RF measurements and SINR traces."

    monkeypatch.setattr(full_text, "_fetch_bytes", fake_fetch_bytes)
    db = tmp_path / "taxonomy.sqlite"
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        first_paper_id = pipeline.conn.execute(
            "SELECT id FROM papers WHERE title = ?",
            ("Example Wireless Dataset Paper",),
        ).fetchone()["id"]
        full_text_run_id = pipeline.discover_full_text(run_id, paper_id=first_paper_id)
        pdf_artifact = pipeline.conn.execute(
            """
            SELECT * FROM paper_text_artifacts
            WHERE run_id = ? AND paper_id = ? AND source_type = 'pdf_text'
            """,
            (full_text_run_id, first_paper_id),
        ).fetchone()
        pdf_link = pipeline.conn.execute(
            """
            SELECT * FROM paper_text_links
            WHERE run_id = ? AND paper_id = ? AND link_text = 'web_search_result_pdf'
            """,
            (full_text_run_id, first_paper_id),
        ).fetchone()
    finally:
        pipeline.close()

    assert pdf_artifact is not None
    assert pdf_artifact["fetch_status"] == "fetched"
    assert "Open5G Measurements" in pdf_artifact["content_text"]
    assert pdf_link is not None
    assert pdf_link["link_type"] == "pdf"


def test_discover_full_text_aggregates_failed_pdf_reviews_per_paper(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WIRELESS_TAXONOMY_ENABLE_WEB_SEARCH", "1")
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._openalex_candidates", lambda doi, title=None: [])
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._crossref_candidates", lambda doi: [])
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._semantic_scholar_candidates", lambda doi, title, authors=None: [])
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._unpaywall_candidates", lambda doi: [])
    monkeypatch.setattr(
        "wireless_taxonomy.analyze.full_text._arxiv_candidates",
        lambda title: [
            ("arxiv_pdf", "https://arxiv.org/pdf/2501.00001"),
            ("arxiv_pdf", "https://arxiv.org/pdf/2501.00002"),
        ],
    )
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._openreview_candidates", lambda title: [])
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._web_search_candidates", lambda title: [])

    def fake_fetch_bytes(url: str) -> bytes:
        raise OSError(f"blocked {url.rsplit('/', 1)[-1]}")

    monkeypatch.setattr(full_text, "_fetch_bytes", fake_fetch_bytes)
    db = tmp_path / "taxonomy.sqlite"
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        first_paper_id = pipeline.conn.execute(
            "SELECT id FROM papers WHERE title = ?",
            ("Example Wireless Dataset Paper",),
        ).fetchone()["id"]
        pipeline.conn.execute("UPDATE papers SET doi = NULL, paper_url = NULL, pdf_url = NULL WHERE id = ?", (first_paper_id,))
        pipeline.conn.commit()
        full_text_run_id = pipeline.discover_full_text(run_id, paper_id=first_paper_id)
        error_artifacts = pipeline.conn.execute(
            """
            SELECT COUNT(*) AS count FROM paper_text_artifacts
            WHERE run_id = ? AND paper_id = ? AND source_type = 'pdf_text' AND fetch_status = 'error'
            """,
            (full_text_run_id, first_paper_id),
        ).fetchone()["count"]
        review_rows = pipeline.conn.execute(
            "SELECT * FROM review_items WHERE run_id = ?",
            (full_text_run_id,),
        ).fetchall()
    finally:
        pipeline.close()

    assert error_artifacts == 2
    assert len(review_rows) == 1
    assert "2 candidate PDF(s), 0 fetched" == review_rows[0]["suggested_value"]
    assert "blocked" in review_rows[0]["evidence"]


def test_discover_full_text_caches_resolver_candidates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WIRELESS_TAXONOMY_ENABLE_WEB_SEARCH", "1")
    calls = {"openalex": 0}

    def fake_openalex(doi, title=None):
        calls["openalex"] += 1
        return [("openalex_pdf", str(FIXTURES / "example-fulltext.pdf"))]

    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._openalex_candidates", fake_openalex)
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._crossref_candidates", lambda doi: [])
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._semantic_scholar_candidates", lambda doi, title, authors=None: [])
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._unpaywall_candidates", lambda doi: [])
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._arxiv_candidates", lambda title: [])
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._openreview_candidates", lambda title: [])
    monkeypatch.setattr("wireless_taxonomy.analyze.full_text._web_search_candidates", lambda title: [])
    db = tmp_path / "taxonomy.sqlite"
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        first_paper_id = pipeline.conn.execute(
            "SELECT id FROM papers WHERE title = ?",
            ("Example Wireless Dataset Paper",),
        ).fetchone()["id"]
        pipeline.discover_full_text(run_id, paper_id=first_paper_id)
        pipeline.discover_full_text(run_id, paper_id=first_paper_id)
        cache_rows = pipeline.conn.execute(
            "SELECT COUNT(*) AS count FROM resolver_cache WHERE provider = 'openalex'"
        ).fetchone()["count"]
    finally:
        pipeline.close()

    assert calls["openalex"] == 1
    assert cache_rows == 1


def test_resolver_cache_does_not_store_transient_failures() -> None:
    class FakeCache:
        def __init__(self) -> None:
            self.saved = False

        def get_candidates(self, provider, cache_key):
            return None

        def set_candidates(self, provider, cache_key, candidates):
            self.saved = True

    cache = FakeCache()
    discoverer = full_text.FullTextDiscoverer(allow_remote=True, candidate_cache=cache)

    def failing_resolver():
        raise full_text.TransientResolverError("429")

    result = discoverer._cached_candidates("semantic_scholar", "paper-key", failing_resolver)

    assert result == []
    assert cache.saved is False


def test_semantic_scholar_fetch_uses_api_key_rate_gate_and_429_retry(monkeypatch) -> None:
    class FakeRateLimiter:
        def __init__(self) -> None:
            self.wait_count = 0

        def wait(self) -> None:
            self.wait_count += 1

    limiter = FakeRateLimiter()
    calls = []
    sleeps = []

    def fake_fetch_json_once(url: str, headers: dict[str, str]) -> dict:
        calls.append(headers.copy())
        if len(calls) == 1:
            raise full_text.HTTPError(url, 429, "Too Many Requests", {"Retry-After": "1.5"}, None)
        return {"data": []}

    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    monkeypatch.setenv("S2_API_KEY", "s2-secret")
    monkeypatch.setenv("WIRELESS_TAXONOMY_SEMANTIC_SCHOLAR_RETRIES", "2")
    monkeypatch.setattr(full_text, "_SEMANTIC_SCHOLAR_RATE_LIMITER", limiter)
    monkeypatch.setattr(full_text, "_fetch_json_once", fake_fetch_json_once)
    monkeypatch.setattr(full_text.time, "sleep", lambda seconds: sleeps.append(seconds))

    payload = full_text._fetch_json("https://api.semanticscholar.org/graph/v1/paper/search?query=example")

    assert payload == {"data": []}
    assert limiter.wait_count == 2
    assert calls[0]["x-api-key"] == "s2-secret"
    assert sleeps == [2.2]


def test_openreview_pdf_fetch_uses_rate_gate_and_429_retry(monkeypatch) -> None:
    class FakeRateLimiter:
        def __init__(self) -> None:
            self.wait_count = 0

        def wait(self) -> None:
            self.wait_count += 1

    limiter = FakeRateLimiter()
    calls = []
    sleeps = []

    def fake_fetch_bytes_once(url: str, headers: dict[str, str]) -> bytes:
        calls.append((url, headers.copy()))
        if len(calls) == 1:
            raise full_text.HTTPError(url, 429, "Too Many Requests", {"Retry-After": "1"}, None)
        return b"%PDF example"

    monkeypatch.setenv("WIRELESS_TAXONOMY_OPENREVIEW_RETRIES", "2")
    monkeypatch.setenv("WIRELESS_TAXONOMY_OPENREVIEW_MIN_INTERVAL_SECONDS", "1.25")
    monkeypatch.setattr(full_text, "_OPENREVIEW_RATE_LIMITER", limiter)
    monkeypatch.setattr(full_text, "_fetch_bytes_once", fake_fetch_bytes_once)
    monkeypatch.setattr(full_text.time, "sleep", lambda seconds: sleeps.append(seconds))

    data = full_text._fetch_bytes("https://openreview.net/pdf?id=abc123")

    assert data == b"%PDF example"
    assert limiter.wait_count == 2
    assert calls[0][1]["User-Agent"] == "wireless-taxonomy/0.1"
    assert sleeps == [2.5]


def test_openreview_resolver_extracts_pdf_and_landing_candidates(monkeypatch) -> None:
    title = "Example Wireless Dataset Paper"
    monkeypatch.setattr(
        full_text,
        "_fetch_json",
        lambda url: {
            "notes": [
                {
                    "id": "note123",
                    "forum": "forum123",
                    "content": {"title": {"value": title}},
                }
            ]
        },
    )

    candidates = full_text._openreview_candidates(title)
    urls = {url for _, url in candidates}

    assert "https://openreview.net/forum?id=forum123" in urls
    assert "https://openreview.net/pdf?id=forum123" in urls


def test_semantic_scholar_resolver_accepts_title_author_sibling_records(monkeypatch) -> None:
    title = "Example Wireless Dataset Paper"
    calls = []

    def fake_json(url: str) -> dict:
        calls.append(url)
        if "/paper/DOI:" in url:
            return {
                "paperId": "official",
                "url": "https://www.semanticscholar.org/paper/official",
                "title": title,
                "externalIds": {"DOI": "10.1145/123.456"},
            }
        return {
            "data": [
                {
                    "paperId": "wrong",
                    "title": "A Different Dataset Paper",
                    "authors": [{"name": "Unrelated Author"}],
                    "openAccessPdf": {"url": "https://wrong.example/paper.pdf"},
                },
                {
                    "paperId": "preprint",
                    "url": "https://www.semanticscholar.org/paper/preprint",
                    "title": title,
                    "authors": [{"name": "Ada Lovelace"}, {"name": "Grace Hopper"}],
                    "openAccessPdf": {"url": "https://arxiv.org/pdf/2501.12345"},
                    "externalIds": {"ArXiv": "2501.12345"},
                },
            ]
        }

    monkeypatch.setattr(full_text, "_fetch_json", fake_json)

    candidates = full_text._semantic_scholar_candidates("10.1145/123.456", title, "Ada Lovelace, Grace Hopper")
    urls = {url for _, url in candidates}

    assert any("limit=10" in call for call in calls)
    assert "https://arxiv.org/pdf/2501.12345" in urls
    assert "https://arxiv.org/abs/2501.12345" in urls
    assert "https://wrong.example/paper.pdf" not in urls


def test_add_pdfs_matches_local_pdf_by_filename_and_persists_text(tmp_path: Path) -> None:
    db = tmp_path / "taxonomy.sqlite"
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    (pdf_dir / "example wireless dataset paper.pdf").write_bytes((FIXTURES / "example-fulltext.pdf").read_bytes())
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        stage_run = pipeline.add_pdfs(run_id, pdf_dir)
        artifact = pipeline.conn.execute(
            """
            SELECT pta.*, p.title
            FROM paper_text_artifacts pta
            JOIN papers p ON p.id = pta.paper_id
            WHERE pta.run_id = ? AND pta.source_type = 'local_pdf_text'
            """,
            (stage_run,),
        ).fetchone()
        link = pipeline.conn.execute(
            "SELECT * FROM paper_text_links WHERE run_id = ? AND link_text LIKE 'local_pdf:%'",
            (stage_run,),
        ).fetchone()
    finally:
        pipeline.close()

    assert artifact is not None
    assert artifact["title"] == "Example Wireless Dataset Paper"
    assert artifact["fetch_status"] == "fetched"
    assert "Open5G Measurements" in artifact["content_text"]
    assert link is not None
    assert link["link_type"] == "pdf"


def test_acm_browser_pdf_url_uses_acm_doi() -> None:
    assert acm_browser._acm_pdf_url({"doi": "10.1145/3718958.3750520"}) == "https://dl.acm.org/doi/pdf/10.1145/3718958.3750520"
    assert acm_browser._acm_pdf_url({"paper_url": "https://dl.acm.org/doi/10.1145/3718958.3750520"}) == (
        "https://dl.acm.org/doi/pdf/10.1145/3718958.3750520"
    )


def test_fetch_acm_browser_persists_authenticated_pdf_text(tmp_path: Path, monkeypatch) -> None:
    class FakeAcmFetcher:
        provider_name = "fake_acm_browser"

        def __init__(self, profile_dir, headless=False, browser_channel=None, cdp_url=None, delay_seconds=None):
            self.profile_dir = profile_dir

        def fetch_many(self, rows, limit=None):
            paper = list(rows)[0]
            artifact = PaperTextArtifact(
                paper_id=paper["id"],
                source_type="acm_browser_pdf_text",
                source_url="https://dl.acm.org/doi/pdf/10.1145/123.456",
                fetch_status="fetched",
                content_text="Example Wireless Dataset Paper uses Open5G Measurements dataset with RF measurements.",
                content_sha256="fakehash",
                error_message=None,
            )
            return [
                PaperTextEnrichment(
                    paper_id=paper["id"],
                    artifacts=[artifact],
                    links=[PaperTextLink(paper["id"], artifact.source_url or "", "acm_browser_pdf", "pdf", 0.95)],
                    snippets=[],
                )
            ]

    monkeypatch.setattr("wireless_taxonomy.pipeline.AuthenticatedAcmBrowserFetcher", FakeAcmFetcher)
    db = tmp_path / "taxonomy.sqlite"
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        first_paper_id = pipeline.conn.execute(
            "SELECT id FROM papers WHERE title = ?",
            ("Example Wireless Dataset Paper",),
        ).fetchone()["id"]
        pipeline.conn.execute("UPDATE papers SET doi = ? WHERE id = ?", ("10.1145/123.456", first_paper_id))
        pipeline.conn.commit()
        stage_run = pipeline.fetch_acm_browser(run_id, tmp_path / "profile", paper_id=first_paper_id)
        artifact = pipeline.conn.execute(
            "SELECT * FROM paper_text_artifacts WHERE run_id = ? AND source_type = 'acm_browser_pdf_text'",
            (stage_run,),
        ).fetchone()
        link = pipeline.conn.execute(
            "SELECT * FROM paper_text_links WHERE run_id = ? AND link_text = 'acm_browser_pdf'",
            (stage_run,),
        ).fetchone()
    finally:
        pipeline.close()

    assert artifact is not None
    assert artifact["fetch_status"] == "fetched"
    assert "Open5G Measurements" in artifact["content_text"]
    assert link is not None
    assert link["link_type"] == "pdf"


def test_programmatic_full_text_resolvers_cover_openalex_unpaywall_and_arxiv(monkeypatch) -> None:
    title = "Example Wireless Dataset Paper"
    arxiv_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>https://arxiv.org/abs/2501.12345</id>
        <title>{title}</title>
        <link href="https://arxiv.org/pdf/2501.12345" title="pdf" type="application/pdf" />
      </entry>
    </feed>
    """.encode()

    def fake_json(url: str) -> dict:
        if "api.openalex.org/works?" in url:
            return {
                "results": [
                    {
                        "title": title,
                        "best_oa_location": {
                            "pdf_url": "https://open.example/paper.pdf",
                            "landing_page_url": "https://open.example/paper",
                        },
                    }
                ]
            }
        if "api.unpaywall.org" in url:
            return {
                "best_oa_location": {
                    "url_for_pdf": "https://repo.example/paper.pdf",
                    "url_for_landing_page": "https://repo.example/paper",
                }
            }
        return {}

    monkeypatch.setenv("WIRELESS_TAXONOMY_UNPAYWALL_EMAIL", "taxonomy@example.com")
    monkeypatch.setattr(full_text, "_fetch_json", fake_json)
    monkeypatch.setattr(full_text, "_fetch_bytes", lambda url: arxiv_xml)

    candidates = (
        full_text._openalex_candidates(None, title)
        + full_text._unpaywall_candidates("10.1145/123.456")
        + full_text._arxiv_candidates(title)
    )
    urls = {url for _, url in candidates}

    assert "https://open.example/paper.pdf" in urls
    assert "https://repo.example/paper.pdf" in urls
    assert "https://arxiv.org/pdf/2501.12345" in urls
    assert full_text._unique(["https://arxiv.org/pdf/2408.04275", "https://arxiv.org/pdf/2408.04275v4"]) == [
        "https://arxiv.org/pdf/2408.04275"
    ]


def test_arxiv_resolver_falls_back_to_relaxed_title_search(monkeypatch) -> None:
    title = "InfiniteHBD: Building Datacenter-Scale High-Bandwidth Domain for LLM"
    empty_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"></feed>
    """
    match_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>https://arxiv.org/abs/2502.03885v6</id>
        <title>{title}</title>
        <link href="https://arxiv.org/pdf/2502.03885v6" title="pdf" type="application/pdf" />
      </entry>
    </feed>
    """.encode()
    calls = []

    def fake_fetch_bytes(url: str) -> bytes:
        calls.append(url)
        return empty_xml if len(calls) == 1 else match_xml

    monkeypatch.setattr(full_text, "_fetch_bytes", fake_fetch_bytes)

    candidates = full_text._arxiv_candidates(title)
    urls = {url for _, url in candidates}

    assert len(calls) >= 2
    assert "https://arxiv.org/pdf/2502.03885v6" in urls


def test_agentic_paper_analysis_writes_structured_claims_and_workbook_links(tmp_path: Path) -> None:
    db = tmp_path / "taxonomy.sqlite"
    paper_page = FIXTURES / "example_paper_page.html"
    out = tmp_path / "agentic.json"
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        first_paper_id = pipeline.conn.execute(
            "SELECT id FROM papers WHERE title = ?",
            ("Example Wireless Dataset Paper",),
        ).fetchone()["id"]
        pipeline.conn.execute("UPDATE papers SET paper_url = ? WHERE id = ?", (str(paper_page), first_paper_id))
        pipeline.conn.commit()
        pipeline.enrich_paper_text(run_id)

        analysis_run_id = pipeline.agentic_paper_analysis(run_id, paper_id=first_paper_id)
        pipeline.export(analysis_run_id, str(out), "json", scope="exact")
        analysis = pipeline.conn.execute(
            "SELECT * FROM paper_agentic_analyses WHERE run_id = ? AND paper_id = ?",
            (analysis_run_id, first_paper_id),
        ).fetchone()
        dataset_claim = pipeline.conn.execute(
            "SELECT * FROM paper_analysis_dataset_claims WHERE run_id = ?",
            (analysis_run_id,),
        ).fetchone()
        workbook_link = pipeline.conn.execute(
            """
            SELECT d.canonical_name, pdl.relationship_type, pdl.review_needed
            FROM paper_dataset_links pdl
            JOIN datasets d ON d.id = pdl.dataset_id
            WHERE pdl.run_id = ?
            """,
            (analysis_run_id,),
        ).fetchone()
        evidence_types = {
            row["claim_type"]
            for row in pipeline.conn.execute(
                "SELECT claim_type FROM evidence_claims WHERE run_id = ?",
                (analysis_run_id,),
            ).fetchall()
        }
    finally:
        pipeline.close()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert analysis is not None
    assert analysis["wireless_label"] == "yes"
    assert analysis["is_wireless"] == 1
    assert json.loads(analysis["modalities_json"]) == ["RF measurements", "SINR", "throughput"]
    assert json.loads(analysis["osi_layers_json"]) == ["L1", "L4", "L7"]
    assert dataset_claim is not None
    assert dataset_claim["dataset_name"] == "Open5G Measurements"
    assert dataset_claim["review_needed"] == 0
    assert workbook_link is not None
    assert workbook_link["canonical_name"] == "Open5G Measurements"
    assert {"agentic_wireless_classification", "agentic_dataset_claim", "modality", "osi_layer"} <= evidence_types
    assert len(payload["paper_agentic_analyses"]) == 1
    assert payload["paper_agentic_analyses"][0]["analysis"]["dataset_claims"][0]["dataset_name"] == "Open5G Measurements"


def test_agentic_paper_analysis_can_use_llm_json_contract(tmp_path: Path, monkeypatch) -> None:
    class FakeRouter:
        def __init__(self, settings):
            self.settings = settings

        def complete(self, request):
            parsed = {
                "wireless": {
                    "label": "yes",
                    "is_wireless": True,
                    "confidence": 0.97,
                    "evidence": "The paper discusses RF measurements and SINR traces.",
                },
                "modalities": ["RF measurements", "SINR"],
                "osi_layers": ["L1"],
                "datasets": [
                    {
                        "dataset_name": "Open5G Measurements",
                        "relationship_type": "reused",
                        "confidence": 0.94,
                        "evidence_text": "We use dataset named Open5G Measurements with RF measurements.",
                        "source_url": "https://example.org/open5g",
                        "modalities": ["RF measurements", "SINR"],
                        "osi_layers": ["L1"],
                        "availability_status": "unclear",
                    }
                ],
                "summary": "LLM structured analysis fixture.",
                "review_needed": False,
            }
            return LlmResponse("fake", "fake-model", json.dumps(parsed), parsed)

    monkeypatch.setattr("wireless_taxonomy.analyze.agentic_paper.LlmRouter", FakeRouter)
    db = tmp_path / "taxonomy.sqlite"
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        first_paper_id = pipeline.conn.execute(
            "SELECT id FROM papers WHERE title = ?",
            ("Example Wireless Dataset Paper",),
        ).fetchone()["id"]
        analysis_run_id = pipeline.agentic_paper_analysis(run_id, paper_id=first_paper_id, use_llm=True)
        analysis = pipeline.conn.execute(
            "SELECT * FROM paper_agentic_analyses WHERE run_id = ?",
            (analysis_run_id,),
        ).fetchone()
        dataset_claim = pipeline.conn.execute(
            "SELECT * FROM paper_analysis_dataset_claims WHERE run_id = ?",
            (analysis_run_id,),
        ).fetchone()
    finally:
        pipeline.close()

    assert analysis is not None
    assert analysis["provider_name"] == "llm_agentic_paper_v0:fake:fake-model"
    assert analysis["wireless_confidence"] == 0.97
    assert json.loads(analysis["modalities_json"]) == ["RF measurements", "SINR"]
    assert dataset_claim is not None
    assert dataset_claim["dataset_name"] == "Open5G Measurements"
    assert dataset_claim["source_url"] == "https://example.org/open5g"


def test_reflect_paper_analysis_flags_ungrounded_llm_claims(tmp_path: Path, monkeypatch) -> None:
    class FakeRouter:
        def __init__(self, settings):
            self.settings = settings

        def complete(self, request):
            parsed = {
                "wireless": {
                    "label": "yes",
                    "is_wireless": True,
                    "confidence": 0.99,
                    "evidence": "This paper proves quantum underwater networking with zebra traces.",
                },
                "modalities": [],
                "osi_layers": [],
                "datasets": [
                    {
                        "dataset_name": "Dataset",
                        "relationship_type": "reused",
                        "confidence": 0.99,
                        "evidence_text": "The paper uses ZebraNet-9000 traces.",
                        "source_url": "https://example.org/zebranet",
                        "modalities": [],
                        "osi_layers": [],
                    }
                ],
                "summary": "Unsupported hallucinated fixture.",
                "review_needed": False,
            }
            return LlmResponse("fake", "fake-model", json.dumps(parsed), parsed)

    monkeypatch.setattr("wireless_taxonomy.analyze.agentic_paper.LlmRouter", FakeRouter)
    db = tmp_path / "taxonomy.sqlite"
    out = tmp_path / "reflection.json"
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        first_paper_id = pipeline.conn.execute(
            "SELECT id FROM papers WHERE title = ?",
            ("Example Wireless Dataset Paper",),
        ).fetchone()["id"]
        pipeline.enrich_paper_text(run_id)
        analysis_run_id = pipeline.agentic_paper_analysis(run_id, paper_id=first_paper_id, use_llm=True)
        reflection_run_id = pipeline.reflect_paper_analysis(run_id, analysis_run_id=analysis_run_id, paper_id=first_paper_id)
        pipeline.export(reflection_run_id, str(out), "json", scope="exact")
        reflection = pipeline.conn.execute(
            "SELECT * FROM paper_analysis_reflections WHERE run_id = ? AND paper_id = ?",
            (reflection_run_id, first_paper_id),
        ).fetchone()
        dataset_claim = pipeline.conn.execute(
            "SELECT * FROM paper_analysis_dataset_claims WHERE run_id = ? AND paper_id = ?",
            (analysis_run_id, first_paper_id),
        ).fetchone()
        review_count = pipeline.conn.execute(
            "SELECT COUNT(*) AS count FROM review_items WHERE run_id = ?",
            (reflection_run_id,),
        ).fetchone()["count"]
    finally:
        pipeline.close()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert reflection is not None
    assert reflection["decision"] == "review"
    assert dataset_claim is not None
    assert dataset_claim["review_needed"] == 1
    assert review_count >= 1
    assert payload["paper_analysis_reflections"][0]["issues"]


def test_scope_assessment_flags_unrelated_sources(tmp_path: Path) -> None:
    db = tmp_path / "taxonomy.sqlite"
    csv_path = tmp_path / "biology.csv"
    csv_path.write_text(
        "Paper Title,Authors,Conference,Year\n"
        "Single Cell Atlas of Mouse Liver,Ada Lovelace,BioConf,2025\n"
        "Protein Folding Dynamics in Yeast,Grace Hopper,BioConf,2025\n"
        "Genome Wide Association Study of Wheat,Claude Shannon,BioConf,2025\n",
        encoding="utf-8",
    )
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("BioConf", 2025, "csv", str(csv_path))
        scope_run_id = pipeline.assess_scope(run_id)
        assessment = pipeline.latest_scope_assessment(scope_run_id)
        review_count = pipeline.conn.execute(
            "SELECT COUNT(*) AS count FROM review_items WHERE run_id = ?",
            (scope_run_id,),
        ).fetchone()["count"]
    finally:
        pipeline.close()

    assert assessment is not None
    assert assessment["decision"] == "likely_out_of_scope"
    assert assessment["should_proceed"] == 0
    assert assessment["networking_like_count"] == 0
    assert review_count >= 1


def test_pipeline_through_paper_text_enrichment_has_coherent_audit_trail(tmp_path: Path) -> None:
    db = tmp_path / "taxonomy.sqlite"
    paper_page = FIXTURES / "example_paper_page.html"
    pipeline = Pipeline(load_settings(db))
    try:
        ingest_run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        first_paper_id = pipeline.conn.execute(
            "SELECT id FROM papers WHERE title = ?",
            ("Example Wireless Dataset Paper",),
        ).fetchone()["id"]
        pipeline.conn.execute("UPDATE papers SET paper_url = ? WHERE id = ?", (str(paper_page), first_paper_id))
        pipeline.conn.commit()

        scope_run_id = pipeline.assess_scope(ingest_run_id)
        verify_run_id = pipeline.verify_paper_list(ingest_run_id)
        enrich_run_id = pipeline.enrich_paper_text(ingest_run_id)
        audit_dir = tmp_path / "audit_json"
        ingest_json = pipeline.export(ingest_run_id, str(audit_dir / "run_001_ingest.json"), "json", scope="exact")
        scope_json = pipeline.export(scope_run_id, str(audit_dir / "run_002_assess-scope.json"), "json", scope="exact")
        verify_json = pipeline.export(verify_run_id, str(audit_dir / "run_003_verify-paper-list.json"), "json", scope="exact")
        enrich_json = pipeline.export(enrich_run_id, str(audit_dir / "run_004_enrich-paper-text.json"), "json", scope="exact")

        stages = [
            row["stage"]
            for row in pipeline.conn.execute("SELECT stage FROM pipeline_runs ORDER BY id").fetchall()
        ]
        report = pipeline.latest_paper_list_report(verify_run_id)
        artifacts = pipeline.conn.execute(
            """
            SELECT p.title, pta.source_type, pta.fetch_status, pta.source_url, pta.content_sha256
            FROM paper_text_artifacts pta
            JOIN papers p ON p.id = pta.paper_id
            WHERE pta.run_id = ?
            ORDER BY p.title, pta.source_type
            """,
            (enrich_run_id,),
        ).fetchall()
        link_types = {
            row["link_type"]
            for row in pipeline.conn.execute(
                "SELECT link_type FROM paper_text_links WHERE run_id = ?",
                (enrich_run_id,),
            ).fetchall()
        }
        snippet_count = pipeline.conn.execute(
            "SELECT COUNT(*) AS count FROM paper_text_snippets WHERE run_id = ?",
            (enrich_run_id,),
        ).fetchone()["count"]
        evidence_types = {
            row["claim_type"]
            for row in pipeline.conn.execute(
                "SELECT claim_type FROM evidence_claims WHERE run_id = ?",
                (enrich_run_id,),
            ).fetchall()
        }
        pending_enrichment_reviews = pipeline.conn.execute(
            "SELECT COUNT(*) AS count FROM review_items WHERE run_id = ?",
            (enrich_run_id,),
        ).fetchone()["count"]
    finally:
        pipeline.close()

    ingest_payload = json.loads(ingest_json.read_text(encoding="utf-8"))
    scope_payload = json.loads(scope_json.read_text(encoding="utf-8"))
    verify_payload = json.loads(verify_json.read_text(encoding="utf-8"))
    enrich_payload = json.loads(enrich_json.read_text(encoding="utf-8"))

    assert stages == ["ingest", "assess-scope", "verify-paper-list", "enrich-paper-text"]
    assert report is not None
    assert report["final_confidence"] == 1.0
    assert len(artifacts) >= 4
    assert {row["source_type"] for row in artifacts} >= {"abstract", "paper_url", "pdf_reference"}
    assert {row["fetch_status"] for row in artifacts} >= {"available", "fetched", "reference_only", "remote_skipped"}
    assert all(row["content_sha256"] for row in artifacts)
    assert {"pdf", "dataset_or_artifact", "repository"} <= link_types
    assert snippet_count >= 1
    assert {"paper_text_artifact", "paper_text_snippet"} <= evidence_types
    assert pending_enrichment_reviews == 0
    assert ingest_payload["scope"] == "exact"
    assert [run["stage"] for run in ingest_payload["runs"]] == ["ingest"]
    assert len(ingest_payload["paper_text_artifacts"]) == 0
    assert len(ingest_payload["evidence_claims"]) == 2
    assert len(ingest_payload["scope_assessments"]) == 0
    assert scope_payload["scope"] == "exact"
    assert [run["stage"] for run in scope_payload["runs"]] == ["assess-scope"]
    assert len(scope_payload["scope_assessments"]) == 1
    assert len(scope_payload["paper_text_artifacts"]) == 0
    assert {claim["claim_type"] for claim in scope_payload["evidence_claims"]} == {"scope_assessment"}
    assert verify_payload["scope"] == "exact"
    assert [run["stage"] for run in verify_payload["runs"]] == ["verify-paper-list"]
    assert len(verify_payload["paper_list_verification_reports"]) == 1
    assert len(verify_payload["paper_text_artifacts"]) == 0
    assert enrich_payload["scope"] == "exact"
    assert [run["stage"] for run in enrich_payload["runs"]] == ["enrich-paper-text"]
    assert len(enrich_payload["paper_text_artifacts"]) == len(artifacts)
    assert len(enrich_payload["paper_text_snippets"]) == snippet_count
    assert {claim["claim_type"] for claim in enrich_payload["evidence_claims"]} == {
        "paper_text_artifact",
        "paper_text_snippet",
    }
