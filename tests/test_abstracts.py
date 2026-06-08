from pathlib import Path

from wireless_taxonomy.analyze.abstracts import (
    AbstractEnricher,
    DoiResolver,
    _normalize_doi_url,
    _openalex_abstract,
    _strip_jats,
    _usenix_abstract,
)
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


_USENIX_HTML = """
<html><head><title>Flow Scheduling | USENIX</title></head><body>
<h1 class="title">Flow Scheduling with Imprecise Knowledge</h1>
<div class="field field-name-field-paper-person">Wenxin Li, Tianjin University</div>
<div class="field field-name-field-paper-description field-type-text-long field-label-hidden">
<div class="field-items"><div class="field-item odd"><p>We present QCLIMB, a new flow
scheduling solution designed to minimize FCT by utilizing imprecise flow information
from machine learning techniques.</p></div></div></div>
<div class="bibtex-accordion">BibTeX @inproceedings{li, title={...}} NSDI '24 Open Access</div>
</body></html>
"""


def test_usenix_abstract_extraction_strips_tags_and_trailers() -> None:
    abstract = _usenix_abstract(_USENIX_HTML)
    assert abstract.startswith("We present QCLIMB")
    assert abstract.endswith("machine learning techniques.")
    assert "BibTeX" not in abstract
    assert "Open Access" not in abstract
    assert "field-name" not in abstract


def test_enricher_usenix_fallback_used_when_apis_empty() -> None:
    enricher = AbstractEnricher(
        fetch_json=lambda url: {},  # OpenAlex/Crossref/S2 all empty
        fetch_text=lambda url: _USENIX_HTML,
    )
    result = enricher.fetch(
        "Flow Scheduling with Imprecise Knowledge",
        None,
        "https://www.usenix.org/conference/nsdi24/presentation/li-wenxin",
    )
    assert result is not None
    assert result.provider == "usenix"
    assert "QCLIMB" in result.abstract


def test_enricher_usenix_skipped_for_non_usenix_url() -> None:
    calls: list[str] = []

    def fetch_text(url: str) -> str:
        calls.append(url)
        return _USENIX_HTML

    enricher = AbstractEnricher(fetch_json=lambda url: {}, fetch_text=fetch_text)
    assert enricher.fetch("Title", None, "https://dblp.org/db/conf/nsdi/nsdi2024.html") is None
    # The USENIX provider never fetches a non-USENIX page; only arXiv's title
    # search may use fetch_text (against export.arxiv.org), never the DBLP URL.
    assert all("dblp.org" not in url for url in calls)


def test_enricher_usenix_rejects_title_mismatch() -> None:
    enricher = AbstractEnricher(fetch_json=lambda url: {}, fetch_text=lambda url: _USENIX_HTML)
    result = enricher.fetch(
        "A Totally Unrelated Antenna Paper",
        None,
        "https://www.usenix.org/conference/nsdi24/presentation/li-wenxin",
    )
    assert result is None


def test_doi_resolver_crossref_first_then_openalex() -> None:
    def fake_fetch(url: str) -> dict:
        if "crossref" in url:
            return {"message": {"items": [{"DOI": "10.1145/ABC", "title": ["Some Wireless Paper"]}]}}
        return {}

    resolver = DoiResolver(fetch_json=fake_fetch)
    result = resolver.resolve("Some Wireless Paper")
    assert result is not None
    assert result.provider == "crossref"
    assert result.doi == "10.1145/abc"


def test_doi_resolver_rejects_title_mismatch() -> None:
    def fake_fetch(url: str) -> dict:
        if "crossref" in url:
            return {"message": {"items": [{"DOI": "10.1/x", "title": ["A Totally Different Unrelated Paper"]}]}}
        return {}

    resolver = DoiResolver(fetch_json=fake_fetch)
    assert resolver.resolve("Some Wireless Paper") is None


def test_doi_resolver_openalex_fallback_strips_url_prefix() -> None:
    def fake_fetch(url: str) -> dict:
        if "openalex" in url:
            return {"results": [{"doi": "https://doi.org/10.1/Wireless", "title": "Some Wireless Paper"}]}
        return {}

    resolver = DoiResolver(fetch_json=fake_fetch)
    result = resolver.resolve("Some Wireless Paper")
    assert result is not None
    assert result.provider == "openalex"
    assert result.doi == "10.1/wireless"


def test_normalize_doi_url_variants() -> None:
    assert _normalize_doi_url("https://doi.org/10.1/x") == "10.1/x"
    assert _normalize_doi_url("doi:10.1/x") == "10.1/x"
    assert _normalize_doi_url("10.1/x") == "10.1/x"


def test_enrich_abstracts_pipeline_fills_missing(tmp_path: Path) -> None:
    db = tmp_path / "taxonomy.sqlite"

    class FakeEnricher:
        def fetch(self, title, doi, url=None):
            from wireless_taxonomy.analyze.abstracts import AbstractResult

            return AbstractResult("Backfilled wireless abstract about CSI.", "openalex", "http://x")

    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        pipeline.conn.execute("UPDATE papers SET abstract = NULL")
        pipeline.conn.commit()
        pipeline.enrich_abstracts(run_id, enricher=FakeEnricher(), resolve_dois=False)
        abstracts = [row["abstract"] for row in pipeline.conn.execute("SELECT abstract FROM papers")]
    finally:
        pipeline.close()
    assert all(a == "Backfilled wireless abstract about CSI." for a in abstracts)


def test_enrich_abstracts_pipeline_backfills_missing_dois(tmp_path: Path) -> None:
    db = tmp_path / "taxonomy.sqlite"

    class FakeEnricher:
        def fetch(self, title, doi, url=None):
            return None

    class FakeResolver:
        def resolve(self, title):
            from wireless_taxonomy.analyze.abstracts import DoiResult

            return DoiResult("10.9999/backfilled", "crossref", "http://x")

    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        pipeline.conn.execute("UPDATE papers SET doi = NULL")
        pipeline.conn.commit()
        pipeline.enrich_abstracts(run_id, enricher=FakeEnricher(), doi_resolver=FakeResolver())
        dois = [row["doi"] for row in pipeline.conn.execute("SELECT doi FROM papers")]
    finally:
        pipeline.close()
    assert all(d == "10.9999/backfilled" for d in dois)


_ARXIV_XML = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Massive MIMO Beamforming for Wireless Sensing</title>
    <summary>We present a wireless sensing system that uses massive MIMO
    beamforming and CSI features to localize devices indoors.</summary>
  </entry>
</feed>"""


def test_enricher_arxiv_matches_title_and_returns_abstract() -> None:
    enricher = AbstractEnricher(
        fetch_json=lambda url: {},  # OpenAlex/Crossref/S2 empty
        fetch_text=lambda url: _ARXIV_XML if "arxiv.org" in url else "",
    )
    result = enricher.fetch("Massive MIMO Beamforming for Wireless Sensing", None)
    assert result is not None
    assert result.provider == "arxiv"
    assert "CSI features" in result.abstract


def test_enricher_arxiv_rejects_title_mismatch() -> None:
    enricher = AbstractEnricher(
        fetch_json=lambda url: {},
        fetch_text=lambda url: _ARXIV_XML if "arxiv.org" in url else "",
    )
    # arXiv's top hit is unrelated -> the title guard rejects it.
    assert enricher.fetch("A Completely Different Optical Networking Paper", None) is None


def test_cache_roundtrip(tmp_path: Path) -> None:
    from wireless_taxonomy.analyze.cache import MetadataCache

    path = tmp_path / "cache.json"
    cache = MetadataCache(path)
    cache.set_abstract("Some Title", "10.1/abc", {"abstract": "X" * 50, "provider": "openalex", "source_url": "u"})
    cache.set_doi("Other Title", {"doi": "10.2/def", "provider": "crossref", "source_url": "u2"})
    cache.save()
    assert path.exists()

    reloaded = MetadataCache(path)
    # Hit by DOI and by normalized (case-insensitive) title.
    assert reloaded.get_abstract(None, "10.1/ABC")["abstract"] == "X" * 50
    assert reloaded.get_abstract("some title", None)["provider"] == "openalex"
    assert reloaded.get_doi("OTHER TITLE")["doi"] == "10.2/def"


def test_enricher_cache_short_circuits_network(tmp_path: Path) -> None:
    from wireless_taxonomy.analyze.cache import MetadataCache

    cache = MetadataCache(tmp_path / "c.json")
    calls: list[str] = []

    def counting_fetch(url: str) -> dict:
        calls.append(url)
        long = "A wireless CSI sensing abstract with enough characters to pass. " * 2
        return {"message": {"abstract": long}} if "crossref" in url else {}

    enricher = AbstractEnricher(fetch_json=counting_fetch, fetch_text=lambda url: "", cache=cache)
    first = enricher.fetch("Cached Paper", "10.1/cached")
    assert first is not None and first.provider == "crossref"
    n_after_first = len(calls)

    # Second fetch for the same paper must hit the cache, not the network.
    second = enricher.fetch("Cached Paper", "10.1/cached")
    assert second is not None
    assert second.abstract == first.abstract
    assert len(calls) == n_after_first  # no additional network calls


def test_enricher_caches_and_short_circuits_misses(tmp_path: Path) -> None:
    from wireless_taxonomy.analyze.cache import MetadataCache

    cache = MetadataCache(tmp_path / "miss.json")
    calls: list[str] = []

    enricher = AbstractEnricher(
        fetch_json=lambda url: (calls.append(url) or {}),  # every provider misses
        fetch_text=lambda url: (calls.append(url) or ""),
        cache=cache,
    )
    assert enricher.fetch("Unfindable Paper", "10.1/none") is None
    n = len(calls)
    assert n > 0  # the first attempt hit the network

    # Second attempt for the same paper must be served from the negative cache.
    assert enricher.fetch("Unfindable Paper", "10.1/none") is None
    assert len(calls) == n  # no further network calls


def test_acm_abstract_parses_block_and_meta() -> None:
    from wireless_taxonomy.analyze.abstracts import _acm_abstract

    block_html = (
        '<section id="abstract"><div role="paragraph">'
        "This paper measures 5G performance in the wild across many cities and operators."
        "</div></section>"
    )
    assert "5G performance" in _acm_abstract(block_html)

    meta_html = (
        '<meta name="dc.Description" content="A measurement study of Starlink LEO latency over time.">'
    )
    assert "Starlink" in _acm_abstract(meta_html)
    assert _acm_abstract("<html>no abstract here</html>") == ""
