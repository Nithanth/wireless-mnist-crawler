from pathlib import Path

from wireless_taxonomy.analyze.cache import MetadataCache
from wireless_taxonomy.analyze.oa_availability import OpenAccessResolver, summarize


def _fetch_json(routes):
    def f(url):
        for sub, payload in routes:
            if sub in url:
                return payload
        return {}

    return f


_OPENALEX_OA = {
    "title": "A Wireless Paper",
    "open_access": {"is_oa": True, "oa_status": "gold", "oa_url": "https://x/oa"},
    "best_oa_location": {"pdf_url": "https://x/pdf", "license": "cc-by"},
}


def test_openalex_oa_hit_is_fetchable() -> None:
    resolver = OpenAccessResolver(
        fetch_json=_fetch_json([("api.openalex.org", _OPENALEX_OA)]),
        fetch_text=lambda u: "",
        providers=["openalex"],
    )
    res = resolver.resolve("A Wireless Paper", "10.1/x")
    assert res.fetchable
    assert res.oa_status == "gold"
    assert res.provider == "openalex"
    assert res.pdf_url == "https://x/pdf"
    assert res.license == "cc-by"


def test_semantic_scholar_open_access_pdf() -> None:
    s2 = {"openAccessPdf": {"url": "https://x/pdf", "status": "GREEN", "license": "CC-BY"}}
    resolver = OpenAccessResolver(
        fetch_json=_fetch_json([("semanticscholar.org", s2)]),
        fetch_text=lambda u: "",
        providers=["semantic_scholar"],
    )
    res = resolver.resolve("A Wireless Paper", "10.1/x")
    assert res.fetchable
    assert res.oa_status == "green"
    assert res.provider == "semantic_scholar"


def test_arxiv_url_is_fetchable_without_network() -> None:
    resolver = OpenAccessResolver(providers=["arxiv"])
    res = resolver.resolve("A Wireless Paper", None, url="https://arxiv.org/abs/2401.00001")
    assert res.fetchable
    assert res.provider == "arxiv"
    assert res.pdf_url == "https://arxiv.org/pdf/2401.00001"


def test_usenix_url_is_open_access_without_network() -> None:
    page = "https://www.usenix.org/conference/nsdi24/presentation/author"
    resolver = OpenAccessResolver(
        fetch_json=_fetch_json([]),  # must not need the network
        fetch_text=lambda u: "",
        providers=["usenix", "openalex"],
    )
    res = resolver.resolve("A USENIX Paper", None, url=page)
    assert res.fetchable
    assert res.provider == "usenix"
    assert res.oa_status == "gold"
    assert res.pdf_url == page


def test_closed_paper_is_not_fetchable() -> None:
    resolver = OpenAccessResolver(
        fetch_json=_fetch_json([]),
        fetch_text=lambda u: "",
        providers=["openalex", "semantic_scholar", "arxiv"],
    )
    res = resolver.resolve("Some Paywalled Paper", "10.1/closed")
    assert res.fetchable is False
    assert res.oa_status == "closed"
    assert res.provider == "none"


def test_provider_order_first_oa_wins(monkeypatch) -> None:
    monkeypatch.setenv("WIRELESS_TAXONOMY_CONTACT_EMAIL", "a@b.c")
    up = {"is_oa": True, "oa_status": "hybrid", "best_oa_location": {"url_for_pdf": "https://u/pdf", "license": "cc"}}
    resolver = OpenAccessResolver(
        fetch_json=_fetch_json([("unpaywall.org", up), ("api.openalex.org", _OPENALEX_OA)]),
        fetch_text=lambda u: "",
        providers=["unpaywall", "openalex"],
    )
    res = resolver.resolve("A Wireless Paper", "10.1/x")
    assert res.provider == "unpaywall"
    assert res.oa_status == "hybrid"


def test_unpaywall_skipped_without_email(monkeypatch) -> None:
    monkeypatch.delenv("WIRELESS_TAXONOMY_CONTACT_EMAIL", raising=False)
    up = {"is_oa": True, "best_oa_location": {"url_for_pdf": "https://u/pdf"}}
    resolver = OpenAccessResolver(
        fetch_json=_fetch_json([("unpaywall.org", up)]),
        fetch_text=lambda u: "",
        providers=["unpaywall"],
    )
    assert resolver.resolve("A Wireless Paper", "10.1/x").fetchable is False


def test_cache_short_circuits_second_lookup() -> None:
    cache = MetadataCache(None)
    warm = OpenAccessResolver(
        fetch_json=_fetch_json([("api.openalex.org", _OPENALEX_OA)]),
        fetch_text=lambda u: "",
        providers=["openalex"],
        cache=cache,
    )
    first = warm.resolve("A Wireless Paper", "10.1/x")
    assert first.fetchable

    cold = OpenAccessResolver(
        fetch_json=_fetch_json([]),  # would miss if it hit the network
        fetch_text=lambda u: "",
        providers=["openalex"],
        cache=cache,
    )
    second = cold.resolve("A Wireless Paper", "10.1/x")
    assert second.fetchable
    assert second.oa_status == "gold"
    assert second.provider == "openalex"


def test_summarize_counts_and_percentage() -> None:
    papers = [
        {"fetchable": True, "oa_status": "gold", "provider": "openalex"},
        {"fetchable": True, "oa_status": "green", "provider": "arxiv"},
        {"fetchable": False, "oa_status": "closed", "provider": "none"},
    ]
    s = summarize(papers)
    assert s["total_papers"] == 3
    assert s["fetchable"] == 2
    assert s["fetchable_pct"] == 66.7
    assert s["by_oa_status"] == {"gold": 1, "green": 1}
    assert s["by_source"] == {"arxiv": 1, "openalex": 1}


def test_metadata_cache_oa_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    cache = MetadataCache(path)
    cache.set_oa(
        "A Wireless Paper",
        "10.1/x",
        {
            "fetchable": True,
            "oa_status": "gold",
            "license": "cc-by",
            "pdf_url": "https://x/pdf",
            "provider": "openalex",
            "source_url": "https://api.openalex.org/works/...",
        },
    )
    cache.save()

    reloaded = MetadataCache(path)
    got = reloaded.get_oa("A Wireless Paper", "10.1/x")
    assert got is not None
    assert got["fetchable"] is True
    assert got["oa_status"] == "gold"
    # set_oa indexes the same record under both the DOI key and the title key.
    assert reloaded.stats()["oa"] == 2
    assert reloaded.get_oa(None, "10.1/x") == reloaded.get_oa("A Wireless Paper", None)
