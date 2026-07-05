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


def test_acm_doi_redirects_are_not_fetchable() -> None:
    """OpenAlex reports doi.org/10.1145 URLs as OA, but they redirect to ACM."""
    resolver = OpenAccessResolver(
        fetch_json=_fetch_json([("api.openalex.org", {
            "title": "A Wireless Paper",
            "open_access": {"is_oa": True, "oa_status": "gold", "oa_url": "https://doi.org/10.1145/123456.789012"},
            "best_oa_location": {"pdf_url": "https://doi.org/10.1145/123456.789012", "license": ""},
        })]),
        fetch_text=lambda u: "",
        providers=["openalex"],
    )
    res = resolver.resolve("A Wireless Paper", "10.1145/123456.789012")
    assert res.fetchable is False
    assert res.provider == "none"


def test_acm_direct_url_is_not_fetchable() -> None:
    """Direct dl.acm.org PDF URLs are also blocked."""
    resolver = OpenAccessResolver(
        fetch_json=_fetch_json([("api.semanticscholar.org", {
            "openAccessPdf": {"url": "https://dl.acm.org/doi/pdf/10.1145/123", "status": "GREEN", "license": ""},
        })]),
        fetch_text=lambda u: "",
        providers=["semantic_scholar"],
    )
    res = resolver.resolve("A Wireless Paper", "10.1145/123456.789012")
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
    monkeypatch.delenv("WIRELESS_TAXONOMY_UNPAYWALL_EMAIL", raising=False)
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
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

def test_web_search_provider_appended_only_when_enabled(monkeypatch) -> None:
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)
    resolver = OpenAccessResolver(fetch_json=lambda url: {}, fetch_text=lambda url: "")
    assert "llm_web_search" not in resolver.providers
    resolver_ws = OpenAccessResolver(
        fetch_json=lambda url: {}, fetch_text=lambda url: "", web_search=True
    )
    assert resolver_ws.providers[-1] == "llm_web_search"


def test_web_search_rejects_blocked_and_unverified_urls(monkeypatch) -> None:
    class FakeRouter:
        def __init__(self, url):
            self.url = url

        def complete(self, request):
            from wireless_taxonomy.llm import LlmResponse
            return LlmResponse("google", "gemini-test", "{}", {"pdf_url": self.url})

    resolver = OpenAccessResolver(
        fetch_json=lambda url: {}, fetch_text=lambda url: "",
        web_search=True, router=FakeRouter("https://dl.acm.org/doi/pdf/10.1145/x"),
    )
    assert resolver._llm_web_search("A Paper", None, None) is None  # ACM blocked

    resolver = OpenAccessResolver(
        fetch_json=lambda url: {}, fetch_text=lambda url: "",
        web_search=True, router=FakeRouter("https://ieeexplore.ieee.org/document/1"),
    )
    assert resolver._llm_web_search("A Paper", None, None) is None  # IEEE blocked

    # Good URL but PDF verification fails -> rejected.
    monkeypatch.setattr(
        "wireless_taxonomy.analyze.dataset_extractor._fetch_pdf_bytes",
        lambda url, expected_title=None: None,
    )
    resolver = OpenAccessResolver(
        fetch_json=lambda url: {}, fetch_text=lambda url: "",
        web_search=True, router=FakeRouter("https://example.edu/paper.pdf"),
    )
    assert resolver._llm_web_search("A Paper", None, None) is None


def test_web_search_accepts_verified_pdf(monkeypatch) -> None:
    class FakeRouter:
        def complete(self, request):
            from wireless_taxonomy.llm import LlmResponse
            return LlmResponse(
                "google", "gemini-test", "{}", {"pdf_url": "https://example.edu/paper.pdf"}
            )

    monkeypatch.setattr(
        "wireless_taxonomy.analyze.dataset_extractor._fetch_pdf_bytes",
        lambda url, expected_title=None: b"%PDF-1.4 verified",
    )
    resolver = OpenAccessResolver(
        fetch_json=lambda url: {}, fetch_text=lambda url: "",
        web_search=True, router=FakeRouter(),
    )
    result = resolver._llm_web_search("A Paper", "10.1/x", None)
    assert result is not None and result.fetchable
    assert result.pdf_url == "https://example.edu/paper.pdf"
    assert result.provider == "llm_web_search"


def test_cached_closed_reresolved_when_web_search_enabled(tmp_path: Path) -> None:
    """A timestamped closed entry that was never web-searched is retried when
    web search is now enabled. Legacy entries (no searched_at) are honoured to
    avoid burning Brave/CSE budget on papers that are almost certainly still closed."""
    import time as _t
    path = tmp_path / "cache.json"
    cache = MetadataCache(path)
    # Cached miss with a timestamp but no web search attempted — e.g. was
    # resolved before Brave was configured but after the timestamp field existed.
    cache.set_oa("A Paper", "10.1/x", {
        "fetchable": False, "oa_status": "closed", "license": "",
        "pdf_url": "", "provider": "none", "source_url": "",
        "web_search_attempted": False, "web_search_providers": [],
        "searched_at": _t.time() - 3600,  # 1 hour ago — fresh but no web search
    })

    calls = []
    resolver = OpenAccessResolver(
        fetch_json=lambda url: (calls.append(url) or {}),
        fetch_text=lambda url: (calls.append(url) or ""),
        cache=cache,
        web_search=True,
        router=None,
    )
    resolver.providers = ["openalex", "llm_web_search"]  # skip network-heavy chain

    # 1) Web search ERRORS (rate limit / network): the verdict must stay
    #    retryable — a failed attempt is not an attempt.
    resolver._router = type("R", (), {"complete": lambda self, req: (_ for _ in ()).throw(RuntimeError())})()
    resolver.resolve("A Paper", "10.1/x")
    assert calls  # cache was bypassed and providers were re-queried
    got = cache.get_oa("A Paper", "10.1/x")
    assert got["web_search_attempted"] is False

    # 2) Web search RUNS and finds nothing: now the attempt is recorded...
    class _NoResult:
        parsed = {"pdf_url": ""}

    resolver._router = type("R", (), {"complete": lambda self, req: _NoResult()})()
    resolver.resolve("A Paper", "10.1/x")
    got = cache.get_oa("A Paper", "10.1/x")
    assert got["web_search_attempted"] is True

    # ...so a third resolve is served from cache (no new provider calls).
    n = len(calls)
    resolver.resolve("A Paper", "10.1/x")
    assert len(calls) == n


def test_legacy_closed_entry_not_retried_with_web_search(tmp_path: Path) -> None:
    """Legacy entries (no searched_at) are honoured unconditionally — they went
    through the free waterfall already and re-searching with Brave every run
    wastes paid quota on papers almost certainly still closed."""
    path = tmp_path / "cache.json"
    cache = MetadataCache(path)
    # Legacy entry: no searched_at, no web_search_attempted field.
    cache.set_oa("A Paper", "10.1/x", {
        "fetchable": False, "oa_status": "closed", "license": "",
        "pdf_url": "", "provider": "none", "source_url": "",
    })

    calls = []
    resolver = OpenAccessResolver(
        fetch_json=lambda url: (calls.append(url) or {}),
        fetch_text=lambda url: (calls.append(url) or ""),
        cache=cache,
        web_search=True,
        router=None,
    )
    resolver.providers = ["openalex", "llm_web_search"]

    resolver._router = type("R", (), {"complete": lambda self, req: (_ for _ in ()).throw(RuntimeError())})()
    resolver.resolve("A Paper", "10.1/x")
    assert calls == []  # legacy entry honoured — no Brave/CSE spend


# ── Google CSE provider ──────────────────────────────────────────────────────

_CSE_ITEMS = {
    "items": [
        {"link": "https://dl.acm.org/doi/pdf/10.1145/1.2"},          # blocked domain
        {"link": "https://other.edu/wrong-paper.pdf"},                # fails title check
        {"link": "https://cs.stanford.edu/~author/paper.pdf"},        # verified hit
    ]
}


def _cse_resolver(monkeypatch, fetch_json):
    monkeypatch.setenv("GOOGLE_CSE_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_CSE_ID", "cx")
    return OpenAccessResolver(fetch_json=fetch_json, fetch_text=lambda u: "", web_search=True)


def test_web_search_prefers_google_cse_when_configured(monkeypatch) -> None:
    resolver = _cse_resolver(monkeypatch, lambda url: {})
    assert "google_cse" in resolver.providers
    assert "llm_web_search" not in resolver.providers


def test_web_search_falls_back_to_llm_without_cse_keys(monkeypatch) -> None:
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)
    resolver = OpenAccessResolver(fetch_json=lambda u: {}, fetch_text=lambda u: "", web_search=True)
    assert "llm_web_search" in resolver.providers
    assert "google_cse" not in resolver.providers


def test_google_cse_only_trusts_title_verified_pdf(monkeypatch) -> None:
    """Blocked domains are skipped, unverified PDFs rejected, verified hit wins."""
    resolver = _cse_resolver(monkeypatch, _fetch_json([("customsearch", _CSE_ITEMS)]))
    resolver.providers = ["google_cse"]

    verified: list[str] = []

    def fake_fetch(url, max_bytes=0, expected_title=None, attempts=3):
        verified.append(url)
        return b"%PDF" if "stanford" in url else None

    import wireless_taxonomy.analyze.dataset_extractor as de

    monkeypatch.setattr(de, "_fetch_pdf_bytes", fake_fetch)
    res = resolver.resolve("A Wireless Paper", "10.1/x")
    assert res.fetchable
    assert res.provider == "google_cse"
    assert res.pdf_url == "https://cs.stanford.edu/~author/paper.pdf"
    # ACM link never reached verification; the other two were checked in order.
    assert verified == ["https://other.edu/wrong-paper.pdf", "https://cs.stanford.edu/~author/paper.pdf"]


def test_google_cse_no_verified_hit_returns_closed_but_attempted(monkeypatch, tmp_path) -> None:
    cache = MetadataCache(tmp_path / "c.json")
    resolver = _cse_resolver(monkeypatch, _fetch_json([("customsearch", _CSE_ITEMS)]))
    resolver.cache = cache
    resolver.providers = ["google_cse"]

    import wireless_taxonomy.analyze.dataset_extractor as de

    monkeypatch.setattr(de, "_fetch_pdf_bytes", lambda *a, **k: None)
    res = resolver.resolve("A Wireless Paper", "10.1/x")
    assert not res.fetchable
    got = cache.get_oa("A Wireless Paper", "10.1/x")
    assert got["web_search_attempted"] is True  # ran and found nothing — cache it


def test_google_cse_quota_error_stays_retryable(monkeypatch, tmp_path) -> None:
    cache = MetadataCache(tmp_path / "c.json")

    def exploding_fetch(url):
        if "customsearch" in url:
            raise RuntimeError("HTTP 429: daily quota exceeded")
        return {}

    resolver = _cse_resolver(monkeypatch, exploding_fetch)
    resolver.cache = cache
    resolver.providers = ["google_cse"]
    res = resolver.resolve("A Wireless Paper", "10.1/x")
    assert not res.fetchable
    got = cache.get_oa("A Wireless Paper", "10.1/x")
    assert got["web_search_attempted"] is False  # search never ran — retry next time


# ── CORE provider ────────────────────────────────────────────────────────────

_CORE_RESULTS = {
    "results": [
        {"title": "A Totally Different Paper", "downloadUrl": "https://repo.edu/other.pdf"},
        {
            "title": "A Wireless Paper",
            "downloadUrl": "https://core.ac.uk/download/123.pdf",
            "sourceFulltextUrls": ["https://repo.uni.edu/paper.pdf"],
        },
    ]
}


def test_core_provider_title_matched_and_verified(monkeypatch) -> None:
    monkeypatch.setenv("CORE_API_KEY", "ck")
    resolver = OpenAccessResolver(
        fetch_json=_fetch_json([("api.core.ac.uk", _CORE_RESULTS)]), fetch_text=lambda u: ""
    )
    assert "core" in resolver.providers
    resolver.providers = ["core"]

    verified: list[str] = []

    def fake_fetch(url, max_bytes=0, expected_title=None, attempts=3):
        verified.append(url)
        return b"%PDF" if "repo.uni.edu" in url else None

    import wireless_taxonomy.analyze.dataset_extractor as de

    monkeypatch.setattr(de, "_fetch_pdf_bytes", fake_fetch)
    res = resolver.resolve("A Wireless Paper", "10.1/x")
    assert res.fetchable
    assert res.provider == "core"
    assert res.pdf_url == "https://repo.uni.edu/paper.pdf"
    # Wrong-title result skipped entirely; downloadUrl tried before source URL.
    assert verified == ["https://core.ac.uk/download/123.pdf", "https://repo.uni.edu/paper.pdf"]


def test_core_provider_skipped_without_key(monkeypatch) -> None:
    monkeypatch.delenv("CORE_API_KEY", raising=False)
    resolver = OpenAccessResolver(fetch_json=lambda u: {}, fetch_text=lambda u: "")
    assert "core" not in resolver.providers


def test_core_throttle_enforces_min_interval(monkeypatch) -> None:
    monkeypatch.setenv("WIRELESS_TAXONOMY_CORE_MIN_INTERVAL_SECONDS", "2.5")
    sleeps: list[float] = []
    import wireless_taxonomy.analyze.oa_availability as oa

    monkeypatch.setattr(oa.time, "sleep", lambda s: sleeps.append(s))
    clock = {"t": 100.0}
    monkeypatch.setattr(oa.time, "monotonic", lambda: clock["t"])
    OpenAccessResolver._core_last_call = 0.0
    OpenAccessResolver._core_throttle()  # first call: no wait
    assert sleeps == []
    clock["t"] = 100.5  # only 0.5s later
    OpenAccessResolver._core_throttle()  # second call: must wait ~2.0s
    assert len(sleeps) == 1 and abs(sleeps[0] - 2.0) < 0.01
    OpenAccessResolver._core_last_call = 0.0


# ── Brave Search provider ────────────────────────────────────────────────────

_BRAVE_RESULTS = {
    "web": {
        "results": [
            {"url": "https://dl.acm.org/doi/pdf/10.1145/1.2"},          # blocked domain
            {"url": "https://other.edu/wrong-paper.pdf"},                # fails title check
            {"url": "https://cs.stanford.edu/~author/paper.pdf"},        # verified hit
        ]
    }
}


def _brave_resolver(monkeypatch, fetch_json):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "bk")
    monkeypatch.delenv("GOOGLE_CSE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)
    return OpenAccessResolver(fetch_json=fetch_json, fetch_text=lambda u: "", web_search=True)


def test_web_search_prefers_brave_when_configured(monkeypatch) -> None:
    resolver = _brave_resolver(monkeypatch, lambda url: {})
    assert "brave_search" in resolver.providers
    assert "llm_web_search" not in resolver.providers


def test_brave_and_cse_both_run_when_both_configured(monkeypatch) -> None:
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "bk")
    monkeypatch.setenv("GOOGLE_CSE_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_CSE_ID", "cx")
    resolver = OpenAccessResolver(fetch_json=lambda u: {}, fetch_text=lambda u: "", web_search=True)
    assert resolver.providers.index("brave_search") < resolver.providers.index("google_cse")
    assert "llm_web_search" not in resolver.providers


def test_brave_only_trusts_title_verified_pdf(monkeypatch) -> None:
    """Blocked domains skipped, unverified PDFs rejected, verified hit wins."""
    monkeypatch.setenv("WIRELESS_TAXONOMY_BRAVE_MIN_INTERVAL_SECONDS", "0")
    resolver = _brave_resolver(monkeypatch, _fetch_json([("api.search.brave.com", _BRAVE_RESULTS)]))
    resolver.providers = ["brave_search"]

    verified: list[str] = []

    def fake_fetch(url, max_bytes=0, expected_title=None, attempts=3):
        verified.append(url)
        return b"%PDF" if "stanford" in url else None

    import wireless_taxonomy.analyze.dataset_extractor as de

    monkeypatch.setattr(de, "_fetch_pdf_bytes", fake_fetch)
    res = resolver.resolve("A Wireless Paper", "10.1/x")
    assert res.fetchable
    assert res.provider == "brave_search"
    assert res.pdf_url == "https://cs.stanford.edu/~author/paper.pdf"
    assert verified == ["https://other.edu/wrong-paper.pdf", "https://cs.stanford.edu/~author/paper.pdf"]


def test_brave_rate_limit_error_stays_retryable(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WIRELESS_TAXONOMY_BRAVE_MIN_INTERVAL_SECONDS", "0")
    cache = MetadataCache(tmp_path / "c.json")

    def exploding_fetch(url):
        if "api.search.brave.com" in url:
            raise RuntimeError("HTTP 429: rate limit")
        return {}

    resolver = _brave_resolver(monkeypatch, exploding_fetch)
    resolver.cache = cache
    resolver.providers = ["brave_search"]
    res = resolver.resolve("A Wireless Paper", "10.1/x")
    assert not res.fetchable
    got = cache.get_oa("A Wireless Paper", "10.1/x")
    assert got["web_search_attempted"] is False  # search never ran — retry next time


def test_closed_verdict_searched_by_old_provider_is_retried_with_new(monkeypatch, tmp_path) -> None:
    """A closed verdict web-searched by Gemini must be re-resolved once
    Brave/CSE are configured — the new providers may find what Gemini missed."""
    monkeypatch.setenv("WIRELESS_TAXONOMY_BRAVE_MIN_INTERVAL_SECONDS", "0")
    cache = MetadataCache(tmp_path / "c.json")
    # Simulate an old cache entry: closed, searched by the Gemini provider.
    import time as _t
    cache.set_oa("A Wireless Paper", "10.1/x", {
        "fetchable": False, "oa_status": "closed", "license": "", "pdf_url": "",
        "provider": "none", "source_url": "",
        "web_search_attempted": True, "web_search_providers": ["llm_web_search"],
        "searched_at": _t.time() - 3600,  # recent but different provider set
    })

    resolver = _brave_resolver(monkeypatch, _fetch_json([("api.search.brave.com", _BRAVE_RESULTS)]))
    resolver.cache = cache
    resolver.providers = ["brave_search"]

    import wireless_taxonomy.analyze.dataset_extractor as de
    monkeypatch.setattr(de, "_fetch_pdf_bytes", lambda url, **k: b"%PDF" if "stanford" in url else None)

    res = resolver.resolve("A Wireless Paper", "10.1/x")
    assert res.fetchable  # Brave re-searched and found the Stanford mirror
    assert res.provider == "brave_search"
    got = cache.get_oa("A Wireless Paper", "10.1/x")
    assert got["fetchable"] is True


def test_closed_verdict_searched_by_same_providers_is_not_researched(monkeypatch, tmp_path) -> None:
    cache = MetadataCache(tmp_path / "c.json")
    cache.set_oa("A Wireless Paper", "10.1/x", {
        "fetchable": False, "oa_status": "closed", "license": "", "pdf_url": "",
        "provider": "none", "source_url": "",
        "web_search_attempted": True, "web_search_providers": ["brave_search"],
    })
    calls = []

    def counting_fetch(url):
        calls.append(url)
        return {}

    resolver = _brave_resolver(monkeypatch, counting_fetch)
    resolver.cache = cache
    resolver.providers = ["brave_search"]
    res = resolver.resolve("A Wireless Paper", "10.1/x")
    assert not res.fetchable
    assert calls == []  # served from cache; no re-search


def test_stale_negative_retried_without_web_search(tmp_path) -> None:
    """A cached 'closed' verdict older than the TTL is re-resolved even when
    web search is disabled — papers appear on arXiv/repos over time."""
    import time as _time
    cache = MetadataCache(tmp_path / "c.json")
    cache.set_oa("A Wireless Paper", "10.1/x", {
        "fetchable": False, "oa_status": "closed", "license": "", "pdf_url": "",
        "provider": "none", "source_url": "",
        "web_search_attempted": False, "web_search_providers": [],
        "searched_at": _time.time() - 90 * 86400,  # 90 days old — stale
    })
    resolver = OpenAccessResolver(
        fetch_json=_fetch_json([("api.openalex.org", _OPENALEX_OA)]),
        fetch_text=lambda u: "",
        providers=["openalex"],
        cache=cache,
    )
    res = resolver.resolve("A Wireless Paper", "10.1/x")
    assert res.fetchable  # stale negative was retried and now found


def test_fresh_negative_honoured_without_web_search(tmp_path) -> None:
    """A cached 'closed' verdict within the TTL is served from cache."""
    import time as _time
    cache = MetadataCache(tmp_path / "c.json")
    cache.set_oa("A Wireless Paper", "10.1/x", {
        "fetchable": False, "oa_status": "closed", "license": "", "pdf_url": "",
        "provider": "none", "source_url": "",
        "web_search_attempted": False, "web_search_providers": [],
        "searched_at": _time.time() - 3600,  # 1 hour old — fresh
    })
    calls = []

    def counting_fetch(url):
        calls.append(url)
        return {}

    resolver = OpenAccessResolver(
        fetch_json=counting_fetch,
        fetch_text=lambda u: "",
        providers=["openalex"],
        cache=cache,
    )
    res = resolver.resolve("A Wireless Paper", "10.1/x")
    assert not res.fetchable
    assert calls == []  # served from cache; no network


def test_positive_verdict_cached_indefinitely(tmp_path) -> None:
    """A found PDF URL is never re-resolved regardless of age."""
    import time as _time
    cache = MetadataCache(tmp_path / "c.json")
    cache.set_oa("A Wireless Paper", "10.1/x", {
        "fetchable": True, "oa_status": "green", "license": "cc-by",
        "pdf_url": "https://arxiv.org/pdf/1234.5678", "provider": "arxiv",
        "source_url": "", "web_search_attempted": False,
        "web_search_providers": [], "searched_at": _time.time() - 365 * 86400,
    })
    calls = []

    def counting_fetch(url):
        calls.append(url)
        return {}

    resolver = OpenAccessResolver(
        fetch_json=counting_fetch,
        fetch_text=lambda u: "",
        providers=["openalex"],
        cache=cache,
    )
    res = resolver.resolve("A Wireless Paper", "10.1/x")
    assert res.fetchable
    assert res.pdf_url == "https://arxiv.org/pdf/1234.5678"
    assert calls == []  # positives never expire


def test_google_cse_query_cap(monkeypatch):
    """CSE queries beyond the budget raise a retryable _WebSearchError."""
    from wireless_taxonomy.analyze.oa_availability import OpenAccessResolver, _WebSearchError
    import pytest

    monkeypatch.setenv("WIRELESS_TAXONOMY_CSE_MAX_QUERIES", "2")
    OpenAccessResolver._cse_queries = 0  # reset class counter
    OpenAccessResolver._check_cse_cap()
    OpenAccessResolver._check_cse_cap()
    with pytest.raises(_WebSearchError, match="budget exhausted"):
        OpenAccessResolver._check_cse_cap()
    OpenAccessResolver._cse_queries = 0  # clean up for other tests
