"""Tests for DatasetExtractor, focusing on the PDF text fallback path."""

from unittest.mock import MagicMock, patch

import pytest

from wireless_taxonomy.analyze.dataset_extractor import (
    DatasetExtractor,
    _fetch_pdf_bytes,
    _is_acm_blocked,
    _title_words_in_text,
)


def _mock_router(fail_with_pdf=False, response_json=None):
    """Create a mock LlmRouter that optionally fails when pdf_bytes is set."""
    if response_json is None:
        response_json = {"datasets": [{"name": "TestDataset", "relationship_type": "introduced",
            "modalities": ["traces"], "osi_layers": ["L3"], "availability": None,
            "availability_url": "", "availability_notes": "", "collection_environment": "Real World Deployment",
            "known_users": [], "confidence": 0.9, "evidence_text": "We collected traces."}]}

    def complete(request):
        if fail_with_pdf and request.pdf_bytes:
            raise RuntimeError("HTTP 400: content policy violation")
        resp = MagicMock()
        resp.parsed = response_json
        resp.content = str(response_json)
        resp.provider = "test"
        resp.model = "test-model"
        return resp

    router = MagicMock()
    router.complete = MagicMock(side_effect=complete)
    return router


@patch("wireless_taxonomy.analyze.dataset_extractor._fetch_pdf_bytes", return_value=b"%PDF-1.4 fake pdf content with enough bytes to pass the length check" * 10)
@patch("wireless_taxonomy.analyze.dataset_extractor._fetch_crossref_bibtex", return_value=None)
@patch("wireless_taxonomy.analyze.dataset_extractor._check_url_live", return_value=None)
def test_text_fallback_on_pdf_rejection(mock_url, mock_bib, mock_fetch):
    """When PDF-as-bytes fails all providers, extractor falls back to pypdf text."""
    router = _mock_router(fail_with_pdf=True)
    extractor = DatasetExtractor(router=router, cache=None, conn=None)

    with patch("wireless_taxonomy.llm._pdf_bytes_to_text", return_value="This is a wireless paper about 5G measurements " * 50):
        result = extractor.extract(
            paper_id=1, title="Test Paper", authors="Smith, J.",
            venue="NSDI", year=2024, doi="10.1/test",
            pdf_url="https://example.com/paper.pdf", abstract="Test abstract",
        )

    assert not result.error, f"Expected no error but got: {result.error}"
    assert result.extraction_source == "pdf_text_fallback"
    assert len(result.datasets) == 1
    assert result.datasets[0].name == "TestDataset"
    # First call was with pdf_bytes (failed), second was text-only (succeeded)
    assert router.complete.call_count == 2
    second_call = router.complete.call_args_list[1]
    assert second_call[0][0].pdf_bytes is None


@patch("wireless_taxonomy.analyze.dataset_extractor._fetch_pdf_bytes", return_value=None)
@patch("wireless_taxonomy.analyze.dataset_extractor._fetch_crossref_bibtex", return_value=None)
@patch("wireless_taxonomy.analyze.dataset_extractor._check_url_live", return_value=None)
def test_abstract_only_no_fallback_needed(mock_url, mock_bib, mock_fetch):
    """When there's no PDF, extraction uses abstract directly (no fallback)."""
    router = _mock_router(fail_with_pdf=False)
    extractor = DatasetExtractor(router=router, cache=None, conn=None)

    result = extractor.extract(
        paper_id=2, title="Abstract Paper", authors="Jones, A.",
        venue="SIGCOMM", year=2023, doi="", pdf_url=None,
        abstract="We measured 5G network performance.",
    )

    assert not result.error
    assert result.extraction_source == "abstract"
    assert router.complete.call_count == 1
    assert router.complete.call_args_list[0][0][0].pdf_bytes is None


@patch("wireless_taxonomy.analyze.dataset_extractor._fetch_pdf_bytes", return_value=b"%PDF-1.4 fake" * 10)
@patch("wireless_taxonomy.analyze.dataset_extractor._fetch_crossref_bibtex", return_value=None)
@patch("wireless_taxonomy.analyze.dataset_extractor._check_url_live", return_value=None)
def test_text_fallback_short_extraction_returns_error(mock_url, mock_bib, mock_fetch):
    """When PDF fails and pypdf extracts too little text, return error."""
    router = _mock_router(fail_with_pdf=True)
    extractor = DatasetExtractor(router=router, cache=None, conn=None)

    with patch("wireless_taxonomy.llm._pdf_bytes_to_text", return_value="short"):
        result = extractor.extract(
            paper_id=3, title="Bad PDF Paper", authors="Brown, B.",
            venue="IMC", year=2022, doi="", pdf_url="https://example.com/bad.pdf",
            abstract=None,
        )

    assert result.error
    assert "too short" in result.error
    assert result.datasets == []


class TestAcmBlocked:
    def test_direct_acm_url(self):
        assert _is_acm_blocked("https://dl.acm.org/doi/pdf/10.1145/123")

    def test_acm_doi_redirect(self):
        assert _is_acm_blocked("https://doi.org/10.1145/3730567.3764475")

    def test_arxiv_not_blocked(self):
        assert not _is_acm_blocked("https://arxiv.org/pdf/2505.21733v2")

    def test_usenix_not_blocked(self):
        assert not _is_acm_blocked("https://www.usenix.org/system/files/nsdi24-paper.pdf")

    def test_non_acm_doi_not_blocked(self):
        assert not _is_acm_blocked("https://doi.org/10.1109/TWC.2024.123")


class TestTitleWordsInText:
    def test_exact_title_present(self):
        assert _title_words_in_text(
            "Efficient Multi-WAN Transport for 5G with OTTER",
            "Efficient Multi-WAN Transport for 5G with OTTER Alice Bob Abstract We present...",
        )

    def test_hyphenated_linebreak_tolerated(self):
        # "Communication" broken across lines still passes word-threshold matching.
        assert _title_words_in_text(
            "Radar Backscatter Communication with Low-power Tags",
            "Radar Backscatter Communica- tion with Low-power Tags University Abstract",
        )

    def test_wrong_paper_rejected(self):
        assert not _title_words_in_text(
            "Massive MIMO Baseband Processing on a Single Server",
            "A Study of Coral Reef Bleaching Patterns in the Pacific Ocean Abstract We surveyed...",
        )

    def test_short_words_ignored(self):
        # Only words > 3 chars count, so stop-words don't inflate the match.
        assert not _title_words_in_text(
            "On the Use of the Web for the Study of DNS",
            "Completely unrelated document about biology and chemistry experiments",
        )


class TestFetchPdfBytes:
    def _response(self, body: bytes):
        resp = MagicMock()
        resp.read = MagicMock(return_value=body)
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def test_valid_pdf_returned(self):
        pdf = b"%PDF-1.4 content"
        with patch("urllib.request.urlopen", return_value=self._response(pdf)):
            assert _fetch_pdf_bytes("https://x/paper.pdf") == pdf

    def test_non_pdf_rejected(self):
        with patch("urllib.request.urlopen", return_value=self._response(b"<html>landing page</html>")):
            assert _fetch_pdf_bytes("https://x/paper.pdf") is None

    def test_oversized_pdf_rejected_not_truncated(self):
        # A body larger than max_bytes must be rejected, not silently truncated.
        big = b"%PDF" + b"x" * 100
        with patch("urllib.request.urlopen", return_value=self._response(big)):
            assert _fetch_pdf_bytes("https://x/paper.pdf", max_bytes=50) is None

    def test_transient_error_retried(self):
        pdf = b"%PDF-1.4 content"
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionResetError("transient")
            return self._response(pdf)

        with patch("urllib.request.urlopen", side_effect=flaky), patch("time.sleep"):
            assert _fetch_pdf_bytes("https://x/paper.pdf") == pdf
        assert calls["n"] == 2

    def test_title_mismatch_rejected(self):
        pdf = b"%PDF-1.4 content"
        with patch("urllib.request.urlopen", return_value=self._response(pdf)), \
             patch("wireless_taxonomy.analyze.dataset_extractor._pdf_matches_title", return_value=False):
            assert _fetch_pdf_bytes("https://x/paper.pdf", expected_title="Some Paper") is None


@patch("wireless_taxonomy.analyze.dataset_extractor._fetch_pdf_bytes", return_value=None)
@patch("wireless_taxonomy.analyze.dataset_extractor._fetch_crossref_bibtex", return_value=None)
@patch("wireless_taxonomy.analyze.dataset_extractor._check_url_live", return_value=None)
def test_no_pdf_no_abstract_skips_llm_entirely(mock_url, mock_bib, mock_fetch):
    """Title-only papers are never sent to the LLM — extraction from a title
    alone is pure hallucination and wasted spend."""
    router = _mock_router()
    extractor = DatasetExtractor(router=router, cache=None, conn=None)
    result = extractor.extract(
        paper_id=10, title="Mystery Paper", authors="Doe, J.",
        venue="ICC", year=2024, doi="", pdf_url=None, abstract="",
    )
    assert result.extraction_source == "skipped_no_text"
    assert result.datasets == []
    router.complete.assert_not_called()


@patch("wireless_taxonomy.analyze.dataset_extractor._fetch_pdf_bytes", return_value=None)
@patch("wireless_taxonomy.analyze.dataset_extractor._fetch_crossref_bibtex", return_value=None)
@patch("wireless_taxonomy.analyze.dataset_extractor._check_url_live", return_value=None)
def test_abstract_only_prompt_uses_strict_mode(mock_url, mock_bib, mock_fetch):
    """Abstract-only extraction must instruct the LLM to extract ONLY explicitly
    described datasets (anti-hallucination guard)."""
    router = _mock_router(response_json={"datasets": []})
    extractor = DatasetExtractor(router=router, cache=None, conn=None)
    result = extractor.extract(
        paper_id=11, title="Some 5G Paper", authors="Doe, J.",
        venue="ICC", year=2024, doi="", pdf_url=None,
        abstract="We propose a scheduling algorithm for 5G RAN slicing.",
    )
    assert result.extraction_source == "abstract"
    prompt = router.complete.call_args[0][0].prompt
    assert "STRICT ABSTRACT MODE" in prompt
    assert "EXPLICITLY" in prompt


class _DictCache:
    def __init__(self):
        self.d = {}
    def get_llm(self, key):
        return self.d.get(key)
    def set_llm(self, key, value):
        self.d[key] = value


def _router_with_identity(provider, model, response_json=None):
    router = _mock_router(response_json=response_json)
    entry = MagicMock()
    entry.provider = provider
    entry.model = model
    router.configured_providers = MagicMock(return_value=(entry,))
    router.select_provider = MagicMock(return_value=entry)
    return router


@patch("wireless_taxonomy.analyze.dataset_extractor._fetch_pdf_bytes", return_value=None)
@patch("wireless_taxonomy.analyze.dataset_extractor._fetch_crossref_bibtex", return_value=None)
@patch("wireless_taxonomy.analyze.dataset_extractor._check_url_live", return_value=None)
def test_extraction_cache_is_model_scoped(mock_url, mock_bib, mock_fetch):
    """Switching models must re-run extraction, not serve another model's output."""
    cache = _DictCache()
    kwargs = dict(paper_id=20, title="Model Scope Paper", authors="A",
                  venue="ICC", year=2024, doi="", pdf_url=None,
                  abstract="We collected the FooSet dataset of 5G traces in a lab.")

    r1 = _router_with_identity("google", "flash-1")
    DatasetExtractor(router=r1, cache=cache, conn=None).extract(**kwargs)
    assert r1.complete.call_count == 1

    # Same model chain: served from cache, no new call.
    r1b = _router_with_identity("google", "flash-1")
    DatasetExtractor(router=r1b, cache=cache, conn=None).extract(**kwargs)
    assert r1b.complete.call_count == 0

    # Different model chain: fresh extraction.
    r2 = _router_with_identity("openai", "gpt-x")
    DatasetExtractor(router=r2, cache=cache, conn=None).extract(**kwargs)
    assert r2.complete.call_count == 1


@patch("wireless_taxonomy.analyze.dataset_extractor._fetch_pdf_bytes", return_value=None)
@patch("wireless_taxonomy.analyze.dataset_extractor._fetch_crossref_bibtex", return_value=None)
@patch("wireless_taxonomy.analyze.dataset_extractor._check_url_live", return_value=None)
def test_legacy_v1_cache_entries_are_migrated_not_rebilled(mock_url, mock_bib, mock_fetch):
    """Pre-model-scoped (v1) cache entries are adopted under the new key."""
    import hashlib as h
    from wireless_taxonomy.analyze.dataset_extractor import _extraction_cache_key

    cache = _DictCache()
    abstract = "We collected the BarSet dataset of WiFi CSI in an office."
    content_hash = h.sha256(abstract.encode()).hexdigest()[:16]
    legacy_key = _extraction_cache_key(21, content_hash)  # v1 (no model)
    cache.set_llm(legacy_key, {"datasets": [], "source": "abstract"})

    router = _router_with_identity("google", "flash-1")
    result = DatasetExtractor(router=router, cache=cache, conn=None).extract(
        paper_id=21, title="Legacy Paper", authors="B", venue="IMC", year=2023,
        doi="", pdf_url=None, abstract=abstract,
    )
    assert router.complete.call_count == 0  # migrated, not re-billed
    v2_key = _extraction_cache_key(21, content_hash, "google/flash-1")
    assert cache.get_llm(v2_key) is not None


@patch("wireless_taxonomy.analyze.dataset_extractor._fetch_pdf_bytes", return_value=None)
@patch("wireless_taxonomy.analyze.dataset_extractor._fetch_crossref_bibtex", return_value=None)
@patch("wireless_taxonomy.analyze.dataset_extractor._check_url_live", return_value=None)
def test_refresh_skips_cache_read_but_writes_fresh_result(mock_url, mock_bib, mock_fetch):
    """refresh=True forces a fresh LLM call for one paper and overwrites the cache."""
    cache = _DictCache()
    kwargs = dict(paper_id=30, title="Refresh Me", authors="A",
                  venue="ICC", year=2024, doi="", pdf_url=None,
                  abstract="We collected the BazSet dataset of LTE traces in a lab.")

    r1 = _router_with_identity("google", "flash-1")
    DatasetExtractor(router=r1, cache=cache, conn=None).extract(**kwargs)
    assert r1.complete.call_count == 1

    # Cached: no call.
    r2 = _router_with_identity("google", "flash-1")
    DatasetExtractor(router=r2, cache=cache, conn=None).extract(**kwargs)
    assert r2.complete.call_count == 0

    # refresh=True: fresh call despite the cache.
    r3 = _router_with_identity("google", "flash-1")
    DatasetExtractor(router=r3, cache=cache, conn=None).extract(**kwargs, refresh=True)
    assert r3.complete.call_count == 1

    # And the refreshed result is served from cache afterwards.
    r4 = _router_with_identity("google", "flash-1")
    DatasetExtractor(router=r4, cache=cache, conn=None).extract(**kwargs)
    assert r4.complete.call_count == 0


def test_within_paper_dedup():
    """Near-duplicate datasets from same paper are merged."""
    from wireless_taxonomy.analyze.dataset_extractor import _parse_dataset_records

    raw = [
        {"name": "WiFi CSI Gesture Dataset", "relationship_type": "introduced",
         "modalities": ["CSI"], "osi_layers": ["L1"], "confidence": "high",
         "evidence_text": "We collected WiFi CSI.", "availability": None,
         "availability_notes": "", "availability_url": "",
         "collection_environment": "Physical Lab Testbed", "known_users": []},
        {"name": "WiFi CSI Gesture Data", "relationship_type": "introduced",
         "modalities": ["CSI"], "osi_layers": ["L1"], "confidence": "high",
         "evidence_text": "WiFi gesture data.", "availability": None,
         "availability_notes": "", "availability_url": "",
         "collection_environment": "Physical Lab Testbed", "known_users": []},
    ]
    records, dropped = _parse_dataset_records(raw)
    assert len(records) == 1
    # Longer name wins
    assert records[0].name == "WiFi CSI Gesture Dataset"
    assert len(dropped) == 1
    assert dropped[0].reason == "dedup_merged"


def test_different_relationships_not_deduped():
    """Same name but different relationship_type are NOT merged."""
    from wireless_taxonomy.analyze.dataset_extractor import _parse_dataset_records

    raw = [
        {"name": "CRAWDAD WiFi Dataset", "relationship_type": "introduced",
         "modalities": ["WiFi"], "osi_layers": ["L2"], "confidence": "high",
         "evidence_text": "We release CRAWDAD.", "availability": True,
         "availability_notes": "", "availability_url": "",
         "collection_environment": "Real World Deployment", "known_users": []},
        {"name": "CRAWDAD WiFi Dataset", "relationship_type": "reused",
         "modalities": ["WiFi"], "osi_layers": ["L2"], "confidence": "high",
         "evidence_text": "We use CRAWDAD.", "availability": True,
         "availability_notes": "", "availability_url": "",
         "collection_environment": "Real World Deployment", "known_users": []},
    ]
    records, dropped = _parse_dataset_records(raw)
    assert len(records) == 2
    assert len(dropped) == 0


def test_dropped_records_tracked():
    """Low confidence and garbage names are tracked in dropped list."""
    from wireless_taxonomy.analyze.dataset_extractor import _parse_dataset_records

    raw = [
        {"name": "5G Trace Dataset", "relationship_type": "reused",
         "modalities": ["5G"], "osi_layers": ["L3"], "confidence": "low",
         "evidence_text": "trace data", "availability": None,
         "availability_notes": "", "availability_url": "",
         "collection_environment": "Unknown", "known_users": []},
        {"name": "our data", "relationship_type": "reused",
         "modalities": [], "osi_layers": [], "confidence": "high",
         "evidence_text": "we use our data", "availability": None,
         "availability_notes": "", "availability_url": "",
         "collection_environment": "Unknown", "known_users": []},
        {"name": "Widar 3.0", "relationship_type": "reused",
         "modalities": ["WiFi CSI"], "osi_layers": ["L1"], "confidence": "high",
         "evidence_text": "We evaluate on Widar.", "availability": True,
         "availability_notes": "", "availability_url": "https://example.com",
         "collection_environment": "Physical Lab Testbed", "known_users": []},
    ]
    records, dropped = _parse_dataset_records(raw)
    assert len(records) == 1
    assert records[0].name == "Widar 3.0"
    assert len(dropped) == 2
    reasons = {d.reason for d in dropped}
    assert "low_confidence" in reasons
    assert "garbage_name" in reasons or "garbage_name_exact" in reasons


def test_evidence_grounding():
    """Evidence grounding check flags fabricated evidence."""
    from wireless_taxonomy.analyze.dataset_extractor import DatasetRecord, _ground_evidence

    source = (
        "We collected 5G NR traces across 12 routes in New York City over "
        "3 months using commercial smartphones connected to T-Mobile."
    )
    ds_grounded = DatasetRecord(
        name="5G NYC Traces", relationship_type="introduced",
        modalities=["5G NR"], osi_layers=["L1"], availability=True,
        availability_notes="", availability_url="", confidence="high",
        collection_environment="Real World Deployment", known_users=[],
        evidence_text="We collected 5G NR traces across 12 routes in New York City over 3 months.",
    )
    ds_fabricated = DatasetRecord(
        name="Indoor WiFi Localization Data", relationship_type="introduced",
        modalities=["WiFi"], osi_layers=["L1"], availability=None,
        availability_notes="", availability_url="", confidence="medium",
        collection_environment="Physical Lab Testbed", known_users=[],
        evidence_text="We deployed 50 Raspberry Pi access points across a university library for indoor localization.",
    )
    _ground_evidence([ds_grounded, ds_fabricated], source)
    assert ds_grounded.grounded is True
    assert ds_fabricated.grounded is False


def test_crossref_bibtex_is_cached():
    """CrossRef BibTeX lookups hit the network once, then serve from cache."""
    from unittest.mock import patch as _patch

    cache = _DictCache()
    ext = DatasetExtractor(router=_mock_router(), cache=cache, conn=None)
    with _patch(
        "wireless_taxonomy.analyze.dataset_extractor._fetch_crossref_bibtex",
        return_value="@inproceedings{x2024y, title={T}}",
    ) as fetch:
        assert ext._cached_crossref_bibtex("10.1234/abc") is not None
        assert ext._cached_crossref_bibtex("10.1234/abc") is not None
        assert fetch.call_count == 1  # second call served from cache

    # Failures are NOT cached (stay retryable)
    cache2 = _DictCache()
    ext2 = DatasetExtractor(router=_mock_router(), cache=cache2, conn=None)
    with _patch(
        "wireless_taxonomy.analyze.dataset_extractor._fetch_crossref_bibtex",
        return_value=None,
    ) as fetch2:
        assert ext2._cached_crossref_bibtex("10.9999/fail") is None
        assert ext2._cached_crossref_bibtex("10.9999/fail") is None
        assert fetch2.call_count == 2  # retried, not cached


def test_url_liveness_is_cached_with_ttl():
    """URL live checks are cached; stale entries re-check."""
    from unittest.mock import patch as _patch

    cache = _DictCache()
    ext = DatasetExtractor(router=_mock_router(), cache=cache, conn=None)
    with _patch(
        "wireless_taxonomy.analyze.dataset_extractor._check_url_live",
        return_value=True,
    ) as check:
        assert ext._cached_url_live("https://example.com/data") is True
        assert ext._cached_url_live("https://example.com/data") is True
        assert check.call_count == 1  # cached

    # Expired entry (8 days old) triggers a fresh check
    from datetime import datetime, timedelta, timezone
    key = "urllive:https://example.com/data"
    cache.d[key]["checked_at"] = (
        datetime.now(timezone.utc) - timedelta(days=8)
    ).isoformat()
    with _patch(
        "wireless_taxonomy.analyze.dataset_extractor._check_url_live",
        return_value=False,
    ) as check2:
        assert ext._cached_url_live("https://example.com/data") is False
        assert check2.call_count == 1
