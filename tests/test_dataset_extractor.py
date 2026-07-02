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
