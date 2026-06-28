"""Tests for DatasetExtractor, focusing on the PDF text fallback path."""

from unittest.mock import MagicMock, patch

import pytest

from wireless_taxonomy.analyze.dataset_extractor import DatasetExtractor


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
