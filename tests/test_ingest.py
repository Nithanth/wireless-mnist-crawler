from pathlib import Path

from wireless_taxonomy.ingest.base import validate_paper_seeds
from wireless_taxonomy.ingest.url import UrlIngestAdapter, paper_seeds_from_llm_payload
from wireless_taxonomy.models import PaperSeed


FIXTURES = Path(__file__).parent / "fixtures"


def test_sigcomm_2024_fixture_ingests_paper_seeds() -> None:
    adapter = UrlIngestAdapter("SIGCOMM", 2024, str(FIXTURES / "sigcomm_2024_accepted.html"))
    seeds = adapter.fetch()
    assert len(seeds) == 2
    assert seeds[0].title.startswith("Integrated Two-way Radar")
    assert "Ryu Okubo" in seeds[0].authors
    assert seeds[0].doi == "10.1145/3651890.3672226"


def test_sigcomm_2025_fixture_preserves_links_and_abstracts() -> None:
    adapter = UrlIngestAdapter("SIGCOMM", 2025, str(FIXTURES / "sigcomm_2025_papers_info.html"))
    seeds = adapter.fetch()
    assert len(seeds) == 2
    assert seeds[0].pdf_url == "https://example.org/example-wireless.pdf"
    assert "Open5G Measurements" in (seeds[0].abstract or "")


def test_fixture_path_resolves_from_src_working_directory(monkeypatch) -> None:
    monkeypatch.chdir(Path(__file__).parents[1] / "src")
    adapter = UrlIngestAdapter("SIGCOMM", 2025, "tests/fixtures/sigcomm_2025_papers_info.html")
    seeds = adapter.fetch()
    assert len(seeds) == 2


def test_program_style_page_extracts_titles_authors_abstracts_and_dois() -> None:
    adapter = UrlIngestAdapter("SIGCOMM", 2025, str(FIXTURES / "sigcomm_2025_program_style.html"))
    seeds = adapter.fetch()
    assert len(seeds) == 2
    assert seeds[0].session == "NetAI"
    assert seeds[0].doi == "10.1145/3718958.3750468"
    assert seeds[0].paper_url == "https://dl.acm.org/doi/10.1145/3718958.3750468"
    assert "Chenchen Shou" in seeds[0].authors[0]
    assert "transceiver-centric" in (seeds[0].abstract or "")


def test_validation_detects_duplicates_missing_authors_and_low_confidence() -> None:
    seeds = [
        PaperSeed("A Paper", [], "SIGCOMM", 2025, "fixture", source_confidence=0.40),
        PaperSeed("A Paper", ["Ada"], "SIGCOMM", 2025, "fixture", source_confidence=0.95),
    ]
    items = validate_paper_seeds(seeds)
    reasons = {item.review_reason for item in items}
    assert "Missing authors" in reasons
    assert "Source extraction confidence below threshold" in reasons
    assert "Duplicate paper title detected" in reasons


def test_llm_payload_is_normalized_to_paper_seeds() -> None:
    payload = {
        "papers": [
            {
                "title": "A Heterogeneous Page Paper",
                "authors": "Ada Lovelace; Grace Hopper",
                "abstract": "A full abstract.",
                "doi": "10.1145/example",
                "paper_url": "https://dl.acm.org/doi/10.1145/example",
                "confidence": 0.92,
                "evidence_text": "A Heterogeneous Page Paper Ada Lovelace...",
            }
        ]
    }
    seeds = paper_seeds_from_llm_payload(payload, "SIGCOMM", 2025, "https://example.org", "generic_llm")
    assert len(seeds) == 1
    assert seeds[0].title == "A Heterogeneous Page Paper"
    assert seeds[0].authors == ["Ada Lovelace; Grace Hopper"]
    assert seeds[0].source_confidence == 0.92


def test_distinct_venue_years_dedupes_and_sorts(tmp_path) -> None:
    from wireless_taxonomy.ingest.gold import distinct_venue_years

    sheet = tmp_path / "gold.csv"
    sheet.write_text(
        "Paper Title,Conference,Year\n"
        "A,SIGCOMM,2024\n"
        "B,SIGCOMM,2024\n"
        "C,NSDI,2023\n"
        "D,IEEE Trans. Wireless Comm.,2024\n",
        encoding="utf-8",
    )
    assert distinct_venue_years([str(sheet)]) == [
        ("IEEE Trans. Wireless Comm.", 2024),
        ("NSDI", 2023),
        ("SIGCOMM", 2024),
    ]


def test_distinct_venue_years_unions_multiple_sheets(tmp_path) -> None:
    from wireless_taxonomy.ingest.gold import distinct_venue_years

    a = tmp_path / "a.csv"
    a.write_text("Paper Title,Conference,Year\nA,IMC,2023\n", encoding="utf-8")
    b = tmp_path / "b.csv"
    b.write_text("Paper Title,Conference,Year\nB,IMC,2023\nC,NSDI,2024\n", encoding="utf-8")
    assert distinct_venue_years([str(a), str(b)]) == [("IMC", 2023), ("NSDI", 2024)]
