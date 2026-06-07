from pathlib import Path

import pytest

from wireless_taxonomy.config import load_settings
from wireless_taxonomy.eval import overlap
from wireless_taxonomy.ingest.gold import GoldSheetReader
from wireless_taxonomy.pipeline import Pipeline
from wireless_taxonomy.textnorm import normalize_doi, normalize_title

FIXTURES = Path(__file__).parent / "fixtures"


def test_normalize_title_and_doi() -> None:
    assert normalize_title("  Déjà-Vu:  RF Sensing! ") == "deja vu rf sensing"
    assert normalize_title("A B") == normalize_title("a  b")
    assert normalize_doi("https://doi.org/10.1145/ABC.123") == "10.1145/abc.123"
    assert normalize_doi("doi:10.1/X/") == "10.1/x"


def test_match_prefers_doi_then_title_then_fuzzy() -> None:
    a = [
        overlap.PaperRef.build("a1", "Wireless Sensing at the Edge", "10.1/AA"),
        overlap.PaperRef.build("a2", "Beamforming for 6G Networks"),
        overlap.PaperRef.build("a3", "A Datacenter Congestion Study extra words"),
    ]
    b = [
        overlap.PaperRef.build("b1", "completely different title", "10.1/aa"),  # DOI match to a1
        overlap.PaperRef.build("b2", "beamforming for 6g networks"),  # title match to a2
        overlap.PaperRef.build("b3", "A Datacenter Congestion Study"),  # fuzzy to a3
    ]
    result = overlap.match(a, b, fuzzy_threshold=0.8)
    matched_keys = {(x.key, y.key) for x, y in result.matched}
    assert ("a1", "b1") in matched_keys
    assert ("a2", "b2") in matched_keys
    assert ("a3", "b3") in matched_keys
    assert not result.unmatched_a and not result.unmatched_b


def test_match_fuzzy_disabled_at_threshold_one() -> None:
    a = [overlap.PaperRef.build("a1", "A Datacenter Congestion Study extra")]
    b = [overlap.PaperRef.build("b1", "A Datacenter Congestion Study")]
    result = overlap.match(a, b, fuzzy_threshold=1.0)
    assert result.unmatched_a and result.unmatched_b


def test_metrics_and_aggregate() -> None:
    m = overlap.Metrics(tp=2, fp=1, fn=1)
    assert m.jaccard == 0.5
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)

    rows = [
        {"venue": "SIGCOMM", "year": 2024, "tp": 1, "fp": 0, "fn": 1, "fn_missed": 1, "fn_missing_from_universe": 0},
        {"venue": "SIGCOMM", "year": 2025, "tp": 2, "fp": 1, "fn": 0, "fn_missed": 0, "fn_missing_from_universe": 0},
    ]
    agg = overlap.aggregate(rows)
    assert len(agg["per_conference_year"]) == 2
    assert len(agg["per_conference"]) == 1
    conf = agg["per_conference"][0]
    assert conf["venue"] == "SIGCOMM"
    assert conf["tp"] == 3 and conf["fp"] == 1 and conf["fn"] == 1
    assert agg["overall"]["jaccard"] == round(3 / 5, 4)


def test_gold_reader_flexible_columns_and_wireless_filter(tmp_path: Path) -> None:
    sheet = tmp_path / "gold.csv"
    sheet.write_text(
        "Paper Title,Conference,Year,Wireless\n"
        "Wireless Paper,SIGCOMM,2025,yes\n"
        "Wired Paper,SIGCOMM,2025,no\n",
        encoding="utf-8",
    )
    all_rows = GoldSheetReader(str(sheet)).read()
    assert len(all_rows) == 2

    only_wireless = GoldSheetReader(str(sheet), wireless_only=True).read()
    assert [r.title for r in only_wireless] == ["Wireless Paper"]


def test_gold_reader_uses_defaults_when_columns_missing(tmp_path: Path) -> None:
    sheet = tmp_path / "gold.csv"
    sheet.write_text("title\nSome Wireless Paper\n", encoding="utf-8")
    records = GoldSheetReader(str(sheet), default_venue="MobiCom", default_year=2024).read()
    assert records[0].venue == "MobiCom" and records[0].year == 2024


def test_end_to_end_keyword_candidates_and_jaccard(tmp_path: Path) -> None:
    db = tmp_path / "taxonomy.sqlite"
    gold = tmp_path / "gold.csv"
    gold.write_text(
        "title,conference,year\n"
        "Example Wireless Dataset Paper,SIGCOMM,2025\n"
        "Some Paper Not In Conference,SIGCOMM,2025\n",
        encoding="utf-8",
    )
    pipeline = Pipeline(load_settings(db))
    try:
        run_id = pipeline.ingest("SIGCOMM", 2025, "url", str(FIXTURES / "sigcomm_2025_papers_info.html"))
        pipeline.classify_candidates(run_id, use_llm=False)
        pipeline.import_gold(str(gold))

        high = pipeline.evaluate_overlap(classifier="keyword", pass_mode="high")
        low = pipeline.evaluate_overlap(classifier="keyword", pass_mode="low")
    finally:
        pipeline.close()

    high_instance = high["instances"][0]
    assert high_instance["tp"] == 1 and high_instance["fp"] == 0 and high_instance["fn"] == 1
    assert high_instance["jaccard"] == 0.5
    # The missing gold paper is not in the ingested universe -> coverage gap, not a miss.
    assert high_instance["fn_missing_from_universe"] == 1 and high_instance["fn_missed"] == 0

    low_instance = low["instances"][0]
    # Low pass also flags the "maybe" datacenter paper -> one false positive.
    assert low_instance["tp"] == 1 and low_instance["fp"] == 1 and low_instance["fn"] == 1
    assert low_instance["jaccard"] == round(1 / 3, 4)
