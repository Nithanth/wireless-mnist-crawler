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


def test_aggregate_scope_to_universe_drops_workshop_fn() -> None:
    # One gold miss is a workshop paper (missing_from_universe); scoping drops it from FN.
    rows = [
        {"venue": "SIGCOMM", "year": 2024, "tp": 2, "fp": 1, "fn": 2, "fn_missed": 1, "fn_missing_from_universe": 1},
    ]
    full = overlap.aggregate(rows)["overall"]
    scoped = overlap.aggregate(rows, scope_to_universe=True)["overall"]
    # Unscoped: FN=2 -> jaccard 2/5; recall 2/4.
    assert full["fn"] == 2 and full["jaccard"] == round(2 / 5, 4) and full["recall"] == 0.5
    # Scoped: workshop FN dropped -> FN=1 -> jaccard 2/4; recall 2/3. Precision unchanged.
    assert scoped["fn"] == 1 and scoped["jaccard"] == 0.5
    assert scoped["recall"] == round(2 / 3, 4) and scoped["precision"] == full["precision"]
    assert scoped["fn_missing_from_universe"] == 1 and scoped["scoped_to_universe"] is True


def test_to_markdown_renders_tables_and_scope() -> None:
    report = {
        "classifier": "llm",
        "pass_mode": "high",
        "fuzzy_threshold": 0.92,
        "scope_to_universe": True,
        "instances": [
            {"venue": "SIGCOMM", "year": 2024, "jaccard": 0.5, "precision": 0.6, "recall": 0.7,
             "f1": 0.65, "tp": 2, "fp": 1, "fn": 1, "fn_missed": 1, "fn_missing_from_universe": 1},
        ],
        "per_conference": [
            {"venue": "SIGCOMM", "jaccard": 0.5, "precision": 0.6, "recall": 0.7,
             "f1": 0.65, "tp": 2, "fp": 1, "fn": 1, "fn_missed": 1, "fn_missing_from_universe": 1},
        ],
        "overall": {"jaccard": 0.5, "precision": 0.6, "recall": 0.7, "f1": 0.65,
                    "tp": 2, "fp": 1, "fn": 1, "fn_missing_from_universe": 1},
        "mismatches": [
            {"venue": "SIGCOMM", "year": 2024, "false_positives": ["Extra Paper"],
             "false_negatives_classifier_miss": ["Missed Paper"],
             "false_negatives_missing_from_universe": ["Workshop Paper"]},
        ],
    }
    md = overlap.to_markdown(report)
    assert "# Wireless classification vs. manual sheet" in md
    assert "workshop papers dropped" in md
    assert "| SIGCOMM | 2024 |" in md
    assert "Workshop Paper" in md and "Extra Paper" in md


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


class _NoEnricher:
    def fetch(self, title, doi=None, source_url=None):  # noqa: ARG002
        return None


class _NoResolver:
    def resolve(self, title):  # noqa: ARG002
        return None


def _write_classified_csv(result: dict, path: Path) -> None:
    import csv as _csv

    fields = ["title", "authors", "doi", "venue", "year", "label", "confidence", "used_abstract", "has_abstract"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["papers"])


def test_end_to_end_classify_full_set_and_file_eval(tmp_path: Path, monkeypatch) -> None:
    # Keep the pipeline offline: the fixture already carries abstracts + DOIs,
    # but stub the network enricher/resolver so nothing reaches out.
    monkeypatch.setattr("wireless_taxonomy.analyze.abstracts.AbstractEnricher", lambda: _NoEnricher())
    monkeypatch.setattr("wireless_taxonomy.analyze.abstracts.DoiResolver", lambda: _NoResolver())

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
        result = pipeline.classify_conference(
            "SIGCOMM", 2025, use_llm=False, source_type="url",
            source_value=str(FIXTURES / "sigcomm_2025_papers_info.html"),
        )
    finally:
        pipeline.close()

    # classify_conference returns the FULL labelled set (every paper), not just wireless.
    assert result["total_papers"] == 2
    assert result["counts"]["yes"] == 1  # the RF/SINR paper
    assert result["counts"]["maybe"] == 1  # the wired datacenter paper
    assert result["counts"]["no"] == 0

    classified = tmp_path / "classified.csv"
    _write_classified_csv(result, classified)

    from wireless_taxonomy.eval.standalone import eval_files

    # High pass (yes only): TP=1 (wireless paper), FP=0, FN=1 (the off-proceedings gold paper).
    high = eval_files([str(classified)], [str(gold)], pass_mode="high")
    hi = high["instances"][0]
    assert hi["tp"] == 1 and hi["fp"] == 0 and hi["fn"] == 1
    assert hi["jaccard"] == 0.5
    # The missing gold paper is absent from the classified universe -> coverage gap.
    assert hi["fn_missing_from_universe"] == 1 and hi["fn_missed"] == 0

    # Dropping workshops removes that off-proceedings gold paper from the denominator.
    high_drop = eval_files([str(classified)], [str(gold)], pass_mode="high", drop_workshops=True)
    assert high_drop["overall"]["fn"] == 0 and high_drop["overall"]["jaccard"] == 1.0

    # Low pass also flags the "maybe" datacenter paper -> one false positive.
    low = eval_files([str(classified)], [str(gold)], pass_mode="low")
    lo = low["instances"][0]
    assert lo["tp"] == 1 and lo["fp"] == 1 and lo["fn"] == 1
    assert lo["jaccard"] == round(1 / 3, 4)


def test_eval_drop_workshops_requires_label_column(tmp_path: Path) -> None:
    from wireless_taxonomy.eval.standalone import eval_files

    classified = tmp_path / "classified.csv"
    classified.write_text("title,venue,year\nA Wireless Paper,SIGCOMM,2025\n", encoding="utf-8")
    gold = tmp_path / "gold.csv"
    gold.write_text("title,conference,year\nA Wireless Paper,SIGCOMM,2025\n", encoding="utf-8")

    # No label column -> no universe -> drop_workshops can't be honoured.
    with pytest.raises(ValueError):
        eval_files([str(classified)], [str(gold)], drop_workshops=True)

    # Without dropping, a label-less CSV still scores (every row is a predicted positive).
    report = eval_files([str(classified)], [str(gold)])
    assert report["overall"]["tp"] == 1
