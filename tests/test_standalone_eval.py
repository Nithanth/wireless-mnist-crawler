import json

from wireless_taxonomy.eval.standalone import eval_files, eval_files_to_outputs


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_eval_files_scores_tp_fp_fn_per_venue_year(tmp_path) -> None:
    classified = _write(
        tmp_path / "pred.csv",
        "title,authors,doi,venue,year,wireless_label,confidence\n"
        "Wireless Sensing with CSI,A,10.1/aaa,SIGCOMM,2024,yes,0.9\n"
        "A Non-wireless Compiler Paper,B,10.1/bbb,SIGCOMM,2024,yes,0.8\n",  # FP (not in gold)
    )
    gold = _write(
        tmp_path / "gold.csv",
        "Paper Title,Conference,Year,DOI\n"
        "Wireless Sensing with CSI,SIGCOMM,2024,10.1/aaa\n"  # TP via DOI
        "A Curated Wireless Paper Not Predicted,SIGCOMM,2024,10.1/ccc\n",  # FN
    )
    report = eval_files([classified], [gold])
    overall = report["overall"]
    assert (overall["tp"], overall["fp"], overall["fn"]) == (1, 1, 1)
    assert overall["precision"] == 0.5
    assert overall["recall"] == 0.5
    assert len(report["instances"]) == 1
    inst = report["instances"][0]
    assert inst["venue"] == "SIGCOMM" and str(inst["year"]) == "2024"


def test_eval_files_matches_by_title_when_doi_absent(tmp_path) -> None:
    classified = _write(
        tmp_path / "pred.csv",
        "title,doi,venue,year\n"
        "Flow Scheduling With Imprecise Knowledge,,NSDI,2024\n",
    )
    gold = _write(
        tmp_path / "gold.csv",
        "Paper Title,Conference,Year\n"
        "Flow scheduling with imprecise knowledge,NSDI,2024\n",  # title-only match
    )
    report = eval_files([classified], [gold])
    assert report["overall"]["tp"] == 1
    assert report["overall"]["fp"] == 0
    assert report["overall"]["fn"] == 0


def test_eval_files_ignores_gold_venue_years_not_classified(tmp_path) -> None:
    classified = _write(
        tmp_path / "pred.csv",
        "title,doi,venue,year\nSome Wireless Paper,10.1/x,SIGCOMM,2024\n",
    )
    gold = _write(
        tmp_path / "gold.csv",
        "Paper Title,Conference,Year,DOI\n"
        "Some Wireless Paper,SIGCOMM,2024,10.1/x\n"
        "Another Paper,IMC,2023,10.1/y\n",  # IMC 2023 not classified -> ignored, not FN
    )
    report = eval_files([classified], [gold])
    assert report["overall"]["fn"] == 0
    assert report["ignored_gold_instances"] == [
        {"venue": "IMC", "year": "2023", "gold_papers": 1}
    ]


def _two_venue_year_files(tmp_path):
    """SIGCOMM 2024 well-curated (2 gold) + IMC 2025 thin (1 gold, 1 extra FP)."""
    classified = _write(
        tmp_path / "pred.csv",
        "title,doi,venue,year,label\n"
        "Wireless A,10.1/a,SIGCOMM,2024,yes\n"
        "Wireless B,10.1/b,SIGCOMM,2024,yes\n"
        "Curated IMC Paper,10.1/c,IMC,2025,yes\n"
        "Uncurated Wireless Paper,10.1/d,IMC,2025,yes\n",  # IMC FP (not in gold)
    )
    gold = _write(
        tmp_path / "gold.csv",
        "Paper Title,Conference,Year,DOI\n"
        "Wireless A,SIGCOMM,2024,10.1/a\n"
        "Wireless B,SIGCOMM,2024,10.1/b\n"
        "Curated IMC Paper,IMC,2025,10.1/c\n",
    )
    return classified, gold


def test_eval_files_exclude_pulls_venue_year_from_headline(tmp_path) -> None:
    classified, gold = _two_venue_year_files(tmp_path)
    base = eval_files([classified], [gold])
    assert base["overall"]["fp"] == 1  # IMC FP drags the headline

    report = eval_files([classified], [gold], exclude=[("IMC", "2025")])
    # IMC 2025 removed from headline -> only the clean SIGCOMM row remains.
    assert report["overall"]["fp"] == 0
    assert report["overall"]["tp"] == 2
    assert [i["venue"] for i in report["instances"]] == ["SIGCOMM"]
    under = report["under_curated_instances"]
    assert len(under) == 1
    assert under[0]["venue"] == "IMC" and under[0]["reason"] == "excluded"
    assert under[0]["gold_papers"] == 1 and under[0]["fp"] == 1
    # Discrepancy detail is still kept for the excluded venue-year.
    assert any(m["venue"] == "IMC" for m in report["mismatches"])


def test_eval_files_min_gold_pulls_thin_venue_years(tmp_path) -> None:
    classified, gold = _two_venue_year_files(tmp_path)
    report = eval_files([classified], [gold], min_gold=2)
    assert [i["venue"] for i in report["instances"]] == ["SIGCOMM"]
    under = report["under_curated_instances"]
    assert len(under) == 1
    assert under[0]["venue"] == "IMC"
    assert "under-curated" in under[0]["reason"]


def test_eval_files_min_gold_zero_is_noop(tmp_path) -> None:
    classified, gold = _two_venue_year_files(tmp_path)
    report = eval_files([classified], [gold], min_gold=0)
    assert report["under_curated_instances"] == []
    assert len(report["instances"]) == 2


def test_eval_files_to_outputs_writes_json_and_md(tmp_path) -> None:
    classified = _write(
        tmp_path / "pred.csv",
        "title,doi,venue,year\nWireless X,10.1/x,SIGCOMM,2024\n",
    )
    gold = _write(
        tmp_path / "gold.csv",
        "Paper Title,Conference,Year,DOI\nWireless X,SIGCOMM,2024,10.1/x\n",
    )
    json_out = tmp_path / "r.json"
    md_out = tmp_path / "r.md"
    eval_files_to_outputs([classified], [gold], json_out=str(json_out), md_out=str(md_out))
    data = json.loads(json_out.read_text())
    assert data["overall"]["tp"] == 1
    md = md_out.read_text()
    assert "Wireless classification vs. manual sheet" in md
    assert "Overall" in md
