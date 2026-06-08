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
