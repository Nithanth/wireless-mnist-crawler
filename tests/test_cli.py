import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from wireless_taxonomy.cli import _parse_venue_years, app
from wireless_taxonomy.commands.export import _write_pdf_only_datasets

runner = CliRunner()


def _write(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_parse_venue_years_ok_and_bad() -> None:
    import typer

    assert _parse_venue_years(["IMC:2025", " NSDI : 2024 "]) == [("IMC", "2025"), ("NSDI", "2024")]
    for bad in (["IMC2025"], ["IMC:"], [":2025"]):
        try:
            _parse_venue_years(bad)
        except typer.BadParameter:
            continue
        raise AssertionError(f"{bad} should have raised")


def test_eval_missing_file_is_clean_error(tmp_path: Path) -> None:
    gold = _write(tmp_path / "gold.csv", "Paper Title,Conference,Year\nX,SIGCOMM,2024\n")
    result = runner.invoke(app, ["advanced", "eval", "--classified", str(tmp_path / "nope.csv"), "--gold", gold])
    assert result.exit_code != 0
    assert "file not found" in result.output


def test_classify_bad_source_is_clean_error() -> None:
    result = runner.invoke(app, ["advanced", "classify", "--venue", "IMC", "--years", "2024", "--source", "ftp", "--no-llm"])
    assert result.exit_code != 0
    assert "--source must be one of" in result.output


def test_eval_exclude_and_min_gold_pull_from_headline(tmp_path: Path) -> None:
    classified = _write(
        tmp_path / "pred.csv",
        "title,doi,venue,year,label\n"
        "Wireless A,10.1/a,SIGCOMM,2024,yes\n"
        "Wireless B,10.1/b,SIGCOMM,2024,yes\n"
        "Curated IMC,10.1/c,IMC,2025,yes\n"
        "Uncurated IMC,10.1/d,IMC,2025,yes\n",
    )
    gold = _write(
        tmp_path / "gold.csv",
        "Paper Title,Conference,Year,DOI\n"
        "Wireless A,SIGCOMM,2024,10.1/a\n"
        "Wireless B,SIGCOMM,2024,10.1/b\n"
        "Curated IMC,IMC,2025,10.1/c\n",
    )
    result = runner.invoke(app, ["advanced", "eval", "--classified", classified, "--gold", gold, "--exclude", "IMC:2025"])
    assert result.exit_code == 0, result.output
    assert "under-curated / excluded" in result.output
    assert "IMC 2025" in result.output


def test_pdf_only_export_maps_raw_keys_to_disambiguated_final_keys(tmp_path: Path) -> None:
    _write(
        tmp_path / "consolidated_papers.csv",
        "Paper Title,Bibtex Citation Key\nExample Wireless Paper,smith2024examplea\n",
    )
    _write(
        tmp_path / "consolidated_datasets.csv",
        "Canonical Name,Bibtex Citation Keys\nExample Dataset,smith2024examplea\n",
    )
    (tmp_path / "venue_2024_raw.json").write_text(
        json.dumps({
            "runs": [{
                "papers": [{
                    "title": "Example Wireless Paper",
                    "bibtex_key": "smith2024example",
                    "extraction_source": "pdf",
                    "datasets": [{"name": "Example Dataset"}],
                }],
            }],
        }),
        encoding="utf-8",
    )

    _write_pdf_only_datasets(tmp_path)

    with (tmp_path / "consolidated_datasets_pdf_only.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [row["Canonical Name"] for row in rows] == ["Example Dataset"]


def test_cli_help_renders_without_typer_click_compat_error() -> None:
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "wireless_taxonomy.cli", "--help"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    # Primary user-facing commands at the top level.
    assert "init" in result.stdout
    assert "add" in result.stdout
    assert "export" in result.stdout
    assert "status" in result.stdout
    assert "rollback" in result.stdout
    assert "advanced" in result.stdout
    # Pipeline stages should NOT be top-level commands anymore.
    # (Note: "classify" appears in the `add` description, so check for command entries)
    assert "fetch-coverage" not in result.stdout
    assert "extract-datasets" not in result.stdout
    assert "llm-config" not in result.stdout


def test_primary_and_advanced_commands_help() -> None:
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    # Primary commands at top level.
    for cmd in ("init", "add", "export", "status", "use", "rollback"):
        result = subprocess.run(
            [sys.executable, "-m", "wireless_taxonomy.cli", cmd, "--help"],
            cwd=root, env=env, text=True, capture_output=True, check=False,
        )
        assert result.returncode == 0, f"{cmd} --help failed: {result.stderr}"

    # Advanced commands live under the advanced subgroup.
    for cmd in ("classify", "eval", "eval-db", "fetch-coverage", "extract-datasets",
                "merge-results", "reconcile-datasets", "report",
                "llm-config", "prune", "purge-cache", "cache-set-pdf",
                "fill-availability"):
        result = subprocess.run(
            [sys.executable, "-m", "wireless_taxonomy.cli", "advanced", cmd, "--help"],
            cwd=root, env=env, text=True, capture_output=True, check=False,
        )
        assert result.returncode == 0, f"advanced {cmd} --help failed: {result.stderr}"
