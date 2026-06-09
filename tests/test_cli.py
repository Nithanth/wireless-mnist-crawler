import os
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from wireless_taxonomy.cli import _parse_venue_years, app

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
    result = runner.invoke(app, ["eval", "--classified", str(tmp_path / "nope.csv"), "--gold", gold])
    assert result.exit_code != 0
    assert "file not found" in result.output


def test_classify_bad_source_is_clean_error() -> None:
    result = runner.invoke(app, ["classify", "--venue", "IMC", "--years", "2024", "--source", "ftp", "--no-llm"])
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
    result = runner.invoke(app, ["eval", "--classified", classified, "--gold", gold, "--exclude", "IMC:2025"])
    assert result.exit_code == 0, result.output
    assert "under-curated / excluded" in result.output
    assert "IMC 2025" in result.output


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
    # The pruned surface is exactly three commands.
    assert "classify" in result.stdout
    assert "eval" in result.stdout
    assert "llm-config" in result.stdout
    # Old multi-stage commands were removed in the prune.
    for gone in (
        "ingest",
        "enrich-abstracts",
        "classify-candidates",
        "classify-conference",
        "classify-wireless",
        "import-gold",
        "gold-venues",
        "eval-overlap",
        "eval-files",
        "paper-set",
        "diff-sets",
        "status",
    ):
        assert gone not in result.stdout, f"{gone} should have been pruned"


def test_classify_and_eval_in_help() -> None:
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    for cmd in ("classify", "eval", "llm-config"):
        result = subprocess.run(
            [sys.executable, "-m", "wireless_taxonomy.cli", cmd, "--help"],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, f"{cmd} --help failed: {result.stderr}"
