import os
import subprocess
import sys
from pathlib import Path


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
    assert "ingest" in result.stdout
    assert "enrich-abstracts" in result.stdout
    assert "classify-candidates" in result.stdout
    assert "classify-conference" in result.stdout
    assert "import-gold" in result.stdout
    assert "gold-venues" in result.stdout
    assert "eval-overlap" in result.stdout
    assert "eval-files" in result.stdout
    assert "paper-set" in result.stdout
    assert "diff-sets" in result.stdout
    # Half B commands were removed in the slim-down.
    assert "fetch-acm-browser" not in result.stdout
    assert "extract-datasets" not in result.stdout


def test_gold_venues_lists_ingestable_and_skips_unmapped(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    sheet = tmp_path / "gold.csv"
    sheet.write_text(
        "Paper Title,Conference,Year\n"
        "A,SIGCOMM,2024\n"
        "B,NSDI,2023\n"
        "C,IEEE Trans. Wireless Comm.,2024\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "wireless_taxonomy.cli", "gold-venues", "--path", str(sheet)],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert lines == ["NSDI:2023", "SIGCOMM:2024"]
    # The journal has no DBLP stream mapping, so it's skipped (reported on stderr).
    assert "IEEE Trans. Wireless Comm.:2024" in result.stderr

    # --all keeps the unmapped venue too.
    result_all = subprocess.run(
        [sys.executable, "-m", "wireless_taxonomy.cli", "gold-venues", "--path", str(sheet), "--all"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert "IEEE Trans. Wireless Comm.:2024" in result_all.stdout
