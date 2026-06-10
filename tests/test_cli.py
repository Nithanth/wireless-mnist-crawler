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
    assert "discover-full-text" in result.stdout
    assert "add-pdfs" in result.stdout
    assert "fetch-acm-browser" in result.stdout
    assert "classify-paper-list" in result.stdout
    assert "reflect-paper-analysis" in result.stdout


def test_classify_paper_list_prints_wireless_papers(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "wireless_taxonomy.cli",
            "classify-paper-list",
            "--venue",
            "SIGCOMM",
            "--year",
            "2025",
            "--url",
            "tests/fixtures/sigcomm_2025_papers_info.html",
            "--out",
            str(tmp_path / "classifications.json"),
            "--db",
            str(tmp_path / "taxonomy.sqlite"),
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Wireless Papers From Title/Abstract" in result.stdout
    assert "Captured 1 wireless paper(s) out of 2" in result.stdout
    assert "Example Wireless Dataset Paper" in result.stdout
    assert "Example Datacenter Congestion Paper" not in result.stdout


def test_classify_paper_list_can_suppress_wireless_printout(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "wireless_taxonomy.cli",
            "classify-paper-list",
            "--venue",
            "SIGCOMM",
            "--year",
            "2025",
            "--url",
            "tests/fixtures/sigcomm_2025_papers_info.html",
            "--out",
            str(tmp_path / "classifications.json"),
            "--db",
            str(tmp_path / "taxonomy.sqlite"),
            "--no-show-wireless",
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Paper-list classification completed." in result.stdout
    assert "Wireless Papers From Title/Abstract" not in result.stdout
