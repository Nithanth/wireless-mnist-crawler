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
    assert "import-gold" in result.stdout
    assert "eval-overlap" in result.stdout
    assert "paper-set" in result.stdout
    assert "diff-sets" in result.stdout
    # Half B commands were removed in the slim-down.
    assert "fetch-acm-browser" not in result.stdout
    assert "extract-datasets" not in result.stdout
