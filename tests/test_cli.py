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
