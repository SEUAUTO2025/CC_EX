import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_script(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script), *args],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_write_environment_txt_script_runs_from_checkout(tmp_path):
    output = tmp_path / "env.txt"

    result = _run_script(
        "write_environment_txt.py", "--output", str(output)
    )

    assert result.returncode == 0, result.stderr
    assert "planning record only" in output.read_text(encoding="utf-8")


def test_import_reported_baselines_script_runs_from_checkout(tmp_path):
    output = tmp_path / "reported.csv"

    result = _run_script(
        "import_reported_baselines.py", "--output", str(output)
    )

    assert result.returncode == 0, result.stderr
    assert "reported_not_rerun" in output.read_text(encoding="utf-8")
