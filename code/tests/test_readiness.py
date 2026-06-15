import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from dar_td3bc.utils.readiness import (
    assess_result_readiness,
    render_next_operations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_assess_result_readiness_reports_missing_data_first(tmp_path):
    items = assess_result_readiness(tmp_path)

    assert items[0].key == "data"
    assert items[0].status == "missing"
    assert "acquire_data.py" in items[0].command


def test_assess_result_readiness_advances_to_metadata_when_data_exists(tmp_path):
    raw = tmp_path / "data" / "raw" / "neorl2"
    raw.mkdir(parents=True)
    np.savez_compressed(raw / "pipeline_train.npz", observations=np.zeros((1, 52)))
    np.savez_compressed(raw / "pipeline_val.npz", observations=np.zeros((1, 52)))

    items = assess_result_readiness(tmp_path)

    assert items[0].key == "data"
    assert items[0].status == "ok"
    assert items[1].key == "metadata"
    assert items[1].status == "missing"
    assert "prepare_pipeline_data.py" in items[1].command


def test_render_next_operations_is_markdown_with_blocker_status(tmp_path):
    text = render_next_operations(tmp_path)

    assert text.startswith("# DAR-TD3BC Next Operations")
    assert "Cannot produce final paper results yet" in text
    assert "python scripts/acquire_data.py" in text


def test_check_next_operation_script_runs(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "check_next_operation.py"),
            "--root",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "DAR-TD3BC Next Operations" in result.stdout


def test_check_next_operation_script_can_write_output_file(tmp_path):
    output = tmp_path / "NEXT_OPERATIONS.md"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "check_next_operation.py"),
            "--root",
            str(tmp_path),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert "DAR-TD3BC Next Operations" in output.read_text(encoding="utf-8")
