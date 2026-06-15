import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from dar_td3bc.data.acquisition import missing_pipeline_data_request
from dar_td3bc.data.acquisition import save_dataset_npz


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_missing_pipeline_data_request_lists_required_fields():
    text = missing_pipeline_data_request()

    assert "observations" in text
    assert "actions" in text
    assert "rewards" in text
    assert "next_observations" in text
    assert "terminals/dones" in text


def test_acquire_data_reports_clear_request_when_dependency_missing(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "acquire_data.py"),
            "--task",
            "Pipeline",
            "--output",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "NeoRL-2 Pipeline" in result.stdout
    assert "observations" in result.stdout


def test_save_dataset_npz_maps_official_neorl2_field_names(tmp_path):
    output = tmp_path / "pipeline_train.npz"
    save_dataset_npz(
        {
            "obs": np.zeros((2, 52), dtype=np.float32),
            "next_obs": np.ones((2, 52), dtype=np.float32),
            "action": np.zeros((2, 1), dtype=np.float32),
            "reward": np.zeros(2, dtype=np.float32),
            "done": np.array([False, True]),
        },
        output,
    )

    with np.load(output) as data:
        assert sorted(data.files) == [
            "actions",
            "dones",
            "next_observations",
            "observations",
            "rewards",
        ]
        assert data["observations"].shape == (2, 52)
        assert data["next_observations"].shape == (2, 52)
