import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dar_td3bc.data.pipeline_dataset import PipelineArrays


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_npz(path: Path, n: int = 4, done_key: str = "terminals") -> None:
    data = {
        "observations": np.zeros((n, 52), dtype=np.float32),
        "actions": np.zeros((n, 1), dtype=np.float32),
        "rewards": np.arange(n, dtype=np.float32),
        "next_observations": np.ones((n, 52), dtype=np.float32),
        done_key: np.array([False] * (n - 1) + [True]),
    }
    np.savez_compressed(path, **data)


def test_pipeline_arrays_loads_npz_and_maps_done_fields(tmp_path):
    path = tmp_path / "pipeline.npz"
    _write_npz(path, n=5, done_key="dones")

    arrays = PipelineArrays.from_npz(path)

    assert arrays.size == 5
    assert arrays.observations.shape == (5, 52)
    assert arrays.actions.shape == (5, 1)
    assert arrays.rewards.shape == (5, 1)
    assert arrays.next_observations.shape == (5, 52)
    assert arrays.dones.tolist() == [[False], [False], [False], [False], [True]]
    assert arrays.timeouts.tolist() == [[False], [False], [False], [False], [False]]


def test_pipeline_arrays_rejects_wrong_observation_shape(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        observations=np.zeros((3, 51)),
        actions=np.zeros((3, 1)),
        rewards=np.zeros(3),
        next_observations=np.zeros((3, 52)),
        terminals=np.zeros(3, dtype=bool),
    )

    with pytest.raises(ValueError, match="observations must have shape"):
        PipelineArrays.from_npz(path)


def test_validation_summary_reports_shapes_and_ranges(tmp_path):
    path = tmp_path / "pipeline.npz"
    _write_npz(path, n=3)

    summary = PipelineArrays.from_npz(path).validation_summary()

    assert summary["num_transitions"] == 3
    assert summary["observation_shape"] == [3, 52]
    assert summary["action_shape"] == [3, 1]
    assert summary["reward_min"] == 0.0
    assert summary["reward_max"] == 2.0
    assert summary["episode_end_count"] == 1


def test_prepare_pipeline_data_script_writes_metadata(tmp_path):
    train = tmp_path / "train.npz"
    val = tmp_path / "val.npz"
    output = tmp_path / "metadata"
    _write_npz(train, n=4)
    _write_npz(val, n=2)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_pipeline_data.py"),
            "--train",
            str(train),
            "--val",
            str(val),
            "--output-dir",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((output / "pipeline_manifest.json").read_text())
    assert report["train"]["num_transitions"] == 4
    assert report["val"]["num_transitions"] == 2
