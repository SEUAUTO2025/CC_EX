import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from dar_td3bc.training.supervised import pretrain_encoder, train_behavior_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_dataset(path: Path, n: int = 16) -> None:
    obs = np.zeros((n, 52), dtype=np.float32)
    obs[:, 0] = np.linspace(0.0, 1.0, n)
    obs[:, 1] = 0.5
    obs[:, 2:27] = np.arange(25, dtype=np.float32)
    obs[:, 27:52] = np.arange(25, dtype=np.float32) / 25.0
    next_obs = obs.copy()
    next_obs[:, 0] = np.roll(obs[:, 0], -1)
    action = np.tanh(obs[:, 0:1])
    done = np.array([False] * (n - 1) + [True])
    np.savez_compressed(
        path,
        observations=obs,
        next_observations=next_obs,
        actions=action,
        rewards=-np.abs(obs[:, 0] - obs[:, 1]),
        dones=done,
    )


def _config() -> dict:
    return {
        "dataset": {"horizons": [1, 2]},
        "model": {"hidden_dim": 16, "latent_dim": 12, "tcn_dilations": [1, 2]},
        "train": {"batch_size": 8, "behavior_lr": 1e-3, "encoder_lr": 1e-3},
    }


def test_train_behavior_policy_writes_checkpoint_and_metrics(tmp_path):
    train = tmp_path / "train.npz"
    val = tmp_path / "val.npz"
    _write_dataset(train)
    _write_dataset(val)

    run_dir = train_behavior_policy(
        train_path=train,
        val_path=val,
        config=_config(),
        seed=0,
        steps=2,
        output_root=tmp_path / "runs",
        run_name="bc_smoke",
    )

    assert (run_dir / "checkpoint_best.pt").exists()
    assert (run_dir / "metrics_train.csv").exists()
    assert (run_dir / "metrics_validation.csv").exists()


def test_pretrain_encoder_writes_checkpoint_and_metrics(tmp_path):
    train = tmp_path / "train.npz"
    val = tmp_path / "val.npz"
    _write_dataset(train)
    _write_dataset(val)

    run_dir = pretrain_encoder(
        train_path=train,
        val_path=val,
        config=_config(),
        seed=0,
        steps=2,
        output_root=tmp_path / "runs",
        run_name="enc_smoke",
    )

    assert (run_dir / "checkpoint_best.pt").exists()
    assert (run_dir / "metrics_train.csv").exists()
    assert (run_dir / "metrics_validation.csv").exists()


def test_training_scripts_run_on_synthetic_npz(tmp_path):
    train = tmp_path / "train.npz"
    val = tmp_path / "val.npz"
    config = tmp_path / "config.yaml"
    output = tmp_path / "runs"
    _write_dataset(train)
    _write_dataset(val)
    config.write_text(
        "dataset:\n  horizons: [1, 2]\n"
        "model:\n  hidden_dim: 16\n  latent_dim: 12\n  tcn_dilations: [1, 2]\n"
        "train:\n  batch_size: 8\n  behavior_lr: 0.001\n  encoder_lr: 0.001\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    for script in ("train_behavior_policy.py", "pretrain_encoder.py"):
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / script),
                "--config",
                str(config),
                "--train",
                str(train),
                "--val",
                str(val),
                "--steps",
                "2",
                "--output-root",
                str(output),
                "--run-name",
                script.replace(".py", ""),
            ],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
