import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from dar_td3bc.training.offline_rl import train_dar_td3bc, train_td3bc


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_dataset(path: Path, n: int = 20) -> None:
    obs = np.zeros((n, 52), dtype=np.float32)
    obs[:, 0] = np.linspace(0.0, 1.0, n)
    obs[:, 1] = 0.5
    obs[:, 2:27] = np.linspace(0.0, 1.0, 25, dtype=np.float32)
    obs[:, 27:52] = np.linspace(-0.5, 0.5, 25, dtype=np.float32)
    next_obs = obs.copy()
    next_obs[:, 0] = np.roll(obs[:, 0], -1)
    actions = np.tanh((obs[:, 1:2] - obs[:, 0:1]) * 2.0).astype(np.float32)
    rewards = -np.abs(obs[:, 0:1] - obs[:, 1:2]).astype(np.float32)
    dones = np.array([False] * (n - 1) + [True]).reshape(-1, 1)
    np.savez_compressed(
        path,
        observations=obs,
        actions=actions,
        rewards=rewards,
        next_observations=next_obs,
        dones=dones,
    )


def _config() -> dict:
    return {
        "dataset": {"horizons": [1, 2]},
        "model": {
            "hidden_dim": 16,
            "latent_dim": 12,
            "tcn_dilations": [1, 2],
            "residual_scale": 0.25,
            "gate_min": 0.05,
            "gate_kappa": 1.0,
        },
        "train": {
            "batch_size": 8,
            "actor_lr": 1e-3,
            "critic_lr": 1e-3,
            "encoder_lr": 1e-3,
            "behavior_lr": 1e-3,
            "discount": 0.99,
            "tau": 0.01,
            "policy_noise": 0.1,
            "noise_clip": 0.2,
            "policy_delay": 2,
        },
        "loss": {
            "alpha_td3bc": 2.5,
            "alpha_bc": 1.0,
            "alpha_res": 0.01,
            "alpha_tv": 0.01,
            "alpha_pred": 0.1,
        },
    }


def test_train_td3bc_writes_checkpoint_and_metrics(tmp_path):
    train = tmp_path / "train.npz"
    val = tmp_path / "val.npz"
    _write_dataset(train)
    _write_dataset(val)

    run_dir = train_td3bc(
        train_path=train,
        val_path=val,
        config=_config(),
        seed=0,
        steps=2,
        output_root=tmp_path / "runs",
        run_name="td3bc_smoke",
    )

    assert (run_dir / "checkpoint_best.pt").exists()
    assert (run_dir / "checkpoint_last.pt").exists()
    assert (run_dir / "metrics_train.csv").exists()
    assert (run_dir / "metrics_validation.csv").exists()


def test_train_dar_td3bc_writes_checkpoint_and_metrics(tmp_path):
    train = tmp_path / "train.npz"
    val = tmp_path / "val.npz"
    _write_dataset(train)
    _write_dataset(val)

    run_dir = train_dar_td3bc(
        train_path=train,
        val_path=val,
        config=_config(),
        seed=0,
        steps=2,
        output_root=tmp_path / "runs",
        run_name="dar_smoke",
    )

    assert (run_dir / "checkpoint_best.pt").exists()
    assert (run_dir / "checkpoint_last.pt").exists()
    assert (run_dir / "metrics_train.csv").exists()
    assert (run_dir / "metrics_validation.csv").exists()


def test_dar_checkpoint_contains_optimizer_state_for_resume(tmp_path):
    train = tmp_path / "train.npz"
    val = tmp_path / "val.npz"
    _write_dataset(train)
    _write_dataset(val)

    run_dir = train_dar_td3bc(
        train_path=train,
        val_path=val,
        config={**_config(), "progress": False},
        seed=0,
        steps=2,
        output_root=tmp_path / "runs",
        run_name="dar_resume",
    )

    checkpoint = torch.load(run_dir / "checkpoint_last.pt", map_location="cpu")

    assert checkpoint["step"] == 2
    assert "optimizers" in checkpoint
    assert set(checkpoint["optimizers"]) == {
        "behavior",
        "representation",
        "actor",
        "critic",
    }


def test_train_dar_td3bc_resume_continues_from_last_checkpoint(tmp_path):
    train = tmp_path / "train.npz"
    val = tmp_path / "val.npz"
    _write_dataset(train)
    _write_dataset(val)

    train_dar_td3bc(
        train_path=train,
        val_path=val,
        config={**_config(), "progress": False},
        seed=0,
        steps=2,
        output_root=tmp_path / "runs",
        run_name="dar_resume",
    )
    run_dir = train_dar_td3bc(
        train_path=train,
        val_path=val,
        config={**_config(), "progress": False},
        seed=0,
        steps=4,
        output_root=tmp_path / "runs",
        run_name="dar_resume",
        resume=True,
    )

    checkpoint = torch.load(run_dir / "checkpoint_last.pt", map_location="cpu")

    assert checkpoint["step"] == 4


def test_offline_rl_training_scripts_run_on_synthetic_npz(tmp_path):
    train = tmp_path / "train.npz"
    val = tmp_path / "val.npz"
    config = tmp_path / "config.yaml"
    output = tmp_path / "runs"
    _write_dataset(train)
    _write_dataset(val)
    config.write_text(
        "dataset:\n  horizons: [1, 2]\n"
        "model:\n  hidden_dim: 16\n  latent_dim: 12\n"
        "  tcn_dilations: [1, 2]\n  residual_scale: 0.25\n"
        "  gate_min: 0.05\n  gate_kappa: 1.0\n"
        "train:\n  batch_size: 8\n  actor_lr: 0.001\n  critic_lr: 0.001\n"
        "  encoder_lr: 0.001\n  behavior_lr: 0.001\n"
        "  discount: 0.99\n  tau: 0.01\n  policy_noise: 0.1\n"
        "  noise_clip: 0.2\n  policy_delay: 2\n"
        "loss:\n  alpha_td3bc: 2.5\n  alpha_bc: 1.0\n"
        "  alpha_res: 0.01\n  alpha_tv: 0.01\n  alpha_pred: 0.1\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    for script in ("train_td3bc.py", "train_dar_td3bc.py"):
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
