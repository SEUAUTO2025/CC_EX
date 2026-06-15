import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dar_td3bc.evaluation.offline import evaluate_checkpoint
from dar_td3bc.results.aggregate import aggregate_run_metrics
from dar_td3bc.training.offline_rl import train_td3bc


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_dataset(path: Path, n: int = 20) -> None:
    obs = np.zeros((n, 52), dtype=np.float32)
    obs[:, 0] = np.linspace(0.0, 1.0, n)
    obs[:, 1] = 0.5
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
        "model": {"hidden_dim": 16, "latent_dim": 12, "tcn_dilations": [1, 2]},
        "train": {
            "batch_size": 8,
            "actor_lr": 1e-3,
            "critic_lr": 1e-3,
            "discount": 0.99,
            "tau": 0.01,
            "policy_delay": 2,
        },
    }


def test_evaluate_checkpoint_writes_offline_metrics(tmp_path):
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
        run_name="td3bc_eval",
    )

    metrics_path = evaluate_checkpoint(
        checkpoint_path=run_dir / "checkpoint_best.pt",
        val_path=val,
        output_dir=run_dir,
        method="td3bc_mlp",
    )

    frame = pd.read_csv(metrics_path)
    assert set(frame["provenance"]) == {"offline_validation_npz"}
    assert {"offline_action_mse", "tracking_rmse", "action_total_variation"} <= set(
        frame["metric"]
    )
    assert np.isfinite(frame["value"]).all()


def test_aggregate_run_metrics_writes_final_summary(tmp_path):
    run_root = tmp_path / "runs"
    run_dir = run_root / "td3bc_mlp" / "run_0"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics_eval.csv").write_text(
        "method,metric,value,provenance,checkpoint,dataset\n"
        "td3bc_mlp,tracking_rmse,1.0,offline_validation_npz,ckpt,val\n"
        "td3bc_mlp,tracking_rmse,3.0,offline_validation_npz,ckpt,val\n",
        encoding="utf-8",
    )

    output_dir = aggregate_run_metrics(run_root=run_root, output_dir=tmp_path / "agg")

    final_summary = pd.read_csv(output_dir / "final_summary.csv")
    assert final_summary.loc[0, "method"] == "td3bc_mlp"
    assert final_summary.loc[0, "metric"] == "tracking_rmse"
    assert final_summary.loc[0, "mean"] == 2.0
    assert (output_dir / "paired_tests.csv").exists()
    assert (tmp_path / "agg" / "main_results.tex").exists()


def test_evaluate_and_aggregate_scripts_run(tmp_path):
    train = tmp_path / "train.npz"
    val = tmp_path / "val.npz"
    config = tmp_path / "config.yaml"
    _write_dataset(train)
    _write_dataset(val)
    config.write_text(
        "model:\n  hidden_dim: 16\n  latent_dim: 12\n  tcn_dilations: [1, 2]\n"
        "train:\n  batch_size: 8\n  actor_lr: 0.001\n  critic_lr: 0.001\n"
        "  discount: 0.99\n  tau: 0.01\n  policy_delay: 2\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    train_result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_td3bc.py"),
            "--config",
            str(config),
            "--train",
            str(train),
            "--val",
            str(val),
            "--steps",
            "2",
            "--output-root",
            str(tmp_path / "runs"),
            "--run-name",
            "td3bc_script",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert train_result.returncode == 0, train_result.stderr
    checkpoint = next((tmp_path / "runs").glob("td3bc_mlp/*/checkpoint_best.pt"))

    eval_result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate.py"),
            "--checkpoint",
            str(checkpoint),
            "--val",
            str(val),
            "--method",
            "td3bc_mlp",
            "--output-dir",
            str(checkpoint.parent),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert eval_result.returncode == 0, eval_result.stderr

    agg_result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "aggregate_results.py"),
            "--run-root",
            str(tmp_path / "runs"),
            "--output",
            str(tmp_path / "aggregated"),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert agg_result.returncode == 0, agg_result.stderr
    assert (tmp_path / "aggregated" / "final_summary.csv").exists()
