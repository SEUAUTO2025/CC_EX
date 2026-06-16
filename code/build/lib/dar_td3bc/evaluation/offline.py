from __future__ import annotations

from pathlib import Path

import csv

import numpy as np
import torch
import torch.nn.functional as F

from dar_td3bc.data.pipeline_dataset import PipelineArrays
from dar_td3bc.data.pipeline_obs import split_pipeline_obs
from dar_td3bc.evaluation.control_metrics import (
    action_energy,
    action_total_variation,
    iae,
    mae,
    rmse,
)
from dar_td3bc.models.policies import BehaviorPolicy, ResidualActor, compose_residual_action
from dar_td3bc.models.temporal_encoder import DelayEncoder


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    val_path: str | Path,
    output_dir: str | Path,
    method: str | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    val_path = Path(val_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    method_name = method or _infer_method(checkpoint)
    arrays = PipelineArrays.from_npz(val_path)
    observations = torch.as_tensor(arrays.observations, dtype=torch.float32)
    dataset_actions = torch.as_tensor(arrays.actions, dtype=torch.float32)
    policy_actions = _predict_actions(checkpoint, observations)
    policy_actions_np = policy_actions.detach().cpu().numpy().reshape(-1)
    dataset_actions_np = arrays.actions.reshape(-1)
    flow = arrays.observations[:, 0]
    target = arrays.observations[:, 1]

    metrics = {
        "offline_action_mse": float(F.mse_loss(policy_actions, dataset_actions)),
        "offline_action_mae": float(
            np.mean(np.abs(policy_actions_np - dataset_actions_np))
        ),
        "tracking_rmse": rmse(flow, target),
        "tracking_mae": mae(flow, target),
        "tracking_iae": iae(flow, target),
        "action_total_variation": action_total_variation(policy_actions_np),
        "action_energy": action_energy(policy_actions_np),
        "num_validation_transitions": float(arrays.observations.shape[0]),
    }

    output_path = output_dir / "metrics_eval.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "metric",
                "value",
                "provenance",
                "checkpoint",
                "dataset",
            ],
        )
        writer.writeheader()
        for metric, value in metrics.items():
            writer.writerow(
                {
                    "method": method_name,
                    "metric": metric,
                    "value": value,
                    "provenance": "offline_validation_npz",
                    "checkpoint": str(checkpoint_path),
                    "dataset": str(val_path),
                }
            )
    return output_path


def _predict_actions(checkpoint: dict, observations: torch.Tensor) -> torch.Tensor:
    if "actor" in checkpoint and "behavior" not in checkpoint:
        actor = _make_behavior_policy(checkpoint)
        actor.load_state_dict(checkpoint["actor"])
        actor.eval()
        with torch.no_grad():
            return actor(observations).clamp(-1.0, 1.0)

    behavior = _make_behavior_policy(checkpoint)
    encoder = _make_encoder(checkpoint)
    actor = _make_residual_actor(checkpoint)
    behavior.load_state_dict(checkpoint["behavior"])
    encoder.load_state_dict(checkpoint["encoder"])
    actor.load_state_dict(checkpoint["actor"])
    behavior.eval()
    encoder.eval()
    actor.eval()
    model_cfg = checkpoint.get("config", {}).get("model", {})
    residual_scale = float(model_cfg.get("residual_scale", 0.25))
    with torch.no_grad():
        parsed = split_pipeline_obs(observations)
        latent = encoder(parsed.sequence, parsed.current_flow, parsed.target_flow)
        behavior_action = behavior(observations)
        residual = actor(latent)
        return compose_residual_action(
            behavior_action,
            residual,
            torch.ones_like(behavior_action),
            residual_scale=residual_scale,
        )


def _make_behavior_policy(checkpoint: dict) -> BehaviorPolicy:
    hidden_dim = int(checkpoint.get("config", {}).get("model", {}).get("hidden_dim", 256))
    return BehaviorPolicy(input_dim=52, hidden_dim=hidden_dim)


def _make_encoder(checkpoint: dict) -> DelayEncoder:
    model_cfg = checkpoint.get("config", {}).get("model", {})
    return DelayEncoder(
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        latent_dim=int(model_cfg.get("latent_dim", 128)),
        dilations=tuple(int(v) for v in model_cfg.get("tcn_dilations", [1, 2, 4, 8])),
    )


def _make_residual_actor(checkpoint: dict) -> ResidualActor:
    model_cfg = checkpoint.get("config", {}).get("model", {})
    return ResidualActor(
        latent_dim=int(model_cfg.get("latent_dim", 128)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
    )


def _infer_method(checkpoint: dict) -> str:
    return "dar_td3bc" if "behavior" in checkpoint else "td3bc_mlp"
