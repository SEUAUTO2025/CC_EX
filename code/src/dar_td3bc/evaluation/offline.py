from __future__ import annotations

from pathlib import Path

import csv

import numpy as np
import torch
import torch.nn.functional as F

from dar_td3bc.data.pipeline_dataset import PipelineArrays
from dar_td3bc.evaluation.control_metrics import (
    action_energy,
    action_total_variation,
    iae,
    mae,
    rmse,
)
from dar_td3bc.evaluation.policy import predict_checkpoint_actions
from dar_td3bc.utils.device import resolve_device


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    val_path: str | Path,
    output_dir: str | Path,
    method: str | None = None,
    device: str | torch.device | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    val_path = Path(val_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    if device is not None:
        config = dict(config)
        config["device"] = str(device)
        checkpoint["config"] = config
    resolved_device = resolve_device(config)
    method_name = method or _infer_method(checkpoint)
    arrays = PipelineArrays.from_npz(val_path)
    observations = torch.as_tensor(arrays.observations, dtype=torch.float32)
    dataset_actions = torch.as_tensor(
        arrays.actions, dtype=torch.float32, device=resolved_device
    )
    policy_actions = predict_checkpoint_actions(
        checkpoint, observations, device=resolved_device
    )
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


def _infer_method(checkpoint: dict) -> str:
    return "dar_td3bc" if "behavior" in checkpoint else "td3bc_mlp"
