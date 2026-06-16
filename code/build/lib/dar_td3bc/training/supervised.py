from __future__ import annotations

from pathlib import Path
from typing import Any

import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dar_td3bc.algorithms.td3bc import masked_prediction_loss
from dar_td3bc.data.pipeline_dataset import PipelineArrays
from dar_td3bc.data.pipeline_obs import split_pipeline_obs
from dar_td3bc.data.torch_dataset import (
    PipelineTransitionDataset,
    collate_transition_batch,
)
from dar_td3bc.models.policies import BehaviorPolicy, MultiHorizonPredictionHead
from dar_td3bc.models.temporal_encoder import DelayEncoder
from dar_td3bc.utils.run import append_csv_row, make_run_dir
from dar_td3bc.utils.seed import set_global_seed


def train_behavior_policy(
    *,
    train_path: str | Path,
    val_path: str | Path,
    config: dict[str, Any],
    seed: int,
    steps: int,
    output_root: str | Path,
    run_name: str | None = None,
) -> Path:
    set_global_seed(seed)
    train_arrays = PipelineArrays.from_npz(train_path)
    val_arrays = PipelineArrays.from_npz(val_path)
    batch_size = int(config.get("train", {}).get("batch_size", 256))
    lr = float(config.get("train", {}).get("behavior_lr", 3e-4))
    hidden_dim = int(config.get("model", {}).get("hidden_dim", 256))
    run_dir = make_run_dir(output_root, "behavior_policy", seed, run_name=run_name)
    _write_config(run_dir, config, seed, steps)

    model = BehaviorPolicy(input_dim=52, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        PipelineTransitionDataset(train_arrays),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_transition_batch,
    )
    iterator = _cycle(loader)
    best_val = float("inf")
    for step in range(1, steps + 1):
        batch = next(iterator)
        pred = model(batch.obs)
        loss = F.mse_loss(pred, batch.actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        append_csv_row(
            run_dir / "metrics_train.csv",
            {"step": step, "loss_bc": float(loss.detach().cpu())},
        )
        val_loss = _behavior_val_loss(model, val_arrays, batch_size)
        append_csv_row(
            run_dir / "metrics_validation.csv",
            {"step": step, "loss_bc": val_loss},
        )
        if val_loss <= best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "seed": seed,
                    "config": config,
                    "step": step,
                    "val_loss": val_loss,
                },
                run_dir / "checkpoint_best.pt",
            )
    torch.save({"model": model.state_dict(), "seed": seed, "config": config}, run_dir / "checkpoint_last.pt")
    return run_dir


def pretrain_encoder(
    *,
    train_path: str | Path,
    val_path: str | Path,
    config: dict[str, Any],
    seed: int,
    steps: int,
    output_root: str | Path,
    run_name: str | None = None,
) -> Path:
    set_global_seed(seed)
    train_arrays = PipelineArrays.from_npz(train_path)
    val_arrays = PipelineArrays.from_npz(val_path)
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    horizons = tuple(int(v) for v in dataset_cfg.get("horizons", [1, 5, 10, 20]))
    batch_size = int(train_cfg.get("batch_size", 256))
    lr = float(train_cfg.get("encoder_lr", 3e-4))
    hidden_dim = int(model_cfg.get("hidden_dim", 256))
    latent_dim = int(model_cfg.get("latent_dim", 128))
    dilations = tuple(int(v) for v in model_cfg.get("tcn_dilations", [1, 2, 4, 8]))
    run_dir = make_run_dir(output_root, "encoder_pretrain", seed, run_name=run_name)
    _write_config(run_dir, config, seed, steps)

    encoder = DelayEncoder(
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dilations=dilations,
    )
    head = MultiHorizonPredictionHead(latent_dim=latent_dim, horizons=horizons)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=lr
    )
    loader = DataLoader(
        PipelineTransitionDataset(train_arrays, horizons=horizons),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_transition_batch,
    )
    iterator = _cycle(loader)
    best_val = float("inf")
    for step in range(1, steps + 1):
        batch = next(iterator)
        parsed = split_pipeline_obs(batch.obs)
        latent = encoder(parsed.sequence, parsed.current_flow, parsed.target_flow)
        pred = head(latent)
        loss = masked_prediction_loss(pred, batch.future_flows, batch.future_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        append_csv_row(
            run_dir / "metrics_train.csv",
            {"step": step, "loss_pred": float(loss.detach().cpu())},
        )
        val_loss = _encoder_val_loss(encoder, head, val_arrays, horizons, batch_size)
        append_csv_row(
            run_dir / "metrics_validation.csv",
            {"step": step, "loss_pred": val_loss},
        )
        if val_loss <= best_val:
            best_val = val_loss
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "prediction_head": head.state_dict(),
                    "seed": seed,
                    "config": config,
                    "step": step,
                    "val_loss": val_loss,
                },
                run_dir / "checkpoint_best.pt",
            )
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "prediction_head": head.state_dict(),
            "seed": seed,
            "config": config,
        },
        run_dir / "checkpoint_last.pt",
    )
    return run_dir


def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def _behavior_val_loss(
    model: BehaviorPolicy, arrays: PipelineArrays, batch_size: int
) -> float:
    model.eval()
    losses = []
    for batch in DataLoader(
        PipelineTransitionDataset(arrays),
        batch_size=batch_size,
        collate_fn=collate_transition_batch,
    ):
        losses.append(F.mse_loss(model(batch.obs), batch.actions).item())
    model.train()
    return float(sum(losses) / max(len(losses), 1))


@torch.no_grad()
def _encoder_val_loss(
    encoder: DelayEncoder,
    head: MultiHorizonPredictionHead,
    arrays: PipelineArrays,
    horizons: tuple[int, ...],
    batch_size: int,
) -> float:
    encoder.eval()
    head.eval()
    losses = []
    for batch in DataLoader(
        PipelineTransitionDataset(arrays, horizons=horizons),
        batch_size=batch_size,
        collate_fn=collate_transition_batch,
    ):
        parsed = split_pipeline_obs(batch.obs)
        latent = encoder(parsed.sequence, parsed.current_flow, parsed.target_flow)
        pred = head(latent)
        losses.append(
            masked_prediction_loss(pred, batch.future_flows, batch.future_mask).item()
        )
    encoder.train()
    head.train()
    return float(sum(losses) / max(len(losses), 1))


def _write_config(run_dir: Path, config: dict[str, Any], seed: int, steps: int) -> None:
    payload = {"seed": seed, "steps": steps, "config": config}
    (run_dir / "config_resolved.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
