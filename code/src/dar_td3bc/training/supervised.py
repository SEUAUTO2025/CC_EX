from __future__ import annotations

from pathlib import Path
from typing import Any

import json

import torch
import torch.nn.functional as F

from dar_td3bc.algorithms.td3bc import masked_prediction_loss
from dar_td3bc.data.fast_batch import FastTransitionBatches
from dar_td3bc.data.pipeline_dataset import PipelineArrays
from dar_td3bc.data.pipeline_obs import split_pipeline_obs
from dar_td3bc.models.policies import BehaviorPolicy, MultiHorizonPredictionHead
from dar_td3bc.models.temporal_encoder import DelayEncoder
from dar_td3bc.utils.device import resolve_device
from dar_td3bc.utils.progress import progress_enabled, progress_range
from dar_td3bc.utils.run import append_csv_row, make_run_dir, should_run_interval
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
    device = resolve_device(config)
    train_cfg = config.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 1024))
    lr = float(train_cfg.get("behavior_lr", 3e-4))
    log_interval = int(train_cfg.get("log_interval", 100))
    validation_interval = int(train_cfg.get("validation_interval", 1000))
    hidden_dim = int(config.get("model", {}).get("hidden_dim", 256))
    run_dir = make_run_dir(output_root, "behavior_policy", seed, run_name=run_name)
    _write_config(run_dir, config, seed, steps)

    model = BehaviorPolicy(input_dim=52, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_batches = FastTransitionBatches(train_arrays, device=device)
    best_val = float("inf")
    for step in progress_range(
        steps,
        desc=f"behavior seed={seed}",
        enabled=progress_enabled(config),
    ):
        batch = train_batches.sample(batch_size)
        pred = model(batch.obs)
        loss = F.mse_loss(pred, batch.actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if should_run_interval(step, total_steps=steps, interval=log_interval):
            append_csv_row(
                run_dir / "metrics_train.csv",
                {"step": step, "loss_bc": float(loss.detach().cpu())},
            )
        if should_run_interval(step, total_steps=steps, interval=validation_interval):
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
    device = resolve_device(config)
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    horizons = tuple(int(v) for v in dataset_cfg.get("horizons", [1, 5, 10, 20]))
    batch_size = int(train_cfg.get("batch_size", 1024))
    lr = float(train_cfg.get("encoder_lr", 3e-4))
    log_interval = int(train_cfg.get("log_interval", 100))
    validation_interval = int(train_cfg.get("validation_interval", 1000))
    hidden_dim = int(model_cfg.get("hidden_dim", 256))
    latent_dim = int(model_cfg.get("latent_dim", 128))
    dilations = tuple(int(v) for v in model_cfg.get("tcn_dilations", [1, 2, 4, 8]))
    run_dir = make_run_dir(output_root, "encoder_pretrain", seed, run_name=run_name)
    _write_config(run_dir, config, seed, steps)

    encoder = DelayEncoder(
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dilations=dilations,
    ).to(device)
    head = MultiHorizonPredictionHead(latent_dim=latent_dim, horizons=horizons).to(
        device
    )
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=lr
    )
    train_batches = FastTransitionBatches(
        train_arrays,
        horizons=horizons,
        device=device,
    )
    best_val = float("inf")
    for step in progress_range(
        steps,
        desc=f"encoder seed={seed}",
        enabled=progress_enabled(config),
    ):
        batch = train_batches.sample(batch_size)
        parsed = split_pipeline_obs(batch.obs)
        latent = encoder(parsed.sequence, parsed.current_flow, parsed.target_flow)
        pred = head(latent)
        loss = masked_prediction_loss(pred, batch.future_flows, batch.future_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if should_run_interval(step, total_steps=steps, interval=log_interval):
            append_csv_row(
                run_dir / "metrics_train.csv",
                {"step": step, "loss_pred": float(loss.detach().cpu())},
            )
        if should_run_interval(step, total_steps=steps, interval=validation_interval):
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


@torch.no_grad()
def _behavior_val_loss(
    model: BehaviorPolicy, arrays: PipelineArrays, batch_size: int
) -> float:
    model.eval()
    device = next(model.parameters()).device
    batches = FastTransitionBatches(arrays, device=device)
    losses = []
    for batch in batches.iter_batches(batch_size=batch_size, shuffle=False):
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
    device = next(encoder.parameters()).device
    batches = FastTransitionBatches(arrays, horizons=horizons, device=device)
    losses = []
    for batch in batches.iter_batches(batch_size=batch_size, shuffle=False):
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
