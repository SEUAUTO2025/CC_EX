from __future__ import annotations

from pathlib import Path
from typing import Any

import copy
import json

import torch
import torch.nn.functional as F

from dar_td3bc.algorithms.td3bc import (
    masked_prediction_loss,
    soft_update,
    td3bc_actor_loss,
)
from dar_td3bc.data.pipeline_dataset import PipelineArrays
from dar_td3bc.data.pipeline_obs import split_pipeline_obs
from dar_td3bc.data.fast_batch import FastTransitionBatches
from dar_td3bc.models.critic import TwinCritic, critic_disagreement_gate
from dar_td3bc.models.policies import (
    BehaviorPolicy,
    MultiHorizonPredictionHead,
    ResidualActor,
    compose_residual_action,
)
from dar_td3bc.models.temporal_encoder import DelayEncoder
from dar_td3bc.utils.device import resolve_device
from dar_td3bc.utils.progress import progress_enabled, progress_range
from dar_td3bc.utils.run import append_csv_row, make_run_dir, should_run_interval
from dar_td3bc.utils.seed import set_global_seed


def train_td3bc(
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
    model_cfg = config.get("model", {})
    loss_cfg = config.get("loss", {})
    batch_size = int(train_cfg.get("batch_size", 1024))
    hidden_dim = int(model_cfg.get("hidden_dim", 256))
    discount = float(train_cfg.get("discount", 0.99))
    tau = float(train_cfg.get("tau", 0.005))
    policy_noise = float(train_cfg.get("policy_noise", 0.2))
    noise_clip = float(train_cfg.get("noise_clip", 0.5))
    policy_delay = int(train_cfg.get("policy_delay", 2))
    log_interval = int(train_cfg.get("log_interval", 100))
    validation_interval = int(train_cfg.get("validation_interval", 1000))

    run_dir = make_run_dir(output_root, "td3bc_mlp", seed, run_name=run_name)
    _write_config(run_dir, config, seed, steps)

    actor = BehaviorPolicy(input_dim=52, hidden_dim=hidden_dim).to(device)
    critic = TwinCritic(latent_dim=52, hidden_dim=hidden_dim).to(device)
    target_actor = copy.deepcopy(actor)
    target_critic = copy.deepcopy(critic)
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=float(train_cfg.get("actor_lr", 3e-4))
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=float(train_cfg.get("critic_lr", 3e-4))
    )

    train_batches = FastTransitionBatches(train_arrays, device=device)
    best_val = float("inf")
    last_actor_loss = torch.tensor(0.0, device=device)
    for step in progress_range(
        steps,
        desc=f"td3bc seed={seed}",
        enabled=progress_enabled(config),
    ):
        batch = train_batches.sample(batch_size)
        with torch.no_grad():
            noise = torch.randn_like(batch.actions).mul(policy_noise)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (target_actor(batch.next_obs) + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = target_critic(batch.next_obs, next_action)
            target_q = batch.rewards + discount * (1.0 - batch.dones) * torch.minimum(
                target_q1, target_q2
            )

        q1, q2 = critic(batch.obs, batch.actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        if step % policy_delay == 0:
            policy_action = actor(batch.obs)
            q1_pi, q2_pi = critic(batch.obs, policy_action)
            actor_components = td3bc_actor_loss(
                torch.minimum(q1_pi, q2_pi),
                policy_action,
                batch.actions,
                policy_action - batch.actions,
                batch.previous_actions,
                alpha_td3bc=float(loss_cfg.get("alpha_td3bc", 2.5)),
                alpha_bc=float(loss_cfg.get("alpha_bc", 1.0)),
                alpha_res=float(loss_cfg.get("alpha_res", 0.0)),
                alpha_tv=float(loss_cfg.get("alpha_tv", 0.0)),
            )
            actor_optimizer.zero_grad()
            actor_components.total.backward()
            actor_optimizer.step()
            soft_update(actor, target_actor, tau)
            soft_update(critic, target_critic, tau)
            last_actor_loss = actor_components.total.detach()

        if should_run_interval(step, total_steps=steps, interval=log_interval):
            append_csv_row(
                run_dir / "metrics_train.csv",
                {
                    "step": step,
                    "loss_critic": float(critic_loss.detach().cpu()),
                    "loss_actor": float(last_actor_loss.cpu()),
                },
            )
        if should_run_interval(step, total_steps=steps, interval=validation_interval):
            val_loss = _td3bc_val_loss(actor, val_arrays, batch_size)
            append_csv_row(
                run_dir / "metrics_validation.csv",
                {"step": step, "loss_bc": val_loss},
            )
            if val_loss <= best_val:
                best_val = val_loss
                torch.save(
                    _td3bc_checkpoint(
                        actor,
                        critic,
                        target_actor,
                        target_critic,
                        config,
                        seed,
                        step,
                        val_loss,
                    ),
                    run_dir / "checkpoint_best.pt",
                )

    torch.save(
        _td3bc_checkpoint(actor, critic, target_actor, target_critic, config, seed, steps, best_val),
        run_dir / "checkpoint_last.pt",
    )
    return run_dir


def train_dar_td3bc(
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
    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {})
    loss_cfg = config.get("loss", {})
    horizons = tuple(int(v) for v in dataset_cfg.get("horizons", [1, 5, 10, 20]))
    batch_size = int(train_cfg.get("batch_size", 1024))
    hidden_dim = int(model_cfg.get("hidden_dim", 256))
    latent_dim = int(model_cfg.get("latent_dim", 128))
    dilations = tuple(int(v) for v in model_cfg.get("tcn_dilations", [1, 2, 4, 8]))
    discount = float(train_cfg.get("discount", 0.99))
    tau = float(train_cfg.get("tau", 0.005))
    policy_noise = float(train_cfg.get("policy_noise", 0.2))
    noise_clip = float(train_cfg.get("noise_clip", 0.5))
    policy_delay = int(train_cfg.get("policy_delay", 2))
    log_interval = int(train_cfg.get("log_interval", 100))
    validation_interval = int(train_cfg.get("validation_interval", 1000))
    residual_scale = float(model_cfg.get("residual_scale", 0.25))
    gate_min = float(model_cfg.get("gate_min", 0.05))
    gate_kappa = float(model_cfg.get("gate_kappa", 1.0))
    alpha_pred = float(loss_cfg.get("alpha_pred", 1.0))

    run_dir = make_run_dir(output_root, "dar_td3bc", seed, run_name=run_name)
    _write_config(run_dir, config, seed, steps)

    behavior = BehaviorPolicy(input_dim=52, hidden_dim=hidden_dim)
    encoder = DelayEncoder(
        hidden_dim=hidden_dim, latent_dim=latent_dim, dilations=dilations
    )
    prediction_head = MultiHorizonPredictionHead(
        latent_dim=latent_dim, horizons=horizons
    )
    actor = ResidualActor(latent_dim=latent_dim, hidden_dim=hidden_dim)
    critic = TwinCritic(latent_dim=latent_dim, hidden_dim=hidden_dim)
    for model in (behavior, encoder, prediction_head, actor, critic):
        model.to(device)
    target_behavior = copy.deepcopy(behavior)
    target_encoder = copy.deepcopy(encoder)
    target_actor = copy.deepcopy(actor)
    target_critic = copy.deepcopy(critic)

    behavior_optimizer = torch.optim.Adam(
        behavior.parameters(), lr=float(train_cfg.get("behavior_lr", 3e-4))
    )
    representation_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(prediction_head.parameters()),
        lr=float(train_cfg.get("encoder_lr", 3e-4)),
    )
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=float(train_cfg.get("actor_lr", 3e-4))
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=float(train_cfg.get("critic_lr", 3e-4))
    )

    train_batches = FastTransitionBatches(
        train_arrays,
        horizons=horizons,
        device=device,
    )
    best_val = float("inf")
    last_actor_loss = torch.tensor(0.0, device=device)
    last_gate = torch.tensor(1.0, device=device)
    for step in progress_range(
        steps,
        desc=f"dar_td3bc seed={seed}",
        enabled=progress_enabled(config),
    ):
        batch = train_batches.sample(batch_size)
        behavior_loss = F.mse_loss(behavior(batch.obs), batch.actions)
        behavior_optimizer.zero_grad()
        behavior_loss.backward()
        behavior_optimizer.step()

        parsed = split_pipeline_obs(batch.obs)
        latent = encoder(parsed.sequence, parsed.current_flow, parsed.target_flow)
        pred = prediction_head(latent)
        pred_loss_raw = masked_prediction_loss(pred, batch.future_flows, batch.future_mask)
        pred_loss = alpha_pred * pred_loss_raw
        representation_optimizer.zero_grad()
        pred_loss.backward()
        representation_optimizer.step()

        with torch.no_grad():
            parsed_next = split_pipeline_obs(batch.next_obs)
            next_latent = target_encoder(
                parsed_next.sequence,
                parsed_next.current_flow,
                parsed_next.target_flow,
            )
            next_behavior = target_behavior(batch.next_obs)
            next_residual = target_actor(next_latent)
            next_candidate = compose_residual_action(
                next_behavior,
                next_residual,
                torch.ones_like(next_behavior),
                residual_scale=residual_scale,
            )
            target_q1_probe, target_q2_probe = target_critic(
                next_latent, next_candidate
            )
            next_gate, _ = critic_disagreement_gate(
                target_q1_probe,
                target_q2_probe,
                kappa=gate_kappa,
                gate_min=gate_min,
            )
            noise = torch.randn_like(batch.actions).mul(policy_noise)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (compose_residual_action(
                next_behavior,
                next_residual,
                next_gate,
                residual_scale=residual_scale,
            ) + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = target_critic(next_latent, next_action)
            target_q = batch.rewards + discount * (1.0 - batch.dones) * torch.minimum(
                target_q1, target_q2
            )

        with torch.no_grad():
            critic_latent = encoder(
                parsed.sequence,
                parsed.current_flow,
                parsed.target_flow,
            )
        q1, q2 = critic(critic_latent, batch.actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        if step % policy_delay == 0:
            with torch.no_grad():
                actor_latent = encoder(
                    parsed.sequence,
                    parsed.current_flow,
                    parsed.target_flow,
                )
                behavior_action = behavior(batch.obs)
            residual = actor(actor_latent)
            candidate_action = compose_residual_action(
                behavior_action,
                residual,
                torch.ones_like(behavior_action),
                residual_scale=residual_scale,
            )
            q1_probe, q2_probe = critic(actor_latent, candidate_action)
            gate, _ = critic_disagreement_gate(
                q1_probe, q2_probe, kappa=gate_kappa, gate_min=gate_min
            )
            policy_action = compose_residual_action(
                behavior_action,
                residual,
                gate,
                residual_scale=residual_scale,
            )
            q1_pi, q2_pi = critic(actor_latent, policy_action)
            actor_components = td3bc_actor_loss(
                torch.minimum(q1_pi, q2_pi),
                policy_action,
                batch.actions,
                residual,
                batch.previous_actions,
                alpha_td3bc=float(loss_cfg.get("alpha_td3bc", 2.5)),
                alpha_bc=float(loss_cfg.get("alpha_bc", 1.0)),
                alpha_res=float(loss_cfg.get("alpha_res", 0.01)),
                alpha_tv=float(loss_cfg.get("alpha_tv", 0.01)),
            )
            actor_optimizer.zero_grad()
            actor_components.total.backward()
            actor_optimizer.step()
            soft_update(behavior, target_behavior, tau)
            soft_update(encoder, target_encoder, tau)
            soft_update(actor, target_actor, tau)
            soft_update(critic, target_critic, tau)
            last_actor_loss = actor_components.total.detach()
            last_gate = gate.mean().detach()

        if should_run_interval(step, total_steps=steps, interval=log_interval):
            append_csv_row(
                run_dir / "metrics_train.csv",
                {
                    "step": step,
                    "loss_behavior": float(behavior_loss.detach().cpu()),
                    "loss_pred": float(pred_loss.detach().cpu()),
                    "loss_pred_raw": float(pred_loss_raw.detach().cpu()),
                    "alpha_pred": alpha_pred,
                    "loss_critic": float(critic_loss.detach().cpu()),
                    "loss_actor": float(last_actor_loss.cpu()),
                    "gate_mean": float(last_gate.cpu()),
                },
            )
        if should_run_interval(step, total_steps=steps, interval=validation_interval):
            val_loss = _dar_val_loss(
                behavior,
                encoder,
                prediction_head,
                val_arrays,
                horizons,
                batch_size,
                alpha_pred=alpha_pred,
            )
            append_csv_row(
                run_dir / "metrics_validation.csv",
                {"step": step, "loss_validation": val_loss},
            )
            if val_loss <= best_val:
                best_val = val_loss
                torch.save(
                    _dar_checkpoint(
                        behavior,
                        encoder,
                        prediction_head,
                        actor,
                        critic,
                        target_behavior,
                        target_encoder,
                        target_actor,
                        target_critic,
                        config,
                        seed,
                        step,
                        val_loss,
                    ),
                    run_dir / "checkpoint_best.pt",
                )

    torch.save(
        _dar_checkpoint(
            behavior,
            encoder,
            prediction_head,
            actor,
            critic,
            target_behavior,
            target_encoder,
            target_actor,
            target_critic,
            config,
            seed,
            steps,
            best_val,
        ),
        run_dir / "checkpoint_last.pt",
    )
    return run_dir


@torch.no_grad()
def _td3bc_val_loss(
    actor: BehaviorPolicy, arrays: PipelineArrays, batch_size: int
) -> float:
    actor.eval()
    device = next(actor.parameters()).device
    batches = FastTransitionBatches(arrays, device=device)
    losses = []
    for batch in batches.iter_batches(batch_size=batch_size, shuffle=False):
        losses.append(F.mse_loss(actor(batch.obs), batch.actions).item())
    actor.train()
    return float(sum(losses) / max(len(losses), 1))


@torch.no_grad()
def _dar_val_loss(
    behavior: BehaviorPolicy,
    encoder: DelayEncoder,
    prediction_head: MultiHorizonPredictionHead,
    arrays: PipelineArrays,
    horizons: tuple[int, ...],
    batch_size: int,
    alpha_pred: float = 1.0,
) -> float:
    behavior.eval()
    encoder.eval()
    prediction_head.eval()
    device = next(encoder.parameters()).device
    batches = FastTransitionBatches(arrays, horizons=horizons, device=device)
    losses = []
    for batch in batches.iter_batches(batch_size=batch_size, shuffle=False):
        parsed = split_pipeline_obs(batch.obs)
        latent = encoder(parsed.sequence, parsed.current_flow, parsed.target_flow)
        behavior_loss = F.mse_loss(behavior(batch.obs), batch.actions)
        pred_loss = masked_prediction_loss(
            prediction_head(latent), batch.future_flows, batch.future_mask
        )
        losses.append((behavior_loss + alpha_pred * pred_loss).item())
    behavior.train()
    encoder.train()
    prediction_head.train()
    return float(sum(losses) / max(len(losses), 1))


def _td3bc_checkpoint(
    actor: BehaviorPolicy,
    critic: TwinCritic,
    target_actor: BehaviorPolicy,
    target_critic: TwinCritic,
    config: dict[str, Any],
    seed: int,
    step: int,
    val_loss: float,
) -> dict[str, Any]:
    return {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "target_actor": target_actor.state_dict(),
        "target_critic": target_critic.state_dict(),
        "config": config,
        "seed": seed,
        "step": step,
        "val_loss": val_loss,
    }


def _dar_checkpoint(
    behavior: BehaviorPolicy,
    encoder: DelayEncoder,
    prediction_head: MultiHorizonPredictionHead,
    actor: ResidualActor,
    critic: TwinCritic,
    target_behavior: BehaviorPolicy,
    target_encoder: DelayEncoder,
    target_actor: ResidualActor,
    target_critic: TwinCritic,
    config: dict[str, Any],
    seed: int,
    step: int,
    val_loss: float,
) -> dict[str, Any]:
    return {
        "behavior": behavior.state_dict(),
        "encoder": encoder.state_dict(),
        "prediction_head": prediction_head.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "target_behavior": target_behavior.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "target_actor": target_actor.state_dict(),
        "target_critic": target_critic.state_dict(),
        "config": config,
        "seed": seed,
        "step": step,
        "val_loss": val_loss,
    }


def _write_config(run_dir: Path, config: dict[str, Any], seed: int, steps: int) -> None:
    payload = {"seed": seed, "steps": steps, "config": config}
    (run_dir / "config_resolved.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
