from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import csv

import numpy as np
import torch

from dar_td3bc.evaluation.control_metrics import (
    action_energy,
    action_total_variation,
    iae,
    mae,
    rmse,
    target_switch_settling_events,
)
from dar_td3bc.evaluation.neorl_score import normalized_score
from dar_td3bc.evaluation.policy import CheckpointPolicy
from dar_td3bc.utils.device import resolve_device
from dar_td3bc.utils.progress import progress_iter


def evaluate_checkpoint_rollout(
    *,
    checkpoint_path: str | Path,
    env_factory: Callable[[], Any],
    output_dir: str | Path,
    method: str | None = None,
    task: str = "Pipeline",
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    episodes_per_seed: int = 20,
    max_steps: int = 1000,
    output_name: str = "metrics_rollout.csv",
    metadata: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
) -> Path:
    if episodes_per_seed <= 0:
        raise ValueError("episodes_per_seed must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    if device is not None:
        config = dict(config)
        config["device"] = str(device)
        checkpoint["config"] = config
    resolved_device = resolve_device(config)
    policy = CheckpointPolicy(checkpoint, device=resolved_device)
    method_name = method or _infer_method(checkpoint)
    train_seed = _checkpoint_seed(checkpoint)
    metadata = metadata or {}

    eval_seeds = [int(seed) for seed in seeds]
    jobs = [
        (eval_seed, episode)
        for eval_seed in eval_seeds
        for episode in range(episodes_per_seed)
    ]
    rows: list[dict[str, Any]] = []
    for eval_seed, episode in progress_iter(
        jobs,
        desc=f"rollout {method_name}",
        total=len(jobs),
        enabled=True,
    ):
        episode_seed = int(eval_seed) * 1_000_000 + episode
        env = env_factory()
        try:
            episode_metrics = _rollout_episode(
                policy,
                env,
                seed=episode_seed,
                max_steps=max_steps,
            )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        for metric, value in episode_metrics.items():
            row = {
                "method": method_name,
                "metric": metric,
                "value": value,
                "provenance": "rollout_neorl2",
                "checkpoint": str(checkpoint_path),
                "task": task,
                "seed": train_seed,
                "eval_seed": int(eval_seed),
                "episode": int(episode),
            }
            row.update(metadata)
            rows.append(row)

    output_path = output_dir / output_name
    fieldnames = [
        "method",
        "metric",
        "value",
        "provenance",
        "checkpoint",
        "task",
        "seed",
        "eval_seed",
        "episode",
        *metadata.keys(),
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _rollout_episode(
    policy: CheckpointPolicy,
    env: Any,
    *,
    seed: int,
    max_steps: int,
) -> dict[str, float]:
    reset_result = env.reset(seed=seed)
    obs = _reset_obs(reset_result)
    observations = [obs]
    actions: list[float] = []
    raw_return = 0.0

    for _ in range(max_steps):
        action = _policy_action(policy, obs)
        step_result = env.step(action)
        next_obs, reward, done = _parse_step(step_result)
        scalar_action = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        actions.append(scalar_action)
        raw_return += float(reward)
        observations.append(next_obs)
        obs = next_obs
        if done:
            break

    flow = np.asarray([_as_pipeline_obs(item)[0] for item in observations], dtype=float)
    target = np.asarray([_as_pipeline_obs(item)[1] for item in observations], dtype=float)
    action_array = np.asarray(actions, dtype=float)
    metrics = {
        "raw_return": float(raw_return),
        "normalized_score": normalized_score(env, raw_return),
        "episode_length": float(len(actions)),
        "tracking_rmse": rmse(flow, target),
        "tracking_mae": mae(flow, target),
        "tracking_iae": iae(flow, target),
        "action_total_variation": action_total_variation(action_array),
        "action_energy": action_energy(action_array),
    }
    metrics.update(_settling_metrics(flow, target))
    return metrics


def _policy_action(policy: CheckpointPolicy, obs: np.ndarray) -> np.ndarray:
    obs_array = _as_pipeline_obs(obs)
    obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32).reshape(1, -1)
    action = policy.act(obs_tensor)
    return action.detach().cpu().numpy().reshape(-1).astype(np.float32)


def _settling_metrics(flow: np.ndarray, target: np.ndarray) -> dict[str, float]:
    events = target_switch_settling_events(flow, target)
    if not events:
        return {
            "settling_event_count": 0.0,
            "settling_censored_fraction": 0.0,
            "settling_time_mean": 0.0,
            "max_overshoot_mean": 0.0,
        }
    settling_times = [
        event.settling_time for event in events if event.settling_time is not None
    ]
    return {
        "settling_event_count": float(len(events)),
        "settling_censored_fraction": float(
            sum(1 for event in events if event.censored) / len(events)
        ),
        "settling_time_mean": (
            float(np.mean(settling_times)) if settling_times else float("nan")
        ),
        "max_overshoot_mean": float(np.mean([event.max_overshoot for event in events])),
    }


def _reset_obs(reset_result: Any) -> np.ndarray:
    if isinstance(reset_result, tuple):
        return _as_pipeline_obs(reset_result[0])
    return _as_pipeline_obs(reset_result)


def _parse_step(step_result: Any) -> tuple[np.ndarray, float, bool]:
    if not isinstance(step_result, tuple):
        raise TypeError("environment step must return a tuple")
    if len(step_result) == 5:
        obs, reward, terminated, truncated, _info = step_result
        return _as_pipeline_obs(obs), float(reward), bool(terminated or truncated)
    if len(step_result) == 4:
        obs, reward, done, _info = step_result
        return _as_pipeline_obs(obs), float(reward), bool(done)
    raise ValueError(f"expected Gym step tuple of length 4 or 5, got {len(step_result)}")


def _as_pipeline_obs(obs: Any) -> np.ndarray:
    array = np.asarray(obs, dtype=np.float32).reshape(-1)
    if array.shape[0] != 52:
        raise ValueError(f"expected Pipeline observation with 52 values, got {array.shape}")
    return array


def _infer_method(checkpoint: dict[str, Any]) -> str:
    return "dar_td3bc" if "behavior" in checkpoint else "td3bc_mlp"


def _checkpoint_seed(checkpoint: dict[str, Any]) -> int:
    try:
        return int(checkpoint.get("seed", -1))
    except (TypeError, ValueError):
        return -1
