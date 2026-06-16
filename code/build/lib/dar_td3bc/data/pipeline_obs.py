from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ParsedPipelineObs:
    """Structured view of a NeoRL-2 Pipeline observation batch."""

    current_flow: torch.Tensor
    target_flow: torch.Tensor
    sequence: torch.Tensor
    flat: torch.Tensor
    flow_history: torch.Tensor
    action_history: torch.Tensor


def split_pipeline_obs(obs: torch.Tensor) -> ParsedPipelineObs:
    """Parse `[B, 52]` Pipeline observations into flow/action history."""
    if not isinstance(obs, torch.Tensor):
        obs = torch.as_tensor(obs)
    if obs.ndim != 2 or obs.shape[-1] != 52:
        raise ValueError(f"Expected [B, 52], got {tuple(obs.shape)}")

    current_flow = obs[:, 0:1]
    target_flow = obs[:, 1:2]
    flow_history_recent_first = obs[:, 2:27]
    action_history_recent_first = obs[:, 27:52]

    flow_history = torch.flip(flow_history_recent_first, dims=[1])
    action_history = torch.flip(action_history_recent_first, dims=[1])
    sequence = torch.stack((flow_history, action_history), dim=-1)

    return ParsedPipelineObs(
        current_flow=current_flow,
        target_flow=target_flow,
        sequence=sequence,
        flat=obs,
        flow_history=flow_history,
        action_history=action_history,
    )
