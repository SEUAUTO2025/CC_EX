from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from dar_td3bc.data.pipeline_dataset import PipelineArrays
from dar_td3bc.data.trajectory_index import TrajectoryIndex


@dataclass(frozen=True)
class TransitionBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor
    previous_actions: torch.Tensor
    future_flows: torch.Tensor
    future_mask: torch.Tensor


class PipelineTransitionDataset(Dataset[TransitionBatch]):
    def __init__(
        self,
        arrays: PipelineArrays,
        *,
        horizons: tuple[int, ...] = (1, 5, 10, 20),
    ) -> None:
        self.arrays = arrays
        self.horizons = tuple(horizons)
        terminals = arrays.dones.reshape(-1)
        timeouts = arrays.timeouts.reshape(-1)
        future = TrajectoryIndex(
            terminals=terminals,
            timeouts=timeouts,
        ).future_flow_targets(arrays.observations[:, 0], self.horizons)
        prev_actions = arrays.actions.copy()
        prev_actions[1:] = arrays.actions[:-1]
        prev_actions[0] = 0.0

        self.obs = torch.as_tensor(arrays.observations, dtype=torch.float32)
        self.actions = torch.as_tensor(arrays.actions, dtype=torch.float32)
        self.rewards = torch.as_tensor(arrays.rewards, dtype=torch.float32)
        self.next_obs = torch.as_tensor(arrays.next_observations, dtype=torch.float32)
        self.dones = torch.as_tensor(arrays.dones.astype("float32"))
        self.previous_actions = torch.as_tensor(prev_actions, dtype=torch.float32)
        self.future_flows = torch.as_tensor(future.values, dtype=torch.float32)
        self.future_mask = torch.as_tensor(future.mask, dtype=torch.float32)

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, index: int) -> TransitionBatch:
        return TransitionBatch(
            obs=self.obs[index],
            actions=self.actions[index],
            rewards=self.rewards[index],
            next_obs=self.next_obs[index],
            dones=self.dones[index],
            previous_actions=self.previous_actions[index],
            future_flows=self.future_flows[index],
            future_mask=self.future_mask[index],
        )


def collate_transition_batch(items: list[TransitionBatch]) -> TransitionBatch:
    return TransitionBatch(
        obs=torch.stack([item.obs for item in items]),
        actions=torch.stack([item.actions for item in items]),
        rewards=torch.stack([item.rewards for item in items]),
        next_obs=torch.stack([item.next_obs for item in items]),
        dones=torch.stack([item.dones for item in items]),
        previous_actions=torch.stack([item.previous_actions for item in items]),
        future_flows=torch.stack([item.future_flows for item in items]),
        future_mask=torch.stack([item.future_mask for item in items]),
    )
