from __future__ import annotations

from collections.abc import Iterator

import torch

from dar_td3bc.data.pipeline_dataset import PipelineArrays
from dar_td3bc.data.torch_dataset import TransitionBatch
from dar_td3bc.data.trajectory_index import TrajectoryIndex


class FastTransitionBatches:
    """Tensor-backed transition batches without per-item DataLoader collation."""

    def __init__(
        self,
        arrays: PipelineArrays,
        *,
        horizons: tuple[int, ...] = (1, 5, 10, 20),
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        terminals = arrays.dones.reshape(-1)
        timeouts = arrays.timeouts.reshape(-1)
        future = TrajectoryIndex(
            terminals=terminals,
            timeouts=timeouts,
        ).future_flow_targets(arrays.observations[:, 0], horizons)
        prev_actions = arrays.actions.copy()
        prev_actions[1:] = arrays.actions[:-1]
        prev_actions[0] = 0.0

        self.obs = torch.as_tensor(
            arrays.observations, dtype=torch.float32, device=self.device
        )
        self.actions = torch.as_tensor(
            arrays.actions, dtype=torch.float32, device=self.device
        )
        self.rewards = torch.as_tensor(
            arrays.rewards, dtype=torch.float32, device=self.device
        )
        self.next_obs = torch.as_tensor(
            arrays.next_observations, dtype=torch.float32, device=self.device
        )
        self.dones = torch.as_tensor(
            arrays.dones.astype("float32"), dtype=torch.float32, device=self.device
        )
        self.previous_actions = torch.as_tensor(
            prev_actions, dtype=torch.float32, device=self.device
        )
        self.future_flows = torch.as_tensor(
            future.values, dtype=torch.float32, device=self.device
        )
        self.future_mask = torch.as_tensor(
            future.mask, dtype=torch.float32, device=self.device
        )

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def sample(self, batch_size: int) -> TransitionBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        index = torch.randint(
            0,
            len(self),
            (batch_size,),
            device=self.device,
        )
        return self.index_select(index)

    def iter_batches(
        self,
        *,
        batch_size: int,
        shuffle: bool,
    ) -> Iterator[TransitionBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if shuffle:
            order = torch.randperm(len(self), device=self.device)
        else:
            order = torch.arange(len(self), device=self.device)
        for start in range(0, len(self), batch_size):
            yield self.index_select(order[start : start + batch_size])

    def index_select(self, index: torch.Tensor) -> TransitionBatch:
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
