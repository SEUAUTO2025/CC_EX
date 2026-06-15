import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dar_td3bc.data.pipeline_dataset import PipelineArrays
from dar_td3bc.data.torch_dataset import (
    PipelineTransitionDataset,
    collate_transition_batch,
)


def _arrays(n: int = 6) -> PipelineArrays:
    obs = np.zeros((n, 52), dtype=np.float32)
    obs[:, 0] = np.arange(n)
    next_obs = obs + 1.0
    return PipelineArrays(
        observations=obs,
        actions=np.arange(n, dtype=np.float32).reshape(-1, 1) / 10.0,
        rewards=np.arange(n, dtype=np.float32).reshape(-1, 1),
        next_observations=next_obs,
        dones=np.array([False] * (n - 1) + [True]).reshape(-1, 1),
        timeouts=np.zeros((n, 1), dtype=bool),
    )


def test_pipeline_transition_dataset_returns_tensors_and_previous_action():
    dataset = PipelineTransitionDataset(_arrays(), horizons=(1, 2))

    first = dataset[0]
    third = dataset[3]

    assert len(dataset) == 6
    assert isinstance(first.obs, torch.Tensor)
    assert first.previous_actions.item() == 0.0
    assert third.previous_actions.item() == pytest.approx(0.2)
    assert first.future_flows.tolist() == [1.0, 2.0]
    assert first.future_mask.tolist() == [1.0, 1.0]


def test_pipeline_transition_dataset_masks_future_across_terminal():
    dataset = PipelineTransitionDataset(_arrays(n=4), horizons=(1, 2))

    item = dataset[2]

    assert item.future_flows.tolist() == [3.0, 0.0]
    assert item.future_mask.tolist() == [1.0, 0.0]


def test_collate_transition_batch_stacks_dataset_items():
    dataset = PipelineTransitionDataset(_arrays(n=5), horizons=(1, 2))

    batch = next(
        iter(DataLoader(dataset, batch_size=3, collate_fn=collate_transition_batch))
    )

    assert batch.obs.shape == (3, 52)
    assert batch.actions.shape == (3, 1)
    assert batch.future_flows.shape == (3, 2)
    assert batch.future_mask.shape == (3, 2)
    assert batch.previous_actions[:, 0].tolist() == pytest.approx([0.0, 0.0, 0.1])
