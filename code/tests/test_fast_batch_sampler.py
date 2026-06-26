import numpy as np
import torch

from dar_td3bc.data.fast_batch import FastTransitionBatches
from dar_td3bc.data.pipeline_dataset import PipelineArrays


def _arrays(n: int = 12) -> PipelineArrays:
    obs = np.zeros((n, 52), dtype=np.float32)
    obs[:, 0] = np.arange(n, dtype=np.float32)
    obs[:, 1] = 1.0
    actions = np.linspace(-1.0, 1.0, n, dtype=np.float32).reshape(-1, 1)
    rewards = np.ones((n, 1), dtype=np.float32)
    next_obs = obs.copy()
    dones = np.zeros((n, 1), dtype=bool)
    dones[-1] = True
    return PipelineArrays(
        observations=obs,
        actions=actions,
        rewards=rewards,
        next_observations=next_obs,
        dones=dones,
        timeouts=np.zeros((n, 1), dtype=bool),
    )


def test_fast_transition_batches_samples_batch_on_requested_device():
    batches = FastTransitionBatches(
        _arrays(),
        horizons=(1, 2),
        device=torch.device("cpu"),
    )

    batch = batches.sample(5)

    assert batch.obs.shape == (5, 52)
    assert batch.actions.shape == (5, 1)
    assert batch.future_flows.shape == (5, 2)
    assert batch.obs.device.type == "cpu"


def test_fast_transition_batches_iterates_without_python_item_collation():
    batches = FastTransitionBatches(_arrays(), horizons=(1,), device="cpu")

    first = next(batches.iter_batches(batch_size=4, shuffle=False))

    assert first.obs[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
