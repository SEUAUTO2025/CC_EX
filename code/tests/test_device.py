import torch

from dar_td3bc.data.torch_dataset import TransitionBatch
from dar_td3bc.utils.device import resolve_device


def _batch() -> TransitionBatch:
    return TransitionBatch(
        obs=torch.zeros((2, 52)),
        actions=torch.zeros((2, 1)),
        rewards=torch.zeros((2, 1)),
        next_obs=torch.zeros((2, 52)),
        dones=torch.zeros((2, 1)),
        previous_actions=torch.zeros((2, 1)),
        future_flows=torch.zeros((2, 4)),
        future_mask=torch.ones((2, 4)),
    )


def test_resolve_device_falls_back_to_cpu_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    device = resolve_device({"device": "cuda"})

    assert device.type == "cpu"


def test_transition_batch_to_moves_all_tensors():
    moved = _batch().to(torch.device("cpu"))

    assert all(value.device.type == "cpu" for value in moved.__dict__.values())
