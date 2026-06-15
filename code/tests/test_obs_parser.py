import pytest
import torch

from dar_td3bc.data.pipeline_obs import split_pipeline_obs


def test_split_pipeline_obs_returns_oldest_to_newest_sequence():
    obs = torch.zeros(1, 52)
    obs[0, 0] = 123.0
    obs[0, 1] = 80.0
    obs[0, 2:27] = torch.arange(100, 125, dtype=torch.float32)
    obs[0, 27:52] = torch.arange(200, 225, dtype=torch.float32)

    parsed = split_pipeline_obs(obs)

    assert parsed.current_flow.tolist() == [[123.0]]
    assert parsed.target_flow.tolist() == [[80.0]]
    assert parsed.flat.data_ptr() == obs.data_ptr()
    assert parsed.sequence.shape == (1, 25, 2)
    assert parsed.sequence[0, :, 0].tolist() == list(
        reversed([float(v) for v in range(100, 125)])
    )
    assert parsed.sequence[0, :, 1].tolist() == list(
        reversed([float(v) for v in range(200, 225)])
    )


def test_split_pipeline_obs_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"Expected \[B, 52\]"):
        split_pipeline_obs(torch.zeros(52))

    with pytest.raises(ValueError, match=r"Expected \[B, 52\]"):
        split_pipeline_obs(torch.zeros(2, 51))
