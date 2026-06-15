import math

import torch
from torch import nn

from dar_td3bc.algorithms.td3bc import (
    masked_prediction_loss,
    soft_update,
    td3bc_actor_loss,
)


def test_soft_update_moves_target_toward_source():
    source = nn.Linear(2, 1)
    target = nn.Linear(2, 1)
    with torch.no_grad():
        source.weight.fill_(2.0)
        source.bias.fill_(4.0)
        target.weight.fill_(0.0)
        target.bias.fill_(0.0)

    soft_update(source, target, tau=0.25)

    assert torch.allclose(target.weight, torch.full_like(target.weight, 0.5))
    assert torch.allclose(target.bias, torch.full_like(target.bias, 1.0))


def test_masked_prediction_loss_ignores_invalid_horizons():
    prediction = torch.tensor([[1.0, 2.0], [10.0, 20.0]])
    target = torch.tensor([[2.0, 2.0], [30.0, 20.0]])
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    loss = masked_prediction_loss(prediction, target, mask)

    assert math.isclose(loss.item(), 0.5)


def test_td3bc_actor_loss_returns_finite_weighted_components():
    q_pi = torch.tensor([[2.0], [4.0]])
    policy_action = torch.tensor([[0.2], [0.4]])
    dataset_action = torch.tensor([[0.0], [0.5]])
    residual = torch.tensor([[0.2], [-0.1]])
    previous_action = torch.tensor([[0.1], [0.3]])

    components = td3bc_actor_loss(
        q_pi,
        policy_action,
        dataset_action,
        residual,
        previous_action,
        alpha_td3bc=2.5,
        alpha_bc=1.0,
        alpha_res=0.1,
        alpha_tv=0.2,
    )

    assert torch.isfinite(components.total)
    assert components.lambda_q.item() > 0.0
    assert components.bc_term.item() > 0.0
    assert components.residual_term.item() > 0.0
    assert components.tv_term.item() > 0.0
