import torch

from dar_td3bc.models.critic import TwinCritic, critic_disagreement_gate
from dar_td3bc.models.policies import (
    BehaviorPolicy,
    MultiHorizonPredictionHead,
    ResidualActor,
    compose_residual_action,
)


def test_behavior_policy_outputs_bounded_action():
    policy = BehaviorPolicy(input_dim=52, hidden_dim=32)
    obs = torch.randn(7, 52)

    action = policy(obs)

    assert action.shape == (7, 1)
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)


def test_residual_actor_outputs_bounded_residual():
    actor = ResidualActor(latent_dim=12, hidden_dim=32)
    latent = torch.randn(7, 12)

    residual = actor(latent)

    assert residual.shape == (7, 1)
    assert torch.all(residual <= 1.0)
    assert torch.all(residual >= -1.0)


def test_compose_residual_action_clips_to_action_bounds():
    behavior = torch.tensor([[0.95], [-0.95], [0.1]])
    residual = torch.tensor([[1.0], [-1.0], [0.2]])
    gate = torch.tensor([[1.0], [1.0], [0.5]])

    action = compose_residual_action(
        behavior, residual, gate, residual_scale=0.25
    )

    assert action.tolist() == [[1.0], [-1.0], [0.125]]


def test_twin_critic_returns_two_q_values():
    critic = TwinCritic(latent_dim=12, hidden_dim=32)
    latent = torch.randn(7, 12)
    action = torch.randn(7, 1).clamp(-1.0, 1.0)

    q1, q2 = critic(latent, action)

    assert q1.shape == (7, 1)
    assert q2.shape == (7, 1)
    assert torch.isfinite(q1).all()
    assert torch.isfinite(q2).all()


def test_critic_disagreement_gate_is_detached_and_bounded():
    q1 = torch.tensor([[1.0], [2.0], [4.0]], requires_grad=True)
    q2 = torch.tensor([[1.0], [3.0], [1.0]], requires_grad=True)

    gate, disagreement = critic_disagreement_gate(
        q1, q2, kappa=1.0, normalizer=2.0, gate_min=0.1
    )

    assert disagreement.tolist() == [[0.0], [1.0], [3.0]]
    assert gate.requires_grad is False
    assert torch.all(gate <= 1.0)
    assert torch.all(gate >= 0.1)
    assert gate[0].item() == 1.0


def test_prediction_head_outputs_one_value_per_horizon():
    head = MultiHorizonPredictionHead(latent_dim=12, horizons=(1, 5, 10))
    latent = torch.randn(7, 12)

    pred = head(latent)

    assert pred.shape == (7, 3)
    assert head.horizons == (1, 5, 10)
