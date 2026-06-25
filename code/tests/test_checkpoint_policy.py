import torch

from dar_td3bc.evaluation.policy import CheckpointPolicy, predict_checkpoint_actions
from dar_td3bc.models.critic import TwinCritic
from dar_td3bc.models.policies import BehaviorPolicy, ResidualActor
from dar_td3bc.models.temporal_encoder import DelayEncoder


def _small_dar_checkpoint() -> dict:
    config = {
        "model": {
            "hidden_dim": 4,
            "latent_dim": 3,
            "tcn_dilations": [1],
            "residual_scale": 1.0,
            "gate_min": 0.05,
            "gate_kappa": 1.0,
        }
    }
    behavior = BehaviorPolicy(input_dim=52, hidden_dim=4)
    encoder = DelayEncoder(hidden_dim=4, latent_dim=3, dilations=(1,))
    actor = ResidualActor(latent_dim=3, hidden_dim=4)
    critic = TwinCritic(latent_dim=3, hidden_dim=4)

    for model in (behavior, encoder, actor, critic):
        for param in model.parameters():
            param.data.zero_()

    actor.net[-2].bias.data.fill_(1.0)
    critic.q1[-1].bias.data.fill_(0.0)
    critic.q2[-1].bias.data.fill_(2.0)

    return {
        "config": config,
        "behavior": behavior.state_dict(),
        "encoder": encoder.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
    }


def test_dar_checkpoint_prediction_uses_critic_disagreement_gate():
    checkpoint = _small_dar_checkpoint()
    observations = torch.zeros((2, 52), dtype=torch.float32)

    actions, aux = predict_checkpoint_actions(
        checkpoint,
        observations,
        return_aux=True,
    )

    ungated_residual = torch.tanh(torch.ones((2, 1)))
    expected_gate = torch.exp(torch.full((2, 1), -2.0))
    assert torch.allclose(aux["gate"], expected_gate)
    assert torch.allclose(actions, expected_gate * ungated_residual)


def test_checkpoint_policy_act_matches_functional_prediction():
    checkpoint = _small_dar_checkpoint()
    observations = torch.zeros((2, 52), dtype=torch.float32)

    policy = CheckpointPolicy(checkpoint, device="cpu")
    actions = policy.act(observations)
    expected = predict_checkpoint_actions(checkpoint, observations, device="cpu")

    assert torch.allclose(actions, expected)
