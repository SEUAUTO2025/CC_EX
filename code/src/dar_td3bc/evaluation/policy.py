from __future__ import annotations

from typing import Any

import torch

from dar_td3bc.data.pipeline_obs import split_pipeline_obs
from dar_td3bc.models.critic import TwinCritic, critic_disagreement_gate
from dar_td3bc.models.policies import (
    BehaviorPolicy,
    ResidualActor,
    compose_residual_action,
)
from dar_td3bc.models.temporal_encoder import DelayEncoder
from dar_td3bc.utils.device import resolve_device


class CheckpointPolicy:
    def __init__(
        self,
        checkpoint: dict[str, Any],
        *,
        device: torch.device | str | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = (
            torch.device(device)
            if device is not None
            else resolve_device(checkpoint.get("config", {}))
        )
        self.is_dar = not ("actor" in checkpoint and "behavior" not in checkpoint)
        if self.is_dar:
            self._init_dar(checkpoint)
        else:
            self._init_td3bc(checkpoint)

    def act(
        self,
        observations: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        observations = observations.to(self.device)
        if self.is_dar:
            action, aux = self._act_dar(observations)
        else:
            action, aux = self._act_td3bc(observations)
        if return_aux:
            return action, aux
        return action

    def _init_td3bc(self, checkpoint: dict[str, Any]) -> None:
        self.actor = _make_behavior_policy(checkpoint).to(self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor.eval()

    def _init_dar(self, checkpoint: dict[str, Any]) -> None:
        self.behavior = _make_behavior_policy(checkpoint).to(self.device)
        self.encoder = _make_encoder(checkpoint).to(self.device)
        self.actor = _make_residual_actor(checkpoint).to(self.device)
        self.critic = _make_critic(checkpoint).to(self.device)
        self.behavior.load_state_dict(checkpoint["behavior"])
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        for model in (self.behavior, self.encoder, self.actor, self.critic):
            model.eval()

        model_cfg = checkpoint.get("config", {}).get("model", {})
        self.residual_scale = float(model_cfg.get("residual_scale", 0.25))
        self.gate_kappa = float(model_cfg.get("gate_kappa", 1.0))
        self.gate_min = float(model_cfg.get("gate_min", 0.05))
        self.gate_normalizer = float(model_cfg.get("gate_normalizer", 1.0))

    def _act_td3bc(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            action = self.actor(observations).clamp(-1.0, 1.0)
        return action, {}

    def _act_dar(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            parsed = split_pipeline_obs(observations)
            latent = self.encoder(
                parsed.sequence, parsed.current_flow, parsed.target_flow
            )
            behavior_action = self.behavior(observations)
            residual = self.actor(latent)
            candidate_action = compose_residual_action(
                behavior_action,
                residual,
                torch.ones_like(behavior_action),
                residual_scale=self.residual_scale,
            )
            q1, q2 = self.critic(latent, candidate_action)
            gate, disagreement = critic_disagreement_gate(
                q1,
                q2,
                kappa=self.gate_kappa,
                normalizer=self.gate_normalizer,
                gate_min=self.gate_min,
            )
            action = compose_residual_action(
                behavior_action,
                residual,
                gate,
                residual_scale=self.residual_scale,
            )
        return action, {
            "behavior_action": behavior_action,
            "residual": residual,
            "candidate_action": candidate_action,
            "gate": gate,
            "disagreement": disagreement,
        }


def predict_checkpoint_actions(
    checkpoint: dict[str, Any],
    observations: torch.Tensor,
    *,
    device: torch.device | str | None = None,
    return_aux: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Predict deterministic policy actions from a project checkpoint."""
    resolved_device = (
        torch.device(device)
        if device is not None
        else resolve_device(checkpoint.get("config", {}))
    )
    policy = CheckpointPolicy(checkpoint, device=resolved_device)
    return policy.act(observations, return_aux=return_aux)


def _make_behavior_policy(checkpoint: dict[str, Any]) -> BehaviorPolicy:
    hidden_dim = int(checkpoint.get("config", {}).get("model", {}).get("hidden_dim", 256))
    return BehaviorPolicy(input_dim=52, hidden_dim=hidden_dim)


def _make_encoder(checkpoint: dict[str, Any]) -> DelayEncoder:
    model_cfg = checkpoint.get("config", {}).get("model", {})
    return DelayEncoder(
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        latent_dim=int(model_cfg.get("latent_dim", 128)),
        dilations=tuple(int(v) for v in model_cfg.get("tcn_dilations", [1, 2, 4, 8])),
    )


def _make_residual_actor(checkpoint: dict[str, Any]) -> ResidualActor:
    model_cfg = checkpoint.get("config", {}).get("model", {})
    return ResidualActor(
        latent_dim=int(model_cfg.get("latent_dim", 128)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
    )


def _make_critic(checkpoint: dict[str, Any]) -> TwinCritic:
    model_cfg = checkpoint.get("config", {}).get("model", {})
    return TwinCritic(
        latent_dim=int(model_cfg.get("latent_dim", 128)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
    )
