from __future__ import annotations

import torch
from torch import nn


class TwinCritic(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.q1 = self._make_q(latent_dim, hidden_dim)
        self.q2 = self._make_q(latent_dim, hidden_dim)

    @staticmethod
    def _make_q(latent_dim: int, hidden_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat((latent, action), dim=-1)
        return self.q1(x), self.q2(x)


def critic_disagreement_gate(
    q1: torch.Tensor,
    q2: torch.Tensor,
    *,
    kappa: float = 1.0,
    normalizer: float | torch.Tensor = 1.0,
    gate_min: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 <= gate_min <= 1.0:
        raise ValueError(f"gate_min must be in [0, 1], got {gate_min}")
    disagreement = (q1 - q2).abs().detach()
    if isinstance(normalizer, torch.Tensor):
        denom = normalizer.detach().to(disagreement.device, disagreement.dtype)
    else:
        denom = torch.as_tensor(
            normalizer, device=disagreement.device, dtype=disagreement.dtype
        )
    denom = denom.clamp_min(torch.finfo(disagreement.dtype).eps)
    normalized = disagreement / denom
    gate = torch.exp(-kappa * normalized).clamp(gate_min, 1.0).detach()
    return gate, disagreement
