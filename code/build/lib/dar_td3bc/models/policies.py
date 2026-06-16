from __future__ import annotations

import torch
from torch import nn


class BehaviorPolicy(nn.Module):
    def __init__(self, input_dim: int = 52, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ResidualActor(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class MultiHorizonPredictionHead(nn.Module):
    def __init__(
        self, latent_dim: int = 128, horizons: tuple[int, ...] = (1, 5, 10, 20)
    ) -> None:
        super().__init__()
        if not horizons:
            raise ValueError("horizons must not be empty")
        if any(h <= 0 for h in horizons):
            raise ValueError(f"horizons must be positive, got {horizons}")
        self.horizons = tuple(int(h) for h in horizons)
        self.head = nn.Linear(latent_dim, len(self.horizons))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.head(latent)


def compose_residual_action(
    behavior_action: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    *,
    residual_scale: float = 0.25,
) -> torch.Tensor:
    return torch.clamp(
        behavior_action + gate * residual_scale * residual,
        min=-1.0,
        max=1.0,
    )
