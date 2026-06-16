from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ActorLossComponents:
    total: torch.Tensor
    q_term: torch.Tensor
    bc_term: torch.Tensor
    residual_term: torch.Tensor
    tv_term: torch.Tensor
    lambda_q: torch.Tensor


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be in [0, 1], got {tau}")
    with torch.no_grad():
        for source_param, target_param in zip(
            source.parameters(), target.parameters()
        ):
            target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)


def masked_prediction_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError(
            "prediction, target, and mask must have matching shapes, got "
            f"{tuple(prediction.shape)}, {tuple(target.shape)}, "
            f"{tuple(mask.shape)}"
        )
    sq_error = (prediction - target).pow(2) * mask
    return sq_error.sum() / mask.sum().clamp_min(1.0)


def td3bc_actor_loss(
    q_pi: torch.Tensor,
    policy_action: torch.Tensor,
    dataset_action: torch.Tensor,
    residual: torch.Tensor,
    previous_dataset_action: torch.Tensor,
    *,
    alpha_td3bc: float = 2.5,
    alpha_bc: float = 1.0,
    alpha_res: float = 0.01,
    alpha_tv: float = 0.01,
) -> ActorLossComponents:
    lambda_q = alpha_td3bc / q_pi.abs().mean().detach().clamp_min(1e-6)
    q_term = -lambda_q * q_pi.mean()
    bc_term = alpha_bc * F.mse_loss(policy_action, dataset_action)
    residual_term = alpha_res * residual.pow(2).mean()
    tv_term = alpha_tv * F.mse_loss(policy_action, previous_dataset_action)
    total = q_term + bc_term + residual_term + tv_term
    return ActorLossComponents(
        total=total,
        q_term=q_term,
        bc_term=bc_term,
        residual_term=residual_term,
        tv_term=tv_term,
        lambda_q=lambda_q,
    )
