from __future__ import annotations

import torch
from torch import nn


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        self.left_padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.left_padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self.left_padding == 0:
            return y
        return y[..., : -self.left_padding]


class CausalConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.act(self.conv1(x))
        y = self.act(self.conv2(y))
        y = y + residual
        return self.norm(y.transpose(1, 2)).transpose(1, 2)


class DelayEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(2, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [CausalConvBlock(hidden_dim, dilation=d) for d in dilations]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        current_flow: torch.Tensor,
        target_flow: torch.Tensor,
    ) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[-1] != 2:
            raise ValueError(
                "sequence must have shape [B, T, 2], got "
                f"{tuple(sequence.shape)}"
            )
        x = self.input_proj(sequence.transpose(1, 2))
        for block in self.blocks:
            x = block(x)
        h = x[:, :, -1]
        return self.head(torch.cat((h, current_flow, target_flow), dim=-1))
