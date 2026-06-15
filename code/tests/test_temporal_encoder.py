import torch

from dar_td3bc.models.temporal_encoder import CausalConv1d, DelayEncoder


def test_causal_conv_preserves_sequence_length():
    conv = CausalConv1d(2, 4, kernel_size=3, dilation=2)
    x = torch.randn(3, 2, 25)

    y = conv(x)

    assert y.shape == (3, 4, 25)


def test_delay_encoder_returns_latent_vector():
    encoder = DelayEncoder(hidden_dim=16, latent_dim=12, dilations=(1, 2))
    sequence = torch.randn(5, 25, 2)
    current_flow = torch.randn(5, 1)
    target_flow = torch.randn(5, 1)

    z = encoder(sequence, current_flow, target_flow)

    assert z.shape == (5, 12)
    assert torch.isfinite(z).all()
