import pytest
import torch

from segagent.attention.cbam import CBAM2d
from segagent.attention.ca import CoordinateAttention2d
from segagent.attention.eca import EfficientChannelAttention2d
from segagent.attention.mhsa import MultiHeadSelfAttention2d
from segagent.attention.se import SqueezeExcitation2d


@pytest.mark.parametrize(
    "module",
    (
        SqueezeExcitation2d(16, reduction_ratio=4),
        CBAM2d(16, reduction_ratio=4, spatial_kernel_size=7),
        EfficientChannelAttention2d(16, kernel_size=3),
        CoordinateAttention2d(16, reduction_ratio=4),
        MultiHeadSelfAttention2d(16, num_heads=4, max_kv_tokens=16),
    ),
)
def test_attention_forward_backward_preserves_shape(module: torch.nn.Module) -> None:
    image = torch.randn(2, 16, 12, 10, requires_grad=True)
    output = module(image)
    assert output.shape == image.shape
    output.mean().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
