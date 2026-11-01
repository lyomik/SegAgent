"""Efficient Channel Attention module."""

from __future__ import annotations

import torch


class EfficientChannelAttention2d(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv = torch.nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        weights = self.pool(image).flatten(2).transpose(1, 2)
        weights = torch.sigmoid(self.conv(weights)).transpose(1, 2).unsqueeze(-1)
        return image * weights
