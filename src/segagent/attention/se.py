"""Squeeze-and-Excitation attention module."""

from __future__ import annotations

import torch


class SqueezeExcitation2d(torch.nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction_ratio)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.excitation = torch.nn.Sequential(
            torch.nn.Conv2d(channels, hidden, 1, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, channels, 1, bias=True),
            torch.nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image * self.excitation(self.pool(image))
