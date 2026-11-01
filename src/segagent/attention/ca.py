"""Coordinate Attention for two-dimensional feature maps."""

from __future__ import annotations

import torch


class CoordinateAttention2d(torch.nn.Module):
    """Encode channel attention while retaining each spatial coordinate axis."""

    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 32,
        minimum_channels: int = 8,
    ) -> None:
        super().__init__()
        hidden = max(minimum_channels, channels // reduction_ratio)
        self.shared = torch.nn.Sequential(
            torch.nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(hidden),
            torch.nn.Hardswish(),
        )
        self.height_projection = torch.nn.Conv2d(hidden, channels, kernel_size=1)
        self.width_projection = torch.nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        height_context = image.mean(dim=3, keepdim=True)
        width_context = image.mean(dim=2, keepdim=True).transpose(2, 3)
        context = self.shared(torch.cat((height_context, width_context), dim=2))
        height_context, width_context = torch.split(context, (height, width), dim=2)
        width_context = width_context.transpose(2, 3)
        height_weights = torch.sigmoid(self.height_projection(height_context))
        width_weights = torch.sigmoid(self.width_projection(width_context))
        return image * height_weights * width_weights
