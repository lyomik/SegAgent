"""Convolutional Block Attention Module."""

from __future__ import annotations

import torch


class CBAM2d(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        if spatial_kernel_size % 2 != 1:
            raise ValueError("spatial_kernel_size must be odd")
        hidden = max(1, channels // reduction_ratio)
        self.channel_mlp = torch.nn.Sequential(
            torch.nn.Conv2d(channels, hidden, 1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.spatial = torch.nn.Conv2d(
            2,
            1,
            spatial_kernel_size,
            padding=spatial_kernel_size // 2,
            bias=False,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        average = self.channel_mlp(torch.mean(image, dim=(2, 3), keepdim=True))
        maximum = self.channel_mlp(torch.amax(image, dim=(2, 3), keepdim=True))
        image = image * torch.sigmoid(average + maximum)
        spatial = torch.cat(
            (torch.mean(image, dim=1, keepdim=True), torch.amax(image, dim=1, keepdim=True)),
            dim=1,
        )
        return image * torch.sigmoid(self.spatial(spatial))
