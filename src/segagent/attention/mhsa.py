"""Memory-bounded global multi-head self-attention for 2D feature maps."""

from __future__ import annotations

import math

import torch
from torch.nn import functional


class MultiHeadSelfAttention2d(torch.nn.Module):
    """Use full-resolution queries and spatially pooled global keys/values.

    Bounding the key/value token count keeps attention practical at SegFormer's
    highest-resolution encoder stage while every query can still attend over the
    complete feature map.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        max_kv_tokens: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        if max_kv_tokens < 1:
            raise ValueError("max_kv_tokens must be positive")
        self.max_kv_tokens = max_kv_tokens
        self.query_norm = torch.nn.LayerNorm(channels)
        self.key_value_norm = torch.nn.LayerNorm(channels)
        self.attention = torch.nn.MultiheadAttention(
            channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def _key_value_map(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        if height * width <= self.max_kv_tokens:
            return image
        scale = math.sqrt(self.max_kv_tokens / (height * width))
        pooled_height = max(1, int(height * scale))
        pooled_width = max(1, int(width * scale))
        while pooled_height * pooled_width > self.max_kv_tokens:
            if pooled_height >= pooled_width:
                pooled_height -= 1
            else:
                pooled_width -= 1
        return functional.adaptive_avg_pool2d(image, (pooled_height, pooled_width))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = image.shape
        query = image.flatten(2).transpose(1, 2)
        key_value = self._key_value_map(image).flatten(2).transpose(1, 2)
        attended, _ = self.attention(
            self.query_norm(query),
            self.key_value_norm(key_value),
            self.key_value_norm(key_value),
            need_weights=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, channels, height, width)
        return image + attended
