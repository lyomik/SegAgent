"""Build configured 2D attention modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from segagent.attention.cbam import CBAM2d
from segagent.attention.ca import CoordinateAttention2d
from segagent.attention.eca import EfficientChannelAttention2d
from segagent.attention.mhsa import MultiHeadSelfAttention2d
from segagent.attention.se import SqueezeExcitation2d


def build_attention_2d(channels: int, config: Mapping[str, Any]) -> torch.nn.Module:
    name = str(config["name"]).lower()
    if name == "se":
        return SqueezeExcitation2d(channels, int(config.get("reduction_ratio", 16)))
    if name == "cbam":
        return CBAM2d(
            channels,
            int(config.get("reduction_ratio", 16)),
            int(config.get("spatial_kernel_size", 7)),
        )
    if name == "eca":
        return EfficientChannelAttention2d(channels, int(config.get("kernel_size", 3)))
    if name == "ca":
        return CoordinateAttention2d(
            channels,
            int(config.get("reduction_ratio", 32)),
            int(config.get("minimum_channels", 8)),
        )
    if name == "mhsa":
        return MultiHeadSelfAttention2d(
            channels,
            int(config.get("num_heads", 4)),
            int(config.get("max_kv_tokens", 256)),
            float(config.get("dropout", 0.0)),
        )
    raise ValueError(f"Unsupported attention module: {name}")
