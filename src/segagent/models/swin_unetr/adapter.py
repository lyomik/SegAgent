"""Swin-UNETR adapter implementing the shared model boundary."""

from __future__ import annotations

from typing import Any

from segagent.models.base import ModelAdapter


class SwinUNETRAdapter(ModelAdapter):
    def build(self) -> Any:
        try:
            import torch
            from monai.networks.nets import SwinUNETR
        except ImportError as error:
            raise RuntimeError("Swin-UNETR requires PyTorch and MONAI >= 1.5.1") from error

        architecture = self.model_config["architecture"]
        model = SwinUNETR(
            in_channels=self.model_config["in_channels"],
            out_channels=self.model_config["out_channels"],
            depths=tuple(architecture["depths"]),
            num_heads=tuple(architecture["num_heads"]),
            feature_size=architecture["feature_size"],
            use_checkpoint=architecture["use_checkpoint"],
            use_v2=architecture["use_v2"],
            spatial_dims=self.model_config["spatial_dims"],
        )
        checkpoint = self.require_checkpoint()
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_from(weights=weights)
        return model


ADAPTER_CLASS = SwinUNETRAdapter
