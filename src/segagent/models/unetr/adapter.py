"""UNETR adapter implementing the shared model boundary."""

from __future__ import annotations

from typing import Any

from segagent.models.base import ModelAdapter


class UNETRAdapter(ModelAdapter):
    def build(self) -> Any:
        try:
            from monai.networks.nets import UNETR
        except ImportError as error:
            raise RuntimeError("UNETR requires MONAI >= 1.5.1") from error

        architecture = self.model_config["architecture"]
        return UNETR(
            in_channels=self.model_config["in_channels"],
            out_channels=self.model_config["out_channels"],
            img_size=self.require_patch_size(),
            feature_size=architecture["feature_size"],
            hidden_size=architecture["hidden_size"],
            mlp_dim=architecture["mlp_dim"],
            num_heads=architecture["num_heads"],
            proj_type=architecture["proj_type"],
            norm_name=architecture["norm_name"],
            conv_block=architecture["conv_block"],
            res_block=architecture["res_block"],
            dropout_rate=architecture["dropout_rate"],
            spatial_dims=self.model_config["spatial_dims"],
        )


ADAPTER_CLASS = UNETRAdapter
