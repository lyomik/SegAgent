"""MedNeXt adapter implementing the shared model boundary."""

from __future__ import annotations

from typing import Any

from segagent.models.base import ModelAdapter


class MedNeXtAdapter(ModelAdapter):
    def build(self) -> Any:
        try:
            from monai.networks.nets.mednext import create_mednext
        except ImportError as error:
            raise RuntimeError("MedNeXt requires MONAI >= 1.5.1") from error

        architecture = self.model_config["architecture"]
        return create_mednext(
            variant=architecture["variant"],
            spatial_dims=self.model_config["spatial_dims"],
            in_channels=self.model_config["in_channels"],
            out_channels=self.model_config["out_channels"],
            kernel_size=architecture["kernel_size"],
            deep_supervision=architecture["deep_supervision"],
        )


ADAPTER_CLASS = MedNeXtAdapter
