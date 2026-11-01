"""nnU-Net v2 adapter for its externally managed training pipeline."""

from __future__ import annotations

from typing import Any

from segagent.models.base import ModelAdapter


class NNUNetAdapter(ModelAdapter):
    external_pipeline = True

    def build(self) -> Any:
        raise RuntimeError(
            "nnU-Net v2 is built by nnUNetv2_train after fingerprint extraction and planning; "
            "use training_command() instead of the shared in-process trainer"
        )

    def build_smoke(self) -> Any:
        """Build a compact nnU-Net architecture before dataset-specific planning exists."""
        try:
            import torch
            from dynamic_network_architectures.architectures.unet import PlainConvUNet
        except ImportError as error:
            raise RuntimeError("nnU-Net smoke testing requires PyTorch and nnU-Net v2") from error

        spatial_dims = int(self.model_config["spatial_dims"])
        if spatial_dims != 3:
            raise ValueError("The configured nnU-Net smoke architecture expects 3D input")
        return PlainConvUNet(
            input_channels=self.model_config["in_channels"],
            n_stages=4,
            features_per_stage=(32, 64, 128, 256),
            conv_op=torch.nn.Conv3d,
            kernel_sizes=((3, 3, 3),) * 4,
            strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
            n_conv_per_stage=(2, 2, 2, 2),
            num_classes=self.model_config["out_channels"],
            n_conv_per_stage_decoder=(2, 2, 2),
            conv_bias=True,
            norm_op=torch.nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=False,
        )

    def training_command(self, fold: int = 0, continue_training: bool = False) -> list[str]:
        architecture = self.model_config["architecture"]
        dataset_id = architecture["dataset_id"]
        if dataset_id is None:
            raise ValueError("Set model.architecture.dataset_id after exporting the canonical dataset")
        if fold not in architecture["folds"]:
            raise ValueError(f"Fold {fold} is not enabled for nnU-Net")
        command = [
            "nnUNetv2_train",
            str(dataset_id),
            architecture["configuration"],
            str(fold),
            "-tr",
            architecture["trainer"],
            "-p",
            architecture["plans"],
        ]
        if continue_training:
            command.append("--c")
        return command


ADAPTER_CLASS = NNUNetAdapter
