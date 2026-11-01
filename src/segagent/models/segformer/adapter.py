"""SegFormer adapter implementing the shared model boundary."""

from __future__ import annotations

from typing import Any

from segagent.models.base import ModelAdapter


class SegFormerAdapter(ModelAdapter):
    def build(self) -> Any:
        try:
            import torch
            from torch.nn import functional
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as error:
            raise RuntimeError("SegFormer requires PyTorch and Transformers") from error

        checkpoint = self.require_checkpoint()
        out_channels = self.model_config["out_channels"]
        config = SegformerConfig.from_pretrained(checkpoint, local_files_only=True)
        config.id2label = {0: "background", 1: "nodule"}
        config.label2id = {"background": 0, "nodule": 1}
        config.num_labels = out_channels
        network = SegformerForSemanticSegmentation.from_pretrained(
            checkpoint,
            config=config,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
        attention_config = self.model_config.get("attention")

        class SegFormerLogits(torch.nn.Module):
            def __init__(self, model: Any, configured_attention: Any = None) -> None:
                super().__init__()
                self.model = model
                self.attention_name = None
                if configured_attention and configured_attention.get("enabled", False):
                    from segagent.attention.factory import build_attention_2d

                    self.attention_name = str(configured_attention["name"])
                    self.stage_attention = torch.nn.ModuleList(
                        build_attention_2d(channels, configured_attention)
                        for channels in model.config.hidden_sizes
                    )
                else:
                    self.stage_attention = None

            def forward(self, image: Any) -> Any:
                if image.shape[1] == 1:
                    image = image.repeat(1, 3, 1, 1)
                if self.stage_attention is None:
                    logits = self.model(pixel_values=image).logits
                else:
                    encoder_outputs = self.model.segformer(
                        pixel_values=image,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    hidden_states = tuple(
                        attention(hidden)
                        for attention, hidden in zip(
                            self.stage_attention, encoder_outputs.hidden_states, strict=True
                        )
                    )
                    logits = self.model.decode_head(hidden_states)
                return functional.interpolate(
                    logits,
                    size=image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

        return SegFormerLogits(network, attention_config)


ADAPTER_CLASS = SegFormerAdapter
