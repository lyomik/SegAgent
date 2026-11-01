"""Common interface for model-specific adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ModelAdapter(ABC):
    """Boundary between canonical samples and a model implementation."""

    external_pipeline = False

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.model_config = dict(self.config["model"])

    @abstractmethod
    def build(self) -> Any:
        """Construct and return the underlying model."""

    def build_smoke(self) -> Any:
        """Construct the network used by the one-step runtime smoke test."""
        return self.build()

    def prepare_batch(self, batch: Any) -> Any:
        """Validate the common batch boundary without changing preprocessing."""
        if not isinstance(batch, Mapping) or "image" not in batch:
            raise TypeError("A model batch must be a mapping containing an 'image' tensor.")
        return batch

    def require_checkpoint(self) -> Path:
        """Return the configured initialization checkpoint after validating it."""
        value = self.model_config.get("initialization", {}).get("checkpoint")
        if not value:
            raise ValueError(f"{self.model_config['name']} does not configure a checkpoint")
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {self.model_config['name']}: {path}")
        return path

    def require_patch_size(self) -> tuple[int, ...]:
        """Return the frozen patch size or fail before constructing the model."""
        value = self.model_config.get("patch_size")
        spatial_dims = int(self.model_config["spatial_dims"])
        if value is None:
            raise ValueError(
                f"{self.model_config['name']} patch_size is not frozen yet; "
                "set it after the dataset spacing and size audit"
            )
        if not isinstance(value, (list, tuple)) or len(value) != spatial_dims:
            raise ValueError(f"patch_size must contain {spatial_dims} integers")
        return tuple(int(item) for item in value)
