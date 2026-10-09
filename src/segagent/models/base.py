"""Common interface for model-specific adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelAdapter(ABC):
    """Boundary between canonical samples and a model implementation."""

    @abstractmethod
    def build(self) -> Any:
        """Construct and return the underlying model."""

    @abstractmethod
    def prepare_batch(self, batch: Any) -> Any:
        """Adapt a canonical batch without changing shared preprocessing."""
