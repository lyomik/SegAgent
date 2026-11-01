"""Configuration-backed registry for the Stage 1 baseline models."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable

import yaml

from segagent.models.base import ModelAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs/models/baselines.yaml"
REQUIRED_MODEL_FIELDS = {
    "name",
    "display_name",
    "family",
    "backend",
    "adapter",
    "spatial_dims",
    "in_channels",
    "out_channels",
    "initialization",
    "architecture",
    "checkpointing",
}


@dataclass(frozen=True)
class BaselineSpec:
    """One model entry resolved from the baseline manifest."""

    name: str
    config_path: Path
    enabled: bool
    config: dict[str, Any]

    @property
    def model(self) -> dict[str, Any]:
        return self.config["model"]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def load_baselines(manifest_path: Path = DEFAULT_MANIFEST) -> list[BaselineSpec]:
    """Load and validate all baseline definitions in manifest order."""
    manifest_path = manifest_path.resolve()
    manifest = _read_yaml(manifest_path)
    entries = manifest.get("baselines")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"No baselines declared in {manifest_path}")

    task = manifest.get("task", {})
    seen: set[str] = set()
    specs: list[BaselineSpec] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each baseline manifest entry must be a mapping")
        name = str(entry["name"])
        if name in seen:
            raise ValueError(f"Duplicate baseline name: {name}")
        seen.add(name)

        configured_path = Path(str(entry["config"]))
        config_path = configured_path if configured_path.is_absolute() else PROJECT_ROOT / configured_path
        config = _read_yaml(config_path)
        model = config.get("model")
        if not isinstance(model, dict):
            raise ValueError(f"Missing model mapping in {config_path}")
        missing = sorted(REQUIRED_MODEL_FIELDS - model.keys())
        if missing:
            raise ValueError(f"Missing fields in {config_path}: {', '.join(missing)}")
        if model["name"] != name:
            raise ValueError(f"Manifest name {name!r} does not match {model['name']!r}")
        for field in ("in_channels", "out_channels"):
            if field in task and model[field] != task[field]:
                raise ValueError(f"{name} {field}={model[field]} does not match task {task[field]}")
        specs.append(
            BaselineSpec(
                name=name,
                config_path=config_path,
                enabled=bool(entry.get("enabled", True)),
                config=config,
            )
        )
    return specs


def get_baseline(name: str, specs: Iterable[BaselineSpec] | None = None) -> BaselineSpec:
    """Resolve a baseline by its stable registry name."""
    for spec in specs or load_baselines():
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown baseline {name!r}")


def create_adapter(spec: BaselineSpec) -> ModelAdapter:
    """Create the configured adapter without importing its heavy backend eagerly."""
    module = import_module(spec.model["adapter"])
    adapter_class = getattr(module, "ADAPTER_CLASS", None)
    if adapter_class is None or not issubclass(adapter_class, ModelAdapter):
        raise TypeError(f"{spec.model['adapter']} must expose an ADAPTER_CLASS")
    return adapter_class(spec.config)


def with_attention(spec: BaselineSpec, attention_path: Path) -> BaselineSpec:
    """Return a baseline spec with one attention configuration attached."""
    attention_payload = _read_yaml(attention_path.resolve())
    attention = attention_payload.get("attention")
    if not isinstance(attention, dict) or not attention.get("enabled", False):
        raise ValueError(f"Enabled attention mapping required in {attention_path}")
    config = deepcopy(spec.config)
    config["model"]["attention"] = deepcopy(attention)
    return BaselineSpec(
        name=spec.name,
        config_path=spec.config_path,
        enabled=spec.enabled,
        config=config,
    )
