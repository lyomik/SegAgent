"""Canonical dataset records shared by preprocessing and model adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CanonicalCase:
    case_id: str
    patient_id: str
    series_uid: str
    split: str
    image: Path
    label: Path
    instances: Path | None
    metadata: Path


def load_manifest(path: Path) -> list[CanonicalCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent.parent
    return [
        CanonicalCase(
            case_id=item["case_id"],
            patient_id=item["patient_id"],
            series_uid=item["series_uid"],
            split=item["split"],
            image=root / item["image"],
            label=root / item["label"],
            instances=root / item["instances"] if item.get("instances") else None,
            metadata=root / item["metadata"],
        )
        for item in payload["cases"]
    ]
