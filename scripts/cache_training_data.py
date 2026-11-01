#!/usr/bin/env python3
"""Create an uncompressed NIfTI cache for random patch training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from segagent.models.registry import PROJECT_ROOT


DEFAULT_SOURCE = PROJECT_ROOT / "data/processed/metadata/cases.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/training_cache"


def valid_cache(source: Path, cached: Path) -> bool:
    if not cached.exists() or cached.stat().st_size == 0:
        return False
    try:
        source_image = nib.load(str(source))
        cached_image = nib.load(str(cached))
    except (OSError, ValueError):
        return False
    return (
        source_image.shape == cached_image.shape
        and source_image.get_data_dtype() == cached_image.get_data_dtype()
        and np.allclose(source_image.affine, cached_image.affine, atol=1e-6)
    )


def cache_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.nii")
    if temporary.exists():
        temporary.unlink()
    nib.save(nib.load(str(source)), str(temporary))
    temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    source_manifest = args.source.resolve()
    source_root = source_manifest.parent.parent
    output_root = args.output.resolve()
    rows = json.loads(source_manifest.read_text(encoding="utf-8"))["cases"]
    if args.limit is not None:
        rows = rows[: args.limit]
    cached_rows = []
    for index, row in enumerate(rows, start=1):
        print(f"Cache {index}/{len(rows)}: {row['case_id']}", flush=True)
        cached_row = dict(row)
        for field, directory in (("image", "images"), ("label", "labels")):
            source = source_root / row[field]
            destination = output_root / directory / f"{row['case_id']}.nii"
            if not valid_cache(source, destination):
                cache_image(source, destination)
            cached_row[field] = str(destination.relative_to(output_root))
        cached_row["instances"] = str(
            (source_root / row["instances"]).relative_to(output_root, walk_up=True)
        )
        cached_row["metadata"] = str(
            (source_root / row["metadata"]).relative_to(output_root, walk_up=True)
        )
        cached_rows.append(cached_row)

    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    temporary_manifest = metadata_dir / ".cases.json.tmp"
    temporary_manifest.write_text(
        json.dumps({"cases": cached_rows}, indent=2) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(metadata_dir / "cases.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
