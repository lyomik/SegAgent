#!/usr/bin/env python3
"""Export the shared processed dataset to nnU-Net v2 raw-data layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from segagent.data.dataset import load_manifest
from segagent.models.registry import PROJECT_ROOT

DEFAULT_MANIFEST = PROJECT_ROOT / "data/processed/metadata/cases.json"
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data/nnunet/raw"


def ensure_link(source: Path, destination: Path) -> None:
    if destination.is_symlink() and destination.resolve() == source.resolve():
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to replace nnU-Net export path: {destination}")
    destination.symlink_to(source.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", default="LIDC_IDRI")
    args = parser.parse_args()

    cases = load_manifest(args.manifest.resolve())
    dataset_dir = args.raw_root.resolve() / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    for directory in (images_tr, labels_tr, images_ts):
        directory.mkdir(parents=True, exist_ok=True)

    training_cases = []
    validation_cases = []
    test_cases = []
    for case in cases:
        identifier = case.case_id
        if case.split == "test":
            ensure_link(case.image, images_ts / f"{identifier}_0000.nii.gz")
            test_cases.append(identifier)
            continue
        ensure_link(case.image, images_tr / f"{identifier}_0000.nii.gz")
        ensure_link(case.label, labels_tr / f"{identifier}.nii.gz")
        (training_cases if case.split == "train" else validation_cases).append(identifier)

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "nodule": 1},
        "numTraining": len(training_cases) + len(validation_cases),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    (dataset_dir / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8"
    )
    split = [{"train": sorted(training_cases), "val": sorted(validation_cases)}]
    (dataset_dir / "splits_final.json").write_text(
        json.dumps(split, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "dataset_dir": str(dataset_dir),
        "train": len(training_cases),
        "validation": len(validation_cases),
        "test": len(test_cases),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
