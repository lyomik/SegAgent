#!/usr/bin/env python3
"""Apply the frozen shared preprocessing configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import SimpleITK as sitk
import yaml

from segagent.data.dataset import load_manifest
from segagent.data.preprocessing import preprocess_case, same_geometry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/dataset/lung_nodule.yaml"


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_geometry(path: Path) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return reader.GetSize(), reader.GetSpacing(), reader.GetOrigin(), reader.GetDirection()


def same_geometry_info(left: tuple, right: tuple, tolerance: float = 1e-6) -> bool:
    return left[0] == right[0] and all(
        abs(left_value - right_value) <= tolerance
        for left_values, right_values in zip(left[1:], right[1:], strict=True)
        for left_value, right_value in zip(left_values, right_values, strict=True)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete, geometry-consistent outputs and process only missing cases",
    )
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    canonical_dir = (PROJECT_ROOT / config["dataset"]["canonical_dir"]).resolve()
    processed_dir = (PROJECT_ROOT / config["dataset"]["processed_dir"]).resolve()
    cases = load_manifest(canonical_dir / "metadata/cases.json")
    if args.split:
        cases = [case for case in cases if case.split == args.split]
    if args.limit is not None:
        cases = cases[: args.limit]

    settings = config["preprocessing"]
    target_spacing = tuple(float(value) for value in settings["target_spacing_mm"])
    hu_window = tuple(float(value) for value in settings["hu_clip"])
    output_range = tuple(float(value) for value in settings["normalization"]["output_range"])
    manifest_rows = []
    for index, case in enumerate(cases, start=1):
        print(f"Preprocess {index}/{len(cases)}: {case.case_id}", flush=True)
        image_output = processed_dir / "images" / f"{case.case_id}.nii.gz"
        label_output = processed_dir / "labels" / f"{case.case_id}.nii.gz"
        instance_output = processed_dir / "instances" / f"{case.case_id}.nii.gz"
        metadata_output = processed_dir / "metadata/cases" / f"{case.case_id}.json"
        outputs = (image_output, label_output, instance_output, metadata_output)
        existing_outputs = tuple(path.exists() for path in outputs)
        if args.resume and all(existing_outputs):
            image_geometry = read_geometry(image_output)
            label_geometry = read_geometry(label_output)
            instance_geometry = read_geometry(instance_output)
            if not same_geometry_info(image_geometry, label_geometry) or not same_geometry_info(
                image_geometry, instance_geometry
            ):
                raise ValueError(f"Existing processed geometry mismatch for {case.case_id}")
            manifest_rows.append(
                {
                    "case_id": case.case_id,
                    "patient_id": case.patient_id,
                    "series_uid": case.series_uid,
                    "split": case.split,
                    "image": str(image_output.relative_to(processed_dir)),
                    "label": str(label_output.relative_to(processed_dir)),
                    "instances": str(instance_output.relative_to(processed_dir)),
                    "metadata": str(metadata_output.relative_to(processed_dir)),
                }
            )
            continue
        if args.resume and any(existing_outputs):
            missing = [str(path) for path, exists in zip(outputs, existing_outputs) if not exists]
            raise FileExistsError(
                f"Partial outputs for {case.case_id}; remove or repair before resuming. "
                f"Missing: {', '.join(missing)}"
            )
        if any(existing_outputs) and not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite or --resume: {image_output}")
        for path in (image_output, label_output, instance_output, metadata_output):
            path.parent.mkdir(parents=True, exist_ok=True)

        image = sitk.ReadImage(str(case.image))
        label = sitk.ReadImage(str(case.label))
        instance = sitk.ReadImage(str(case.instances)) if case.instances else None
        input_geometry = {
            "size_xyz": list(image.GetSize()),
            "spacing_xyz_mm": list(image.GetSpacing()),
            "origin_xyz_mm": list(image.GetOrigin()),
            "direction": list(image.GetDirection()),
        }
        image, label, instance = preprocess_case(
            image=image,
            label=label,
            instance=instance,
            orientation=settings["orientation"],
            target_spacing=target_spacing,
            image_interpolation=settings["image_interpolation"],
            label_interpolation=settings["label_interpolation"],
            hu_window=hu_window,
            output_range=output_range,
            image_padding_hu=float(settings["padding"]["image_value_hu"]),
        )
        if not same_geometry(image, label) or (
            instance is not None and not same_geometry(image, instance)
        ):
            raise ValueError(f"Processed geometry mismatch for {case.case_id}")
        sitk.WriteImage(image, str(image_output), True)
        sitk.WriteImage(label, str(label_output), True)
        if instance is not None:
            sitk.WriteImage(instance, str(instance_output), True)
        metadata = {
            "case_id": case.case_id,
            "source_metadata": str(case.metadata),
            "orientation": settings["orientation"],
            "size_xyz": list(image.GetSize()),
            "spacing_xyz_mm": list(image.GetSpacing()),
            "origin_xyz_mm": list(image.GetOrigin()),
            "direction": list(image.GetDirection()),
            "hu_clip": list(hu_window),
            "normalization_range": list(output_range),
            "transform_chain": [
                {"operation": "orient", "target": settings["orientation"]},
                {
                    "operation": "resample",
                    "target_spacing_xyz_mm": list(target_spacing),
                    "image_interpolation": settings["image_interpolation"],
                    "label_interpolation": settings["label_interpolation"],
                },
                {
                    "operation": "clip_and_scale",
                    "hu_clip": list(hu_window),
                    "output_range": list(output_range),
                },
            ],
            "input_geometry": input_geometry,
        }
        metadata_output.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_rows.append(
            {
                "case_id": case.case_id,
                "patient_id": case.patient_id,
                "series_uid": case.series_uid,
                "split": case.split,
                "image": str(image_output.relative_to(processed_dir)),
                "label": str(label_output.relative_to(processed_dir)),
                "instances": str(instance_output.relative_to(processed_dir)),
                "metadata": str(metadata_output.relative_to(processed_dir)),
            }
        )
    manifest_path = processed_dir / "metadata/cases.json"
    existing = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8")).get("cases", [])
    merged = {row["case_id"]: row for row in existing}
    merged.update({row["case_id"]: row for row in manifest_rows})
    manifest_path.write_text(
        json.dumps({"cases": sorted(merged.values(), key=lambda row: row["case_id"])}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
