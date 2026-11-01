"""Case-level evaluation, aggregation, and report generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from segagent.evaluation.metrics import binary_segmentation_metrics
from segagent.data.preprocessing import same_geometry


METRIC_NAMES = ("dice", "sensitivity", "hd95_mm", "precision", "specificity")


def evaluate_case(
    prediction_path: Path,
    target_path: Path,
    threshold: float = 0.5,
) -> dict[str, float | int | str]:
    prediction_image = sitk.ReadImage(str(prediction_path))
    target_image = sitk.ReadImage(str(target_path))
    if not same_geometry(prediction_image, target_image):
        raise ValueError(f"geometry mismatch: {prediction_path} vs {target_path}")
    prediction = sitk.GetArrayFromImage(prediction_image) >= threshold
    target = sitk.GetArrayFromImage(target_image) > 0
    spacing_zyx = tuple(reversed(target_image.GetSpacing()))
    result = binary_segmentation_metrics(prediction, target, spacing_zyx).to_dict()
    case_id = target_path.name.removesuffix(".nii.gz").removesuffix(".nii")
    voxel_volume_mm3 = float(np.prod(target_image.GetSpacing()))
    return {
        "case_id": case_id,
        "target_voxels": int(np.count_nonzero(target)),
        "target_volume_mm3": float(np.count_nonzero(target) * voxel_volume_mm3),
        **result,
    }


def assign_lesion_size_groups(
    rows: list[dict[str, float | int | str]],
) -> dict[str, float | None]:
    """Assign empty/small/medium/large groups using non-empty cohort tertiles."""
    positive = np.asarray(
        [float(row["target_volume_mm3"]) for row in rows if float(row["target_volume_mm3"]) > 0],
        dtype=np.float64,
    )
    if positive.size:
        small_max, medium_max = (float(value) for value in np.quantile(positive, [1 / 3, 2 / 3]))
    else:
        small_max = medium_max = None
    for row in rows:
        volume = float(row["target_volume_mm3"])
        if volume == 0:
            group = "empty"
        elif small_max is not None and volume <= small_max:
            group = "small"
        elif medium_max is not None and volume <= medium_max:
            group = "medium"
        else:
            group = "large"
        row["lesion_size_group"] = group
    return {"small_max_mm3": small_max, "medium_max_mm3": medium_max}


def aggregate_metrics(rows: list[dict[str, float | int | str]]) -> dict[str, object]:
    aggregate: dict[str, object] = {"case_count": len(rows)}
    for name in METRIC_NAMES:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        aggregate[name] = {
            "mean": float(finite.mean()) if finite.size else None,
            "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            "median": float(np.median(finite)) if finite.size else None,
            "infinite_count": int(np.count_nonzero(~np.isfinite(values))),
        }
    return aggregate


def write_results(
    output_dir: Path,
    rows: list[dict[str, float | int | str]],
    metadata: dict[str, object] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    size_thresholds = assign_lesion_size_groups(rows)
    if rows:
        with (output_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    payload = {
        "metadata": metadata or {},
        "aggregate": aggregate_metrics(rows),
        "lesion_size_thresholds": size_thresholds,
        "by_lesion_size": {
            group: aggregate_metrics([row for row in rows if row["lesion_size_group"] == group])
            for group in ("empty", "small", "medium", "large")
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
