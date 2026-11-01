"""Binary voxel-level segmentation metrics with explicit empty-mask behavior."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


@dataclass(frozen=True)
class SegmentationMetrics:
    dice: float
    sensitivity: float
    hd95_mm: float
    precision: float
    specificity: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return mask & ~binary_erosion(mask, border_value=0)


def hd95(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, ...],
) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if len(spacing) != prediction.ndim:
        raise ValueError("spacing dimensionality must match the masks")
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return float("inf")

    prediction_surface = _surface(prediction)
    target_surface = _surface(target)
    distance_to_target = distance_transform_edt(~target_surface, sampling=spacing)
    distance_to_prediction = distance_transform_edt(~prediction_surface, sampling=spacing)
    distances = np.concatenate(
        (distance_to_target[prediction_surface], distance_to_prediction[target_surface])
    )
    return float(np.percentile(distances, 95))


def binary_segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, ...] | None = None,
) -> SegmentationMetrics:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if spacing is None:
        spacing = (1.0,) * prediction.ndim

    true_positive = int(np.count_nonzero(prediction & target))
    true_negative = int(np.count_nonzero(~prediction & ~target))
    false_positive = int(np.count_nonzero(prediction & ~target))
    false_negative = int(np.count_nonzero(~prediction & target))
    prediction_count = true_positive + false_positive
    target_count = true_positive + false_negative

    if prediction_count == 0 and target_count == 0:
        dice = sensitivity = precision = 1.0
    else:
        dice = 2 * true_positive / (prediction_count + target_count)
        sensitivity = true_positive / target_count if target_count else 1.0
        precision = true_positive / prediction_count if prediction_count else 0.0
    negative_count = true_negative + false_positive
    specificity = true_negative / negative_count if negative_count else 1.0
    return SegmentationMetrics(
        dice=float(dice),
        sensitivity=float(sensitivity),
        hd95_mm=hd95(prediction, target, spacing),
        precision=float(precision),
        specificity=float(specificity),
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
    )
