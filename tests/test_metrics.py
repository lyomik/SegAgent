import math

import numpy as np

from segagent.evaluation.metrics import binary_segmentation_metrics, hd95


def test_perfect_and_empty_metrics() -> None:
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    perfect = binary_segmentation_metrics(mask, mask)
    empty = binary_segmentation_metrics(np.zeros_like(mask), np.zeros_like(mask))

    assert perfect.dice == 1.0
    assert perfect.sensitivity == 1.0
    assert perfect.precision == 1.0
    assert perfect.hd95_mm == 0.0
    assert empty.dice == 1.0
    assert empty.sensitivity == 1.0
    assert empty.precision == 1.0
    assert empty.hd95_mm == 0.0


def test_one_empty_mask_policy() -> None:
    target = np.zeros((3, 3, 3), dtype=bool)
    target[1, 1, 1] = True
    result = binary_segmentation_metrics(np.zeros_like(target), target)

    assert result.dice == 0.0
    assert result.sensitivity == 0.0
    assert result.precision == 0.0
    assert math.isinf(result.hd95_mm)


def test_hd95_uses_physical_spacing() -> None:
    left = np.zeros((3, 3, 3), dtype=bool)
    right = np.zeros_like(left)
    left[0, 1, 1] = True
    right[1, 1, 1] = True

    assert hd95(left, right, spacing=(2.5, 1.0, 1.0)) == 2.5
