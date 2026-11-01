import importlib.util
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/inference.py"
SPEC = importlib.util.spec_from_file_location("stage1_inference", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Threshold3D(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.cat((-image, image), dim=1)


class Threshold2D(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.cat((-image, image), dim=1)


def test_3d_sliding_window_preserves_shape_and_thresholds() -> None:
    image = np.linspace(-1, 1, 9 * 8 * 7, dtype=np.float32).reshape(9, 8, 7)
    result = MODULE.infer_3d(Threshold3D(), image, (6, 6, 6), torch.device("cpu"), 0.25, 2)
    np.testing.assert_array_equal(result, image > 0)


def test_2d_slice_batches_preserve_nifti_axis_order() -> None:
    image = np.linspace(-1, 1, 7 * 6 * 5, dtype=np.float32).reshape(7, 6, 5)
    result = MODULE.infer_2d(Threshold2D(), image, torch.device("cpu"), 2)
    np.testing.assert_array_equal(result, image > 0)
