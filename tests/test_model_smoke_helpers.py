import pytest

from segagent.training.smoke import centered_slices, minimum_symmetric_padding


def test_centered_slices_preserve_size_at_boundaries() -> None:
    slices = centered_slices((100, 80, 60), (32, 32, 32), (0, 79, 30))
    assert slices == (slice(0, 32), slice(48, 80), slice(14, 46))


def test_centered_slices_reject_oversized_patch() -> None:
    with pytest.raises(ValueError, match="exceeds volume"):
        centered_slices((10, 10), (11, 8), (5, 5))


def test_minimum_symmetric_padding_only_pads_short_axes() -> None:
    assert minimum_symmetric_padding((236, 300), (256, 256)) == ((10, 10), (0, 0))
    assert minimum_symmetric_padding((255, 254), (256, 256)) == ((0, 1), (1, 1))
