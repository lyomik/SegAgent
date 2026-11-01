"""Small, dependency-light helpers for model smoke tests."""

from __future__ import annotations


def minimum_symmetric_padding(
    shape: tuple[int, ...], size: tuple[int, ...]
) -> tuple[tuple[int, int], ...]:
    """Return symmetric padding needed to make every axis at least ``size``."""
    if len(shape) != len(size):
        raise ValueError("shape and size must have the same rank")
    result = []
    for available, requested in zip(shape, size, strict=True):
        total = max(0, requested - available)
        before = total // 2
        result.append((before, total - before))
    return tuple(result)


def centered_slices(
    shape: tuple[int, ...], size: tuple[int, ...], center: tuple[int, ...]
) -> tuple[slice, ...]:
    """Return fixed-size slices centered on a point and kept inside the volume."""
    if not (len(shape) == len(size) == len(center)):
        raise ValueError("shape, size, and center must have the same rank")
    result = []
    for available, requested, midpoint in zip(shape, size, center, strict=True):
        if requested > available:
            raise ValueError(f"Patch size {size} exceeds volume shape {shape}")
        start = min(max(0, midpoint - requested // 2), available - requested)
        result.append(slice(start, start + requested))
    return tuple(result)
