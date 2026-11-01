"""Shared orientation, resampling, and CT intensity transforms."""

from __future__ import annotations

import math

import numpy as np
import SimpleITK as sitk


INTERPOLATORS = {
    "linear": sitk.sitkLinear,
    "nearest": sitk.sitkNearestNeighbor,
    "bspline": sitk.sitkBSpline,
}


def orient_image(image: sitk.Image, orientation: str) -> sitk.Image:
    return sitk.DICOMOrient(image, orientation)


def resample_image(
    image: sitk.Image,
    target_spacing: tuple[float, float, float],
    interpolation: str,
    default_value: float,
    output_pixel_type: int | None = None,
) -> sitk.Image:
    if interpolation not in INTERPOLATORS:
        raise ValueError(f"Unsupported interpolation: {interpolation}")
    old_size = image.GetSize()
    old_spacing = image.GetSpacing()
    new_size = [
        max(1, int(math.floor(size * spacing / target + 0.5)))
        for size, spacing, target in zip(old_size, old_spacing, target_spacing, strict=True)
    ]
    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        INTERPOLATORS[interpolation],
        image.GetOrigin(),
        target_spacing,
        image.GetDirection(),
        default_value,
        image.GetPixelID() if output_pixel_type is None else output_pixel_type,
    )


def normalize_ct(
    image: sitk.Image,
    hu_window: tuple[float, float],
    output_range: tuple[float, float],
) -> sitk.Image:
    hu_min, hu_max = hu_window
    output_min, output_max = output_range
    if hu_max <= hu_min or output_max <= output_min:
        raise ValueError("normalization ranges must be strictly increasing")
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    np.clip(array, hu_min, hu_max, out=array)
    array -= hu_min
    array *= (output_max - output_min) / (hu_max - hu_min)
    array += output_min
    result = sitk.GetImageFromArray(array)
    result.CopyInformation(image)
    return result


def preprocess_case(
    image: sitk.Image,
    label: sitk.Image,
    instance: sitk.Image | None,
    orientation: str,
    target_spacing: tuple[float, float, float],
    image_interpolation: str,
    label_interpolation: str,
    hu_window: tuple[float, float],
    output_range: tuple[float, float],
    image_padding_hu: float,
) -> tuple[sitk.Image, sitk.Image, sitk.Image | None]:
    image = orient_image(image, orientation)
    label = orient_image(label, orientation)
    instance = orient_image(instance, orientation) if instance is not None else None
    image = resample_image(
        image,
        target_spacing,
        image_interpolation,
        image_padding_hu,
        sitk.sitkFloat32,
    )
    label = resample_image(label, target_spacing, label_interpolation, 0, sitk.sitkUInt8)
    if instance is not None:
        instance = resample_image(
            instance,
            target_spacing,
            label_interpolation,
            0,
            sitk.sitkUInt16,
        )
    image = normalize_ct(image, hu_window, output_range)
    return image, label, instance


def same_geometry(left: sitk.Image, right: sitk.Image, tolerance: float = 1e-6) -> bool:
    return (
        left.GetSize() == right.GetSize()
        and np.allclose(left.GetSpacing(), right.GetSpacing(), atol=tolerance)
        and np.allclose(left.GetOrigin(), right.GetOrigin(), atol=tolerance)
        and np.allclose(left.GetDirection(), right.GetDirection(), atol=tolerance)
    )
