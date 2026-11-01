"""DICOM series loading and geometry conversion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk


@dataclass(frozen=True)
class DicomVolume:
    image: sitk.Image
    array_zyx: np.ndarray
    sop_to_slice: dict[str, int]
    ordered_files: tuple[Path, ...]
    source_uids: dict[str, str]
    slice_thickness_mm: float


def _cross(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.cross(left, right)


def read_dicom_volume(series_dir: Path) -> DicomVolume:
    """Read one validated CT series into HU values and preserve DICOM geometry."""
    headers = []
    for path in sorted(series_dir.glob("*.dcm")):
        dataset = pydicom.dcmread(path, stop_before_pixels=True)
        orientation = np.asarray(dataset.ImageOrientationPatient, dtype=np.float64)
        normal = _cross(orientation[:3], orientation[3:])
        position = np.asarray(dataset.ImagePositionPatient, dtype=np.float64)
        headers.append((float(np.dot(position, normal)), path, dataset, orientation, normal))
    if not headers:
        raise FileNotFoundError(f"No DICOM slices found in {series_dir}")
    headers.sort(key=lambda item: item[0])
    positions = np.asarray([item[0] for item in headers], dtype=np.float64)
    if len(np.unique(np.round(positions, 6))) != len(positions):
        raise ValueError(f"Duplicate slice positions in {series_dir}")

    spacing_z = float(np.median(np.diff(positions)))
    first = headers[0][2]
    first_orientation = headers[0][3]
    first_normal = headers[0][4]
    pixel_spacing = [float(value) for value in first.PixelSpacing]
    arrays = []
    sop_to_slice: dict[str, int] = {}
    ordered_files = []
    for slice_index, (_, path, header, orientation, normal) in enumerate(headers):
        if not np.allclose(orientation, first_orientation, atol=1e-6):
            raise ValueError(f"Inconsistent orientation in {series_dir}")
        if not np.allclose(normal, first_normal, atol=1e-6):
            raise ValueError(f"Inconsistent slice normal in {series_dir}")
        dataset = pydicom.dcmread(path)
        pixels = dataset.pixel_array.astype(np.float32)
        pixels *= float(dataset.RescaleSlope)
        pixels += float(dataset.RescaleIntercept)
        arrays.append(pixels)
        sop_to_slice[str(dataset.SOPInstanceUID)] = slice_index
        ordered_files.append(path)

    array_zyx = np.stack(arrays, axis=0)
    image = sitk.GetImageFromArray(array_zyx)
    image.SetSpacing((pixel_spacing[1], pixel_spacing[0], spacing_z))
    image.SetOrigin(tuple(float(value) for value in first.ImagePositionPatient))
    row_direction = first_orientation[:3]
    column_direction = first_orientation[3:]
    direction_matrix = np.column_stack((row_direction, column_direction, first_normal))
    image.SetDirection(tuple(float(value) for value in direction_matrix.ravel()))
    return DicomVolume(
        image=image,
        array_zyx=array_zyx,
        sop_to_slice=sop_to_slice,
        ordered_files=tuple(ordered_files),
        source_uids={
            "study_uid": str(first.StudyInstanceUID),
            "series_uid": str(first.SeriesInstanceUID),
            "frame_of_reference_uid": str(getattr(first, "FrameOfReferenceUID", "")),
        },
        slice_thickness_mm=float(first.SliceThickness),
    )


def image_from_array_like(array_zyx: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(array_zyx)
    image.CopyInformation(reference)
    return image
