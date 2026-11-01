#!/usr/bin/env python3
"""Audit LIDC-IDRI DICOM geometry and XML coverage without changing source data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import pydicom


GEOMETRY_TAGS = [
    "Columns",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "PixelSpacing",
    "RescaleIntercept",
    "RescaleSlope",
    "Rows",
    "SOPInstanceUID",
    "SeriesInstanceUID",
    "SliceThickness",
]


@dataclass
class SeriesAudit:
    patient_id: str
    series_uid: str
    dicom_files: int
    rows: int | None
    columns: int | None
    spacing_x_mm: float | None
    spacing_y_mm: float | None
    spacing_z_mm: float | None
    slice_thickness_mm: float | None
    xml_files: int
    download_complete: bool
    eligible: bool
    issues: str


def local_name(element: ElementTree.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def vector_cross(left: list[float], right: list[float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def vector_dot(left: list[float], right: tuple[float, float, float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def quantiles(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(value for value in values if math.isfinite(value))
    result: dict[str, float | int] = {"count": len(ordered)}
    if not ordered:
        return result
    for fraction, label in ((0, "min"), (0.25, "p25"), (0.5, "median"), (0.75, "p75"), (1, "max")):
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        result[label] = round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)
    return result


def annotation_index(archive_path: Path) -> tuple[dict[str, list[str]], dict[str, object]]:
    series_to_xml: defaultdict[str, list[str]] = defaultdict(list)
    non_ct_xml = 0
    parse_errors: list[str] = []
    with ZipFile(archive_path) as archive:
        bad_entry = archive.testzip()
        for name in archive.namelist():
            if not name.lower().endswith(".xml"):
                continue
            try:
                root = ElementTree.fromstring(archive.read(name))
                uids = [
                    (element.text or "").strip()
                    for element in root.iter()
                    if local_name(element) == "SeriesInstanceUid"
                ]
                if len(uids) == 1 and uids[0]:
                    series_to_xml[uids[0]].append(name)
                else:
                    non_ct_xml += 1
            except ElementTree.ParseError:
                parse_errors.append(name)
    summary = {
        "archive_crc_ok": bad_entry is None,
        "bad_archive_entry": bad_entry,
        "ct_xml_files": sum(len(names) for names in series_to_xml.values()),
        "ct_series_uids": len(series_to_xml),
        "duplicate_ct_series_uids": sum(len(names) > 1 for names in series_to_xml.values()),
        "non_ct_xml_files": non_ct_xml,
        "parse_errors": parse_errors,
    }
    return dict(series_to_xml), summary


def audit_series(
    series_dir: Path,
    xml_names: list[str],
    maximum_slice_thickness: float,
) -> SeriesAudit:
    issues: list[str] = []
    positions: list[float] = []
    sop_uids: list[str] = []
    rows: set[int] = set()
    columns: set[int] = set()
    spacing_x: set[float] = set()
    spacing_y: set[float] = set()
    thicknesses: set[float] = set()
    orientations: set[tuple[float, ...]] = set()
    series_uids: set[str] = set()
    files = sorted(series_dir.glob("*.dcm"))
    download_complete = (series_dir / ".complete").exists()

    for path in files:
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=GEOMETRY_TAGS)
            orientation = tuple(round(float(value), 8) for value in dataset.ImageOrientationPatient)
            position = [float(value) for value in dataset.ImagePositionPatient]
            normal = vector_cross(list(orientation[:3]), list(orientation[3:]))
            positions.append(vector_dot(position, normal))
            orientations.add(orientation)
            sop_uids.append(str(dataset.SOPInstanceUID))
            series_uids.add(str(dataset.SeriesInstanceUID))
            rows.add(int(dataset.Rows))
            columns.add(int(dataset.Columns))
            spacing = [float(value) for value in dataset.PixelSpacing]
            spacing_x.add(round(spacing[0], 8))
            spacing_y.add(round(spacing[1], 8))
            thicknesses.add(round(float(dataset.SliceThickness), 8))
            if "RescaleSlope" not in dataset or "RescaleIntercept" not in dataset:
                issues.append("missing_rescale_parameters")
        except Exception as error:  # The report must retain all malformed-file failures.
            issues.append(f"unreadable_dicom:{path.name}:{type(error).__name__}")

    if not files:
        issues.append("no_dicom_files")
    if not download_complete:
        issues.append("missing_download_complete_marker")
    if series_uids != {series_dir.name}:
        issues.append("series_uid_mismatch")
    if len(sop_uids) != len(set(sop_uids)):
        issues.append("duplicate_sop_instance_uid")
    if len(rows) != 1 or len(columns) != 1:
        issues.append("inconsistent_matrix_size")
    if len(spacing_x) != 1 or len(spacing_y) != 1:
        issues.append("inconsistent_pixel_spacing")
    if len(thicknesses) != 1:
        issues.append("inconsistent_slice_thickness")
    if len(orientations) != 1:
        issues.append("inconsistent_orientation")
    if not xml_names:
        issues.append("missing_xml")
    if len(xml_names) > 1:
        issues.append("duplicate_xml_revision")

    unique_positions = sorted(set(round(value, 6) for value in positions))
    spacing_z: float | None = None
    if len(unique_positions) != len(files):
        issues.append("duplicate_slice_position")
    if len(unique_positions) > 1:
        differences = [
            unique_positions[index + 1] - unique_positions[index]
            for index in range(len(unique_positions) - 1)
        ]
        spacing_z = statistics.median(differences)
        tolerance = max(0.001, spacing_z * 0.02)
        if any(abs(value - spacing_z) > tolerance for value in differences):
            issues.append("nonuniform_or_missing_slices")

    thickness = next(iter(thicknesses), None)
    if thickness is not None and thickness > maximum_slice_thickness:
        issues.append("slice_thickness_above_limit")

    blocking = {
        "duplicate_slice_position",
        "duplicate_sop_instance_uid",
        "inconsistent_matrix_size",
        "inconsistent_orientation",
        "inconsistent_pixel_spacing",
        "inconsistent_slice_thickness",
        "missing_rescale_parameters",
        "missing_download_complete_marker",
        "missing_xml",
        "no_dicom_files",
        "nonuniform_or_missing_slices",
        "series_uid_mismatch",
        "slice_thickness_above_limit",
    }
    unique_issues = sorted(set(issues))
    return SeriesAudit(
        patient_id=series_dir.parent.name,
        series_uid=series_dir.name,
        dicom_files=len(files),
        rows=next(iter(rows), None),
        columns=next(iter(columns), None),
        spacing_x_mm=next(iter(spacing_x), None),
        spacing_y_mm=next(iter(spacing_y), None),
        spacing_z_mm=round(spacing_z, 6) if spacing_z is not None else None,
        slice_thickness_mm=next(iter(thicknesses), None),
        xml_files=len(xml_names),
        download_complete=download_complete,
        eligible=not any(issue.split(":", 1)[0] in blocking for issue in unique_issues),
        issues=";".join(unique_issues),
    )


def write_csv(path: Path, rows: list[SeriesAudit]) -> None:
    fields = list(SeriesAudit.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-slice-thickness", type=float, default=2.5)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    series_to_xml, xml_summary = annotation_index(args.annotations)
    series_dirs = sorted(path for path in args.dicom_dir.glob("*/*") if path.is_dir())

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        audits = list(
            pool.map(
                lambda path: audit_series(
                    path,
                    series_to_xml.get(path.name, []),
                    args.max_slice_thickness,
                ),
                series_dirs,
            )
        )

    write_csv(args.output_dir / "series_audit.csv", audits)
    issue_counts = Counter(
        issue
        for audit in audits
        for issue in audit.issues.split(";")
        if issue
    )
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": {
            "annotations": str(args.annotations),
            "dicom_dir": str(args.dicom_dir),
            "maximum_slice_thickness_mm": args.max_slice_thickness,
        },
        "summary": {
            "patients": len({audit.patient_id for audit in audits}),
            "series": len(audits),
            "dicom_files": sum(audit.dicom_files for audit in audits),
            "eligible_series": sum(audit.eligible for audit in audits),
            "excluded_series": sum(not audit.eligible for audit in audits),
            "issue_counts": dict(sorted(issue_counts.items())),
            "pixel_spacing_x_mm": quantiles(
                [audit.spacing_x_mm for audit in audits if audit.spacing_x_mm is not None]
            ),
            "pixel_spacing_y_mm": quantiles(
                [audit.spacing_y_mm for audit in audits if audit.spacing_y_mm is not None]
            ),
            "slice_spacing_mm": quantiles(
                [audit.spacing_z_mm for audit in audits if audit.spacing_z_mm is not None]
            ),
            "slice_thickness_mm": quantiles(
                [
                    audit.slice_thickness_mm
                    for audit in audits
                    if audit.slice_thickness_mm is not None
                ]
            ),
        },
        "xml": xml_summary,
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
