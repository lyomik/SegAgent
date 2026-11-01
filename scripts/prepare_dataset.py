#!/usr/bin/env python3
"""Build annotation summaries, patient splits, and canonical LIDC-IDRI cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import SimpleITK as sitk
import yaml

from segagent.data.annotations import (
    AnnotationDocument,
    eligible_clusters,
    load_annotation_documents,
    rasterize_consensus_masks,
)
from segagent.data.dicom import image_from_array_like, read_dicom_volume
from segagent.data.splits import (
    SPLIT_NAMES,
    SeriesRecord,
    create_patient_split,
    save_patient_split,
    validate_patient_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/dataset/lung_nodule.yaml"


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid dataset config: {path}")
    return payload


def resolve(path: str) -> Path:
    return (PROJECT_ROOT / path).resolve()


def audit_csv_from_config(config: dict[str, Any]) -> Path:
    run_dir = resolve(config["quality_control"]["reference_audit_run"])
    return run_dir / "artifacts/series_audit.csv"


def read_eligible_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["eligible"] == "True"]
    if not rows:
        raise ValueError(f"No eligible series found in {path}")
    return rows


def summarize_annotations(
    rows: list[dict[str, str]],
    documents: dict[str, AnnotationDocument],
    minimum_readers: int,
) -> tuple[list[SeriesRecord], list[dict[str, object]]]:
    records: list[SeriesRecord] = []
    summaries: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        series_uid = row["series_uid"]
        document = documents[series_uid]
        spacing_z = float(row["spacing_z_mm"])
        clusters = eligible_clusters(
            document,
            spacing_z,
            minimum_readers,
            tolerance=float(row["slice_thickness_mm"]),
        )
        records.append(
            SeriesRecord(
                patient_id=row["patient_id"],
                series_uid=series_uid,
                eligible_nodule_count=len(clusters),
            )
        )
        summaries.append(
            {
                "patient_id": row["patient_id"],
                "series_uid": series_uid,
                "xml_file": document.archive_name,
                "xml_selection_reason": document.selection_reason,
                "reader_annotations": len(document.annotations),
                "eligible_nodules": len(clusters),
            }
        )
        if index % 100 == 0:
            print(f"Annotation summary: {index}/{len(rows)}", flush=True)
    return records, summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_revision_manifest(path: Path, documents: dict[str, AnnotationDocument]) -> None:
    payload = {
        uid: {
            "selected": document.archive_name,
            "reason": document.selection_reason,
            "candidates": list(document.candidates),
        }
        for uid, document in sorted(documents.items())
        if len(document.candidates) > 1
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_split(config: dict[str, Any]) -> tuple[list[SeriesRecord], dict[str, object]]:
    rows = read_eligible_rows(audit_csv_from_config(config))
    archive_path = resolve(config["dataset"]["annotation_archive"])
    documents = load_annotation_documents(archive_path)
    minimum_readers = int(config["construction"]["annotation_consensus"]["minimum_readers"])
    records, summaries = summarize_annotations(rows, documents, minimum_readers)

    canonical_dir = resolve(config["dataset"]["canonical_dir"])
    metadata_dir = canonical_dir / "metadata"
    write_csv(metadata_dir / "annotation_summary.csv", summaries)
    write_revision_manifest(metadata_dir / "annotation_revisions.json", documents)

    split_config = config["construction"]["split"]
    ratios = {name: float(split_config["ratios"][name]) for name in SPLIT_NAMES}
    split = create_patient_split(records, ratios, int(config["construction"]["split_seed"]))
    validate_patient_split(split, records)
    split_path = resolve(config["construction"]["split_file"])
    save_patient_split(split_path, split)
    print(json.dumps({"split_file": str(split_path), **split["patient_counts"]}, indent=2))
    return records, split


def load_annotation_summary(path: Path) -> list[SeriesRecord]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            SeriesRecord(
                patient_id=row["patient_id"],
                series_uid=row["series_uid"],
                eligible_nodule_count=int(row["eligible_nodules"]),
            )
            for row in csv.DictReader(handle)
        ]


def case_id(patient_id: str, series_uid: str) -> str:
    suffix = hashlib.sha256(series_uid.encode()).hexdigest()[:8]
    return f"{patient_id}_{suffix}"


def build_case(
    config: dict[str, Any],
    record: SeriesRecord,
    split_name: str,
    document: AnnotationDocument,
) -> dict[str, object]:
    canonical_dir = resolve(config["dataset"]["canonical_dir"])
    dicom_dir = resolve(config["dataset"]["raw_dicom_dir"]) / record.patient_id / record.series_uid
    volume = read_dicom_volume(dicom_dir)
    consensus_config = config["construction"]["annotation_consensus"]
    spacing_z = float(volume.image.GetSpacing()[2])
    clusters = eligible_clusters(
        document,
        spacing_z,
        int(consensus_config["minimum_readers"]),
        tolerance=volume.slice_thickness_mm,
    )
    semantic, instances = rasterize_consensus_masks(
        clusters,
        volume.sop_to_slice,
        tuple(int(value) for value in volume.array_zyx.shape),
        float(consensus_config["voxel_agreement_fraction"]),
    )
    identifier = case_id(record.patient_id, record.series_uid)
    image_path = canonical_dir / "images" / f"{identifier}.nii.gz"
    label_path = canonical_dir / "labels" / f"{identifier}.nii.gz"
    instance_path = canonical_dir / "instances" / f"{identifier}.nii.gz"
    metadata_path = canonical_dir / "metadata" / "cases" / f"{identifier}.json"
    for path in (image_path, label_path, instance_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    label_image = image_from_array_like(semantic, volume.image)
    instance_image = image_from_array_like(instances, volume.image)
    sitk.WriteImage(sitk.Cast(volume.image, sitk.sitkFloat32), str(image_path), True)
    sitk.WriteImage(sitk.Cast(label_image, sitk.sitkUInt8), str(label_path), True)
    sitk.WriteImage(sitk.Cast(instance_image, sitk.sitkUInt16), str(instance_path), True)
    metadata = {
        "case_id": identifier,
        "patient_id": record.patient_id,
        "series_uid": record.series_uid,
        "split": split_name,
        "eligible_nodules": len(clusters),
        "xml_file": document.archive_name,
        "xml_selection_reason": document.selection_reason,
        "source_uids": volume.source_uids,
        "size_xyz": list(volume.image.GetSize()),
        "spacing_xyz_mm": list(volume.image.GetSpacing()),
        "origin_xyz_mm": list(volume.image.GetOrigin()),
        "direction": list(volume.image.GetDirection()),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "case_id": identifier,
        "patient_id": record.patient_id,
        "series_uid": record.series_uid,
        "split": split_name,
        "image": str(image_path.relative_to(canonical_dir)),
        "label": str(label_path.relative_to(canonical_dir)),
        "instances": str(instance_path.relative_to(canonical_dir)),
        "metadata": str(metadata_path.relative_to(canonical_dir)),
    }


def build_cases(
    config: dict[str, Any],
    split_filter: str | None,
    limit: int | None,
    series_uid: str | None,
) -> None:
    canonical_dir = resolve(config["dataset"]["canonical_dir"])
    summary_path = canonical_dir / "metadata/annotation_summary.csv"
    split_path = resolve(config["construction"]["split_file"])
    if not summary_path.exists() or not split_path.exists():
        raise FileNotFoundError("Run the split action before building canonical cases")
    records = load_annotation_summary(summary_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    series_to_split = {
        series_uid: name
        for name in SPLIT_NAMES
        for series_uid in split["series"][name]
    }
    selected = [
        record
        for record in records
        if (split_filter is None or series_to_split[record.series_uid] == split_filter)
        and (series_uid is None or record.series_uid == series_uid)
    ]
    if limit is not None:
        selected = selected[:limit]
    documents = load_annotation_documents(resolve(config["dataset"]["annotation_archive"]))
    cases = []
    exclusions = []
    for index, record in enumerate(selected, start=1):
        print(f"Build case {index}/{len(selected)}: {record.patient_id}", flush=True)
        try:
            cases.append(
                build_case(
                    config,
                    record,
                    series_to_split[record.series_uid],
                    documents[record.series_uid],
                )
            )
        except ValueError as error:
            exclusion = {
                "patient_id": record.patient_id,
                "series_uid": record.series_uid,
                "reason": str(error),
            }
            exclusions.append(exclusion)
            print(f"Exclude {record.patient_id}: {error}", flush=True)
    manifest_path = canonical_dir / "metadata/cases.json"
    existing = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8")).get("cases", [])
    selected_ids = {case_id(record.patient_id, record.series_uid) for record in selected}
    existing = [item for item in existing if item["case_id"] not in selected_ids]
    merged = {item["case_id"]: item for item in existing}
    merged.update({item["case_id"]: item for item in cases})
    manifest_path.write_text(
        json.dumps({"cases": sorted(merged.values(), key=lambda item: item["case_id"])}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    build_summary = {
        "selected_cases": len(selected),
        "built_cases": len(cases),
        "excluded_cases": len(exclusions),
        "exclusions": exclusions,
    }
    (canonical_dir / "metadata/build_summary.json").write_text(
        json.dumps(build_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("split", "build", "all"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=SPLIT_NAMES)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--series-uid")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    if args.action in {"split", "all"}:
        prepare_split(config)
    if args.action in {"build", "all"}:
        build_cases(config, args.split, args.limit, args.series_uid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
