"""Deterministic stratified patient-level dataset splits."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SeriesRecord:
    patient_id: str
    series_uid: str
    eligible_nodule_count: int


def nodule_count_bin(count: int) -> str:
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 3:
        return "2-3"
    return "4+"


def _target_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {name: total * ratios[name] for name in SPLIT_NAMES}
    result = {name: math.floor(raw[name]) for name in SPLIT_NAMES}
    remaining = total - sum(result.values())
    order = sorted(SPLIT_NAMES, key=lambda name: (raw[name] - result[name], name), reverse=True)
    for name in order[:remaining]:
        result[name] += 1
    return result


def _stratum_quotas(
    sizes: dict[str, int],
    ratios: dict[str, float],
    global_targets: dict[str, int],
) -> dict[str, dict[str, int]]:
    quotas: dict[str, dict[str, int]] = {}
    fractional: list[tuple[float, str, str]] = []
    remaining_by_stratum: dict[str, int] = {}
    assigned_by_split = Counter()
    for stratum, size in sizes.items():
        quotas[stratum] = {}
        for split in SPLIT_NAMES:
            value = size * ratios[split]
            quotas[stratum][split] = math.floor(value)
            assigned_by_split[split] += quotas[stratum][split]
            fractional.append((value - math.floor(value), stratum, split))
        remaining_by_stratum[stratum] = size - sum(quotas[stratum].values())

    remaining_by_split = {
        split: global_targets[split] - assigned_by_split[split]
        for split in SPLIT_NAMES
    }
    for _, stratum, split in sorted(fractional, reverse=True):
        if remaining_by_stratum[stratum] == 0 or remaining_by_split[split] == 0:
            continue
        quotas[stratum][split] += 1
        remaining_by_stratum[stratum] -= 1
        remaining_by_split[split] -= 1

    while any(remaining_by_stratum.values()):
        stratum = next(name for name, value in remaining_by_stratum.items() if value)
        split = max(SPLIT_NAMES, key=lambda name: remaining_by_split[name])
        if remaining_by_split[split] <= 0:
            raise RuntimeError("Unable to satisfy requested split ratios")
        quotas[stratum][split] += 1
        remaining_by_stratum[stratum] -= 1
        remaining_by_split[split] -= 1
    return quotas


def create_patient_split(
    records: Iterable[SeriesRecord],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, object]:
    """Split patients exactly once while stratifying by total nodule burden."""
    records = list(records)
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"ratios must define {SPLIT_NAMES}")
    if not math.isclose(sum(ratios.values()), 1.0):
        raise ValueError("split ratios must sum to one")
    if len({record.series_uid for record in records}) != len(records):
        raise ValueError("series UIDs must be unique")

    patient_records: defaultdict[str, list[SeriesRecord]] = defaultdict(list)
    for record in records:
        patient_records[record.patient_id].append(record)

    strata: defaultdict[str, list[str]] = defaultdict(list)
    patient_nodule_counts: dict[str, int] = {}
    for patient_id, patient_series in patient_records.items():
        count = sum(record.eligible_nodule_count for record in patient_series)
        patient_nodule_counts[patient_id] = count
        strata[nodule_count_bin(count)].append(patient_id)

    targets = _target_counts(len(patient_records), ratios)
    quotas = _stratum_quotas(
        {stratum: len(patient_ids) for stratum, patient_ids in strata.items()},
        ratios,
        targets,
    )
    assignments: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for stratum in sorted(strata):
        patient_ids = sorted(strata[stratum])
        random.Random(f"{seed}:{stratum}").shuffle(patient_ids)
        offset = 0
        for split in SPLIT_NAMES:
            count = quotas[stratum][split]
            assignments[split].extend(patient_ids[offset : offset + count])
            offset += count

    patient_to_split = {
        patient_id: split
        for split, patient_ids in assignments.items()
        for patient_id in patient_ids
    }
    if len(patient_to_split) != len(patient_records):
        raise RuntimeError("not every patient was assigned exactly once")

    series_by_split = {
        split: sorted(
            record.series_uid
            for record in records
            if patient_to_split[record.patient_id] == split
        )
        for split in SPLIT_NAMES
    }
    payload: dict[str, object] = {
        "version": 1,
        "seed": seed,
        "ratios": ratios,
        "stratification": "eligible_nodule_count_bin",
        "patient_counts": {split: len(assignments[split]) for split in SPLIT_NAMES},
        "series_counts": {split: len(series_by_split[split]) for split in SPLIT_NAMES},
        "stratum_counts": dict(
            sorted(Counter(nodule_count_bin(v) for v in patient_nodule_counts.values()).items())
        ),
        "patients": {split: sorted(assignments[split]) for split in SPLIT_NAMES},
        "series": series_by_split,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def save_patient_split(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_patient_split(payload: dict[str, object], records: Iterable[SeriesRecord]) -> None:
    records = list(records)
    patient_sets = {
        split: set(payload["patients"][split])  # type: ignore[index]
        for split in SPLIT_NAMES
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if patient_sets[left] & patient_sets[right]:
            raise ValueError(f"patient leakage between {left} and {right}")
    expected_patients = {record.patient_id for record in records}
    if set.union(*patient_sets.values()) != expected_patients:
        raise ValueError("split does not cover exactly the eligible patients")
    expected_series = {record.series_uid for record in records}
    actual_series = {
        series_uid
        for split in SPLIT_NAMES
        for series_uid in payload["series"][split]  # type: ignore[index]
    }
    if actual_series != expected_series:
        raise ValueError("split does not cover exactly the eligible series")
