#!/usr/bin/env python3
"""Evaluate NIfTI predictions against a frozen label directory."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from segagent.data.dataset import load_manifest
from segagent.evaluation.evaluator import evaluate_case, write_results


def case_key(path: Path) -> str:
    return path.name.removesuffix(".nii.gz").removesuffix(".nii")


def index_nifti(directory: Path) -> dict[str, Path]:
    paths = sorted(directory.glob("*.nii")) + sorted(directory.glob("*.nii.gz"))
    result: dict[str, Path] = {}
    for path in paths:
        key = case_key(path)
        if key in result:
            raise ValueError(f"Duplicate case ID in {directory}: {key}")
        result[key] = path
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    args = parser.parse_args()

    predictions = index_nifti(args.predictions)
    labels = index_nifti(args.labels)
    if args.manifest or args.split:
        if not (args.manifest and args.split):
            parser.error("--manifest and --split must be provided together")
        selected = {
            case.case_id
            for case in load_manifest(args.manifest.resolve())
            if case.split == args.split
        }
        labels = {key: value for key, value in labels.items() if key in selected}
        missing_labels = sorted(selected - set(labels))
        if missing_labels:
            raise ValueError(json.dumps({"missing_labels": missing_labels}, indent=2))
    missing_predictions = sorted(set(labels) - set(predictions))
    extra_predictions = sorted(set(predictions) - set(labels))
    if missing_predictions or extra_predictions:
        raise ValueError(
            json.dumps(
                {
                    "missing_predictions": missing_predictions,
                    "extra_predictions": extra_predictions,
                },
                indent=2,
            )
        )

    rows = [
        evaluate_case(predictions[key], labels[key], threshold=args.threshold)
        for key in sorted(labels)
    ]
    write_results(
        args.output,
        rows,
        metadata={
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "model": args.model,
            "run_id": args.run_id,
            "threshold": args.threshold,
            "predictions": str(args.predictions),
            "labels": str(args.labels),
            "manifest": str(args.manifest.resolve()) if args.manifest else None,
            "split": args.split,
        },
    )
    print(json.dumps({"cases": len(rows), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
