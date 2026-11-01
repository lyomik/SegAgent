#!/usr/bin/env python3
"""Run all five Stage 1 baseline trainings sequentially on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from segagent.data.dataset import load_manifest
from segagent.models.registry import PROJECT_ROOT

IN_PROCESS_MODELS = ("mednext", "segformer", "swin_unetr", "unetr")
DEFAULT_MANIFEST = PROJECT_ROOT / "data/processed/metadata/cases.json"
DEFAULT_BUILD_SUMMARY = PROJECT_ROOT / "data/canonical/metadata/build_summary.json"


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def run_logged(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return process.returncode


def verify_complete_dataset(manifest_path: Path) -> int:
    cases = load_manifest(manifest_path)
    if not DEFAULT_BUILD_SUMMARY.exists():
        raise FileNotFoundError("Canonical build summary is missing")
    summary = json.loads(DEFAULT_BUILD_SUMMARY.read_text(encoding="utf-8"))
    expected = int(summary["built_cases"])
    if len(cases) != expected:
        raise RuntimeError(f"Processed dataset is incomplete: {len(cases)}/{expected} cases")
    for case in cases:
        if not case.image.exists() or not case.label.exists():
            raise FileNotFoundError(f"Missing processed files for {case.case_id}")
    return expected


def run_nnunet(python: str, logs: Path, state: dict[str, Any], state_path: Path) -> int:
    raw_root = PROJECT_ROOT / "data/nnunet/raw"
    preprocessed_root = PROJECT_ROOT / "data/nnunet/preprocessed"
    results_root = PROJECT_ROOT / "checkpoints/stage1/nnunet"
    env = os.environ.copy()
    env.update(
        {
            "nnUNet_raw": str(raw_root),
            "nnUNet_preprocessed": str(preprocessed_root),
            "nnUNet_results": str(results_root),
            "nnUNet_compile": "false",
        }
    )
    export_status = run_logged(
        [python, "scripts/export_nnunet.py"], logs / "nnunet-export.log", env
    )
    if export_status != 0:
        return export_status
    plan_status = run_logged(
        [
            str(Path(python).with_name("nnUNetv2_plan_and_preprocess")),
            "-d",
            "501",
            "-c",
            "3d_fullres",
            "--verify_dataset_integrity",
        ],
        logs / "nnunet-plan-and-preprocess.log",
        env,
    )
    if plan_status != 0:
        return plan_status
    source_split = raw_root / "Dataset501_LIDC_IDRI/splits_final.json"
    target_split = preprocessed_root / "Dataset501_LIDC_IDRI/splits_final.json"
    shutil.copy2(source_split, target_split)
    state["models"]["nnunet"]["phase"] = "training"
    save_state(state_path, state)
    return run_logged(
        [
            str(Path(python).with_name("nnUNetv2_train")),
            "501",
            "3d_fullres",
            "0",
            "-tr",
            "nnUNetTrainer_100epochs",
            "-p",
            "nnUNetPlans",
            "--npz",
        ],
        logs / "nnunet-train.log",
        env,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--queue-id", default="20261026-stage1-all-baselines-v1")
    args = parser.parse_args()

    case_count = verify_complete_dataset(args.manifest.resolve())
    queue_dir = PROJECT_ROOT / "experiments/stage1/training_queue" / args.queue_id
    logs = queue_dir / "logs"
    logs.mkdir(parents=True, exist_ok=False)
    state_path = queue_dir / "queue.json"
    model_names = (*IN_PROCESS_MODELS, "nnunet")
    state: dict[str, Any] = {
        "queue_id": args.queue_id,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "dataset_cases": case_count,
        "models": {name: {"status": "queued"} for name in model_names},
    }
    save_state(state_path, state)
    python = str(PROJECT_ROOT / ".venv/bin/python")
    failures = 0
    for name in model_names:
        state["models"][name].update(
            {"status": "running", "started_at": datetime.now(UTC).isoformat()}
        )
        save_state(state_path, state)
        if name == "nnunet":
            return_code = run_nnunet(python, logs, state, state_path)
        else:
            run_id = f"20261026-{name}-formal-seed42"
            state["models"][name]["run_id"] = run_id
            save_state(state_path, state)
            return_code = run_logged(
                [python, "scripts/train.py", name, "--run-id", run_id],
                logs / f"{name}.log",
            )
        state["models"][name].update(
            {
                "status": "completed" if return_code == 0 else "failed",
                "return_code": return_code,
                "finished_at": datetime.now(UTC).isoformat(),
            }
        )
        failures += return_code != 0
        save_state(state_path, state)

    state["status"] = "completed" if failures == 0 else "completed_with_failures"
    state["finished_at"] = datetime.now(UTC).isoformat()
    save_state(state_path, state)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
