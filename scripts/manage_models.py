#!/usr/bin/env python3
"""Inspect, validate, and initialize Stage 1 baseline model runs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from segagent.models.registry import PROJECT_ROOT, BaselineSpec, create_adapter, get_baseline, load_baselines


BACKEND_MODULES = {
    "monai": ("torch", "monai"),
    "transformers": ("torch", "transformers"),
    "nnunetv2": ("torch", "nnunetv2"),
}


def dependency_status(backend: str) -> tuple[bool, str]:
    modules = BACKEND_MODULES.get(backend, ())
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    return not missing, ", ".join(missing)


def pending_fields(spec: BaselineSpec) -> list[str]:
    model = spec.model
    if model["name"] == "nnunet":
        pending = []
        if model["architecture"].get("dataset_id") is None:
            pending.append("architecture.dataset_id")
    else:
        pending = [field for field in ("patch_size", "batch_size") if model.get(field) is None]
    return pending


def checkpoint_status(spec: BaselineSpec) -> tuple[bool, str]:
    checkpoint = spec.model["initialization"].get("checkpoint")
    if not checkpoint:
        return True, "scratch"
    path = PROJECT_ROOT / checkpoint
    return path.exists(), str(path.relative_to(PROJECT_ROOT))


def list_models(specs: list[BaselineSpec]) -> None:
    header = f"{'NAME':<12} {'BACKEND':<14} {'DIMS':<5} {'INIT':<22} STATUS"
    print(header)
    print("-" * len(header))
    for spec in specs:
        model = spec.model
        checkpoint_ok, initialization = checkpoint_status(spec)
        dependency_ok, missing = dependency_status(model["backend"])
        status: list[str] = []
        if not spec.enabled:
            status.append("disabled")
        if not checkpoint_ok:
            status.append("checkpoint missing")
        if not dependency_ok:
            status.append(f"deps missing: {missing}")
        pending = pending_fields(spec)
        if pending:
            status.append(f"pending: {','.join(pending)}")
        print(
            f"{spec.name:<12} {model['backend']:<14} {model['spatial_dims']}D   "
            f"{initialization:<22} {'; '.join(status) or 'ready'}"
        )


def validate_models(specs: list[BaselineSpec], strict: bool, runtime: bool) -> int:
    failures: list[str] = []
    pending: list[str] = []
    for spec in specs:
        create_adapter(spec)
        dependency_ok, missing = dependency_status(spec.model["backend"])
        checkpoint_ok, checkpoint = checkpoint_status(spec)
        if not dependency_ok:
            message = f"{spec.name}: missing dependencies: {missing}"
            (failures if runtime else pending).append(message)
        if not checkpoint_ok:
            failures.append(f"{spec.name}: missing checkpoint: {checkpoint}")
        unresolved = pending_fields(spec)
        if unresolved:
            pending.append(f"{spec.name}: {', '.join(unresolved)}")

    for message in failures:
        print(f"ERROR   {message}")
    for message in pending:
        print(f"PENDING {message}")
    if failures or (strict and pending):
        return 1
    print(f"Validated {len(specs)} baseline definitions.")
    return 0


def current_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def initialize_run(spec: BaselineSpec, run_id: str | None) -> None:
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = PROJECT_ROOT / "experiments/stage1" / spec.name / run_id
    checkpoint_dir = PROJECT_ROOT / "checkpoints/stage1" / spec.name / run_id
    if experiment_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError(f"Run already exists: {spec.name}/{run_id}")

    for relative in ("logs", "metrics", "predictions"):
        (experiment_dir / relative).mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(spec.config_path, experiment_dir / "model.yaml")
    shutil.copy2(PROJECT_ROOT / "configs/models/baselines.yaml", experiment_dir / "baselines.yaml")
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "stage": 1,
        "model": spec.name,
        "config": str(spec.config_path.relative_to(PROJECT_ROOT)),
        "git_commit": current_commit(),
        "checkpoint_dir": str(checkpoint_dir.relative_to(PROJECT_ROOT)),
    }
    (experiment_dir / "run.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(experiment_dir.relative_to(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List registered baselines and readiness")
    validate = subparsers.add_parser("validate", help="Validate configs and local initialization weights")
    validate.add_argument("--strict", action="store_true", help="Treat unresolved data-dependent fields as errors")
    validate.add_argument("--runtime", action="store_true", help="Require all training backend dependencies")
    init_run = subparsers.add_parser("init-run", help="Create immutable run directories and snapshots")
    init_run.add_argument("model")
    init_run.add_argument("--run-id")
    args = parser.parse_args()

    specs = load_baselines()
    if args.command == "list":
        list_models(specs)
        return 0
    if args.command == "validate":
        return validate_models(specs, args.strict, args.runtime)
    initialize_run(get_baseline(args.model, specs), args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
