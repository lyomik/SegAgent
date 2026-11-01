#!/usr/bin/env python3
"""Wait for nnU-Net training, then run the frozen Stage 1 validation queue."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
MANIFEST = ROOT / "data/training_cache/metadata/cases.json"
LABELS = ROOT / "data/training_cache/labels"
NNUNET_FOLD = ROOT / "checkpoints/stage1/nnunet/Dataset501_LIDC_IDRI/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/fold_0"
RUNS = {
    "mednext": "20261026-mednext-formal-seed42",
    "segformer": "20261031-segformer-formal-seed42-v2",
    "swin_unetr": "20261026-swin_unetr-formal-seed42",
    "unetr": "20261031-unetr-formal-seed42-v2",
}


def log(message: str) -> None:
    print(f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}", flush=True)


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def run(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"COMMAND: {' '.join(command)}\n")
        handle.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - started


def gpu_memory_bytes() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return int(result.stdout.splitlines()[0].strip()) * 1024**2
    except (ValueError, IndexError):
        return 0


def run_nnunet_best(root: Path, env: dict[str, str]) -> None:
    command = [
        str(ROOT / ".venv/bin/nnUNetv2_train"), "501", "3d_fullres", "0",
        "-tr", "nnUNetTrainer_100epochs", "-p", "nnUNetPlans", "--val", "--val_best",
    ]
    log_path = root / "nnunet/inference.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    peak = 0
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        while process.poll() is None:
            peak = max(peak, gpu_memory_bytes())
            time.sleep(1)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
    elapsed = time.perf_counter() - started
    predictions = NNUNET_FOLD / "validation"
    cases = len(list(predictions.glob("*.nii.gz")))
    if cases != 133:
        raise RuntimeError(f"Expected 133 nnU-Net validation predictions, found {cases}")
    checkpoint = torch.load(NNUNET_FOLD / "checkpoint_best.pth", map_location="cpu", weights_only=False)
    parameters = sum(value.numel() for value in checkpoint["network_weights"].values())
    metadata = {
        "model": "nnunet",
        "checkpoint": str((NNUNET_FOLD / "checkpoint_best.pth").resolve()),
        "manifest": str(MANIFEST.resolve()),
        "split": "validation",
        "cases": cases,
        "finished_at": datetime.now(UTC).isoformat(),
        "device": "cuda",
        "parameter_count": parameters,
        "peak_gpu_bytes": peak,
        "inference_seconds": {"mean": elapsed / cases, "std": 0.0, "median": elapsed / cases},
        "timing_note": "Batch wall time divided by 133; per-case dispersion unavailable from nnU-Net validator.",
    }
    (root / "nnunet/inference.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def evaluate(model: str, predictions: Path, root: Path) -> None:
    run(
        [
            str(PYTHON), "scripts/evaluate.py", "--predictions", str(predictions),
            "--labels", str(LABELS), "--output", str(root / model / "evaluation"),
            "--model", model, "--run-id", RUNS.get(model, "nnUNetTrainer_100epochs-fold0-best"),
            "--manifest", str(MANIFEST), "--split", "validation",
        ],
        root / model / "evaluation.log",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)

    if args.wait_pid:
        log(f"waiting for training PID {args.wait_pid}")
        while process_exists(args.wait_pid):
            time.sleep(30)
    if not (NNUNET_FOLD / "checkpoint_final.pth").is_file():
        raise RuntimeError("nnU-Net training ended without checkpoint_final.pth")

    env = os.environ.copy()
    env.update(
        {
            "nnUNet_compile": "false",
            "nnUNet_raw": str(ROOT / "data/nnunet/raw"),
            "nnUNet_preprocessed": str(ROOT / "data/nnunet/preprocessed"),
            "nnUNet_results": str(ROOT / "checkpoints/stage1/nnunet"),
        }
    )
    unetr_run = ROOT / f"experiments/stage1/unetr/{RUNS['unetr']}/run.json"
    if not unetr_run.is_file() or json.loads(unetr_run.read_text(encoding="utf-8"))["status"] != "completed":
        log("running numerically stable UNETR v2 retry")
        run(
            [
                str(PYTHON), "scripts/train.py", "unetr", "--run-id", RUNS["unetr"],
                "--config", "configs/experiments/stage1/baseline_training.yaml",
                "--manifest", str(MANIFEST), "--device", "cuda",
            ],
            root / "unetr/training.log",
        )
    log("running nnU-Net best-checkpoint validation")
    run_nnunet_best(root, env)
    evaluate("nnunet", NNUNET_FOLD / "validation", root)

    for model in ("mednext", "segformer", "swin_unetr", "unetr"):
        run_id = RUNS[model]
        log(f"running {model} validation inference")
        run(
            [
                str(PYTHON), "scripts/inference.py", "--model", model,
                "--checkpoint", str(ROOT / f"checkpoints/stage1/{model}/{run_id}/best.pt"),
                "--manifest", str(MANIFEST), "--split", "validation",
                "--output", str(root / model), "--device", "cuda",
                "--overlap", "0.0", "--sw-batch-size", "4" if model == "mednext" else "1",
            ],
            root / model / "inference.log",
        )
        evaluate(model, root / model / "predictions", root)

    log("building comparison report")
    run([str(PYTHON), "scripts/summarize_stage1.py", "--root", str(root)], root / "summary.log")
    (root / "COMPLETED").write_text(datetime.now(UTC).isoformat() + "\n", encoding="utf-8")
    log("Stage 1 evaluation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
