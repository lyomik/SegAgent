#!/usr/bin/env python3
"""Run one real-data optimization step for every Stage 1 baseline model."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional

from segagent.models.registry import PROJECT_ROOT, create_adapter, load_baselines
from segagent.training.smoke import centered_slices

DEFAULT_CASE = "LIDC-IDRI-0001_8807a8bd"


def load_positive_patch(case_id: str, size: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a lesion-centered patch from the shared preprocessed dataset."""
    root = PROJECT_ROOT / "data/processed"
    image_path = root / "images" / f"{case_id}.nii.gz"
    label_path = root / "labels" / f"{case_id}.nii.gz"
    if not image_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"Missing preprocessed smoke case: {case_id}")

    label = np.asarray(nib.load(str(label_path)).dataobj, dtype=np.int64)
    positive = np.argwhere(label > 0)
    if positive.size == 0:
        raise ValueError(f"Smoke case has no positive voxels: {case_id}")
    center = tuple(int(value) for value in np.rint(positive.mean(axis=0)))
    slices = centered_slices(label.shape, size, center)
    image = np.asarray(nib.load(str(image_path)).dataobj[slices], dtype=np.float32)
    label = label[slices]
    if image.shape != size or label.shape != size or not np.any(label):
        raise RuntimeError("Failed to construct a fixed-size positive smoke patch")
    return torch.from_numpy(image.copy())[None, None], torch.from_numpy(label.copy())[None]


def batch_for_model(
    image_3d: torch.Tensor,
    label_3d: torch.Tensor,
    spatial_dims: int,
    patch_size: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    if spatial_dims == 3:
        return image_3d, label_3d
    if spatial_dims != 2:
        raise ValueError(f"Unsupported spatial_dims={spatial_dims}")

    positive_per_slice = label_3d[0].sum(dim=(0, 1))
    index = int(torch.argmax(positive_per_slice).item())
    image = image_3d[:, :, :, :, index]
    label = label_3d[:, :, :, index]
    image = functional.interpolate(image, size=patch_size, mode="bilinear", align_corners=False)
    label = functional.interpolate(label[:, None].float(), size=patch_size, mode="nearest")[:, 0].long()
    return image, label


def smoke_one(spec: Any, image_3d: torch.Tensor, label_3d: torch.Tensor, device: torch.device) -> dict[str, Any]:
    started = time.perf_counter()
    adapter = create_adapter(spec)
    patch_size = adapter.require_patch_size()
    image, target = batch_for_model(image_3d, label_3d, spec.model["spatial_dims"], patch_size)
    prepared = adapter.prepare_batch({"image": image, "label": target})
    model = adapter.build_smoke().to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    logits = model(prepared["image"].to(device))
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    expected = (spec.model["batch_size"], spec.model["out_channels"], *patch_size)
    if tuple(logits.shape) != expected:
        raise RuntimeError(f"{spec.name}: output {tuple(logits.shape)} != expected {expected}")
    if not torch.isfinite(logits).all():
        raise RuntimeError(f"{spec.name}: forward output contains non-finite values")
    loss = functional.cross_entropy(logits.float(), prepared["label"].to(device))
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError(f"{spec.name}: missing or non-finite gradients")
    optimizer.step()
    peak_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    result = {
        "status": "passed",
        "model": spec.name,
        "backend": spec.model["backend"],
        "input_shape": list(prepared["image"].shape),
        "output_shape": list(logits.shape),
        "target_shape": list(prepared["label"].shape),
        "loss": float(loss.detach().cpu()),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_gpu_memory_mb": round(peak_bytes / 1024**2, 1),
        "optimization_step": True,
    }
    if adapter.external_pipeline:
        command = adapter.training_command()
        executable = shutil.which(command[0]) or str(Path(sys.executable).with_name(command[0]))
        if not Path(executable).exists():
            raise RuntimeError(f"{spec.name}: missing external training executable {command[0]}")
        result["external_training_command"] = command
    del model, optimizer, logits, loss, gradients
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default=DEFAULT_CASE)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--model", action="append", dest="models")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    specs = [spec for spec in load_baselines() if spec.enabled]
    if args.models:
        requested = set(args.models)
        specs = [spec for spec in specs if spec.name in requested]
        missing = requested - {spec.name for spec in specs}
        if missing:
            raise ValueError(f"Unknown or disabled models: {', '.join(sorted(missing))}")
    specs_3d = [spec for spec in specs if spec.model["spatial_dims"] == 3]
    max_3d = (
        tuple(max(spec.model["patch_size"][axis] for spec in specs_3d) for axis in range(3))
        if specs_3d
        else (64, 64, 64)
    )
    image, label = load_positive_patch(args.case_id, max_3d)

    run_id = args.run_id or (
        args.output_dir.name
        if args.output_dir is not None
        else datetime.now().strftime("%Y%m%d-%H%M%S-model-smoke")
    )
    output_dir = args.output_dir or PROJECT_ROOT / "experiments/stage1/model_smoke" / run_id
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for spec in specs:
        try:
            result = smoke_one(spec, image, label, device)
        except Exception as error:  # noqa: BLE001 - record one failure and continue the matrix.
            result = {"status": "failed", "model": spec.name, "error": f"{type(error).__name__}: {error}"}
        results.append(result)
        (output_dir / f"{spec.name}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"{spec.name}: {result['status']}", flush=True)

    summary = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "case_id": args.case_id,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "models_passed": sum(result["status"] == "passed" for result in results),
        "models_total": len(results),
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(output_dir.relative_to(PROJECT_ROOT))
    return 0 if summary["models_passed"] == summary["models_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
