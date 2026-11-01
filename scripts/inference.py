#!/usr/bin/env python3
"""Run resumable full-case inference for an in-process Stage 1 baseline."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from segagent.data.dataset import load_manifest
from segagent.models.registry import (
    PROJECT_ROOT,
    create_adapter,
    get_baseline,
    load_baselines,
    with_attention,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "data/training_cache/metadata/cases.json"


def _prediction_path(output_dir: Path, case_id: str) -> Path:
    return output_dir / "predictions" / f"{case_id}.nii.gz"


def _atomic_save_nifti(array: np.ndarray, source: nib.spatialimages.SpatialImage, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.nii.gz")
    image = nib.Nifti1Image(array.astype(np.uint8), source.affine, source.header)
    image.set_data_dtype(np.uint8)
    nib.save(image, temporary)
    temporary.replace(path)


def _logits(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, (list, tuple)) else output


@torch.inference_mode()
def infer_3d(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: tuple[int, ...],
    device: torch.device,
    overlap: float,
    sw_batch_size: int = 1,
) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).copy())[None, None].to(device)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        output = sliding_window_inference(
            tensor,
            roi_size=patch_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
            mode="gaussian",
        )
    return _logits(output).argmax(dim=1)[0].byte().cpu().numpy()


@torch.inference_mode()
def infer_2d(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    # NIfTI arrays use (x, y, z); SegFormer treats each z plane independently.
    prediction = np.zeros(image.shape, dtype=np.uint8)
    for start in range(0, image.shape[2], batch_size):
        stop = min(start + batch_size, image.shape[2])
        batch = np.moveaxis(np.asarray(image[:, :, start:stop], dtype=np.float32), 2, 0)
        tensor = torch.from_numpy(batch.copy())[:, None].to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            labels = _logits(model(tensor)).argmax(dim=1).byte().cpu().numpy()
        prediction[:, :, start:stop] = np.moveaxis(labels, 0, 2)
    return prediction


def _completed_prediction(path: Path, source: nib.spatialimages.SpatialImage) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        prediction = nib.load(path)
        return prediction.shape == source.shape and np.allclose(prediction.affine, source.affine)
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("mednext", "segformer", "swin_unetr", "unetr"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--slice-batch-size", type=int, default=4)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--attention-config", type=Path)
    args = parser.parse_args()

    if not 0 <= args.overlap < 1:
        raise ValueError("--overlap must be in [0, 1)")
    spec = get_baseline(args.model, load_baselines())
    if args.attention_config:
        if spec.name != "segformer":
            raise ValueError("Attention inference is currently implemented for SegFormer only")
        spec = with_attention(spec, args.attention_config)
    cases = sorted(
        (case for case in load_manifest(args.manifest.resolve()) if case.split == args.split),
        key=lambda case: case.case_id,
    )
    if args.case_limit is not None:
        cases = cases[: args.case_limit]
    if not cases:
        raise RuntimeError(f"No {args.split} cases in {args.manifest}")

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device(device_name)
    model = create_adapter(spec).build()
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    args.output.mkdir(parents=True, exist_ok=True)
    timings_path = args.output / "case_inference.jsonl"
    prior_timings: dict[str, dict[str, Any]] = {}
    if timings_path.exists():
        for line in timings_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            prior_timings[row["case_id"]] = row

    started_at = datetime.now(UTC).isoformat()
    timings = dict(prior_timings)
    peak_gpu_bytes = 0
    for index, case in enumerate(cases, start=1):
        source = nib.load(str(case.image))
        destination = _prediction_path(args.output, case.case_id)
        if case.case_id in timings and _completed_prediction(destination, source):
            print(json.dumps({"case": case.case_id, "status": "skipped", "progress": f"{index}/{len(cases)}"}), flush=True)
            continue

        image = np.asarray(source.dataobj, dtype=np.float32)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        if int(spec.model["spatial_dims"]) == 3:
            prediction = infer_3d(
                model,
                image,
                tuple(spec.model["patch_size"]),
                device,
                args.overlap,
                args.sw_batch_size,
            )
        else:
            prediction = infer_2d(model, image, device, args.slice_batch_size)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            case_peak = int(torch.cuda.max_memory_allocated(device))
            peak_gpu_bytes = max(peak_gpu_bytes, case_peak)
        else:
            case_peak = 0
        elapsed = time.perf_counter() - started
        _atomic_save_nifti(prediction, source, destination)
        row = {
            "case_id": case.case_id,
            "seconds": elapsed,
            "peak_gpu_bytes": case_peak,
            "shape": list(source.shape),
            "foreground_voxels": int(prediction.sum()),
        }
        timings[case.case_id] = row
        with timings_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps({**row, "progress": f"{index}/{len(cases)}"}), flush=True)

    selected = [timings[case.case_id] for case in cases]
    seconds = np.asarray([row["seconds"] for row in selected], dtype=np.float64)
    peak_gpu_bytes = max(peak_gpu_bytes, max(int(row["peak_gpu_bytes"]) for row in selected))
    metadata = {
        "model": args.model,
        "attention": spec.model.get("attention"),
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "split": args.split,
        "cases": len(cases),
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "device": str(device),
        "parameter_count": parameter_count,
        "peak_gpu_bytes": peak_gpu_bytes,
        "inference_seconds": {
            "mean": float(seconds.mean()),
            "std": float(seconds.std(ddof=1)) if len(seconds) > 1 else 0.0,
            "median": float(np.median(seconds)),
        },
        "overlap": args.overlap if int(spec.model["spatial_dims"]) == 3 else None,
        "sw_batch_size": args.sw_batch_size if int(spec.model["spatial_dims"]) == 3 else None,
        "slice_batch_size": args.slice_batch_size if int(spec.model["spatial_dims"]) == 2 else None,
    }
    (args.output / "inference.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
