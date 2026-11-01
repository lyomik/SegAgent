#!/usr/bin/env python3
"""Train one in-process Stage 1 segmentation baseline."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import yaml
from torch.nn import functional

from segagent.data.dataset import CanonicalCase, load_manifest
from segagent.models.registry import (
    PROJECT_ROOT,
    create_adapter,
    get_baseline,
    load_baselines,
    with_attention,
)
from segagent.training.smoke import centered_slices, minimum_symmetric_padding

DEFAULT_TRAINING_CONFIG = PROJECT_ROOT / "configs/experiments/stage1/baseline_training.yaml"
PROCESSED_MANIFEST = PROJECT_ROOT / "data/processed/metadata/cases.json"
TRAINING_CACHE_MANIFEST = PROJECT_ROOT / "data/training_cache/metadata/cases.json"
DEFAULT_MANIFEST = (
    TRAINING_CACHE_MANIFEST if TRAINING_CACHE_MANIFEST.exists() else PROCESSED_MANIFEST
)
DEFAULT_BUILD_SUMMARY = PROJECT_ROOT / "data/canonical/metadata/build_summary.json"


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return payload


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def choose_center(label: np.ndarray, rng: random.Random, positive_probability: float) -> tuple[int, ...]:
    positive = np.argwhere(label > 0)
    if positive.size and rng.random() < positive_probability:
        return tuple(int(value) for value in positive[rng.randrange(len(positive))])
    return tuple(rng.randrange(length) for length in label.shape)


def load_patch(
    case: CanonicalCase,
    patch_size: tuple[int, ...],
    spatial_dims: int,
    rng: random.Random,
    positive_probability: float,
    augment: bool,
    flip_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    label_volume = np.asarray(nib.load(str(case.label)).dataobj, dtype=np.int64)
    center = choose_center(label_volume, rng, positive_probability)
    image_proxy = nib.load(str(case.image)).dataobj
    if spatial_dims == 3:
        slices = centered_slices(label_volume.shape, patch_size, center)
        image = np.asarray(image_proxy[slices], dtype=np.float32)
        label = label_volume[slices]
    elif spatial_dims == 2:
        z_index = center[2]
        image_slice = np.asarray(image_proxy[:, :, z_index], dtype=np.float32)
        label_slice = label_volume[:, :, z_index]
        padding = minimum_symmetric_padding(label_slice.shape, patch_size)
        if any(before or after for before, after in padding):
            image_slice = np.pad(image_slice, padding, mode="constant", constant_values=0.0)
            label_slice = np.pad(label_slice, padding, mode="constant", constant_values=0)
        padded_center = tuple(
            midpoint + before for midpoint, (before, _) in zip(center[:2], padding, strict=True)
        )
        xy_slices = centered_slices(label_slice.shape, patch_size, padded_center)
        image = image_slice[xy_slices]
        label = label_slice[xy_slices]
    else:
        raise ValueError(f"Unsupported spatial_dims={spatial_dims}")

    image_tensor = torch.from_numpy(image.copy())[None]
    label_tensor = torch.from_numpy(label.copy()).long()
    if augment:
        for axis in range(spatial_dims):
            if rng.random() < flip_probability:
                image_tensor = torch.flip(image_tensor, dims=(axis + 1,))
                label_tensor = torch.flip(label_tensor, dims=(axis,))
    return image_tensor, label_tensor


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = logits.softmax(dim=1)[:, 1]
    target_float = (target == 1).float()
    dimensions = tuple(range(1, target.ndim))
    intersection = (probabilities * target_float).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + target_float.sum(dim=dimensions)
    return 1.0 - ((2.0 * intersection + 1e-5) / (denominator + 1e-5)).mean()


def foreground_dice(logits: torch.Tensor, target: torch.Tensor) -> float:
    prediction = logits.argmax(dim=1) == 1
    truth = target == 1
    intersection = torch.logical_and(prediction, truth).sum().item()
    denominator = prediction.sum().item() + truth.sum().item()
    return 1.0 if denominator == 0 else 2.0 * intersection / denominator


def run_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cases: list[CanonicalCase],
    model_config: dict[str, Any],
    settings: dict[str, Any],
    rng: random.Random,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for _ in range(int(settings["steps_per_epoch"])):
        case = cases[rng.randrange(len(cases))]
        image, target = load_patch(
            case,
            tuple(model_config["patch_size"]),
            int(model_config["spatial_dims"]),
            rng,
            float(settings["positive_patch_probability"]),
            True,
            float(settings["augmentation"]["random_flip_probability"]),
        )
        image, target = image[None].to(device), target[None].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(image)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            ce = functional.cross_entropy(logits.float(), target)
            dice = soft_dice_loss(logits.float(), target)
            loss = float(settings["loss"]["cross_entropy_weight"]) * ce + float(
                settings["loss"]["dice_weight"]
            ) * dice
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {float(loss.detach().cpu())}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(settings.get("max_gradient_norm", 1.0))
        )
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    cases: list[CanonicalCase],
    model_config: dict[str, Any],
    settings: dict[str, Any],
    seed: int,
    device: torch.device,
) -> float:
    model.eval()
    rng = random.Random(seed)
    scores = []
    limit = min(len(cases), int(settings["validation"]["cases_per_epoch"]))
    for case in cases[:limit]:
        image, target = load_patch(
            case,
            tuple(model_config["patch_size"]),
            int(model_config["spatial_dims"]),
            rng,
            1.0,
            False,
            0.0,
        )
        logits = model(image[None].to(device))
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        scores.append(foreground_dice(logits, target[None].to(device)))
    return float(np.mean(scores))


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_dice: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_val_dice": best_dice,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-id")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--allow-incomplete-data", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--attention-config", type=Path)
    parser.add_argument("--experiment-stage", choices=("stage1", "stage2"), default="stage1")
    args = parser.parse_args()

    spec = get_baseline(args.model, load_baselines())
    if args.attention_config:
        if spec.name != "segformer":
            raise ValueError("Stage 2 attention is currently implemented for SegFormer only")
        spec = with_attention(spec, args.attention_config)
    adapter = create_adapter(spec)
    if adapter.external_pipeline:
        raise ValueError("nnU-Net uses its external training pipeline")
    settings = read_yaml(args.config.resolve())["training"]
    if args.epochs is not None:
        settings["epochs"] = args.epochs
    if args.steps_per_epoch is not None:
        settings["steps_per_epoch"] = args.steps_per_epoch

    cases = load_manifest(args.manifest.resolve())
    expected = 888
    if DEFAULT_BUILD_SUMMARY.exists():
        build_summary = json.loads(DEFAULT_BUILD_SUMMARY.read_text(encoding="utf-8"))
        expected = int(build_summary["built_cases"])
    if len(cases) < expected and not args.allow_incomplete_data:
        raise RuntimeError(
            f"Refusing formal training with {len(cases)}/{expected} processed cases; "
            "pass --allow-incomplete-data only for development runs"
        )
    train_cases = sorted((case for case in cases if case.split == "train"), key=lambda case: case.case_id)
    validation_cases = sorted(
        (case for case in cases if case.split == "validation"), key=lambda case: case.case_id
    )
    if not train_cases or not validation_cases:
        raise RuntimeError("Training and validation cases are both required")

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device(device_name)
    seed = int(settings["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)

    attention_name = spec.model.get("attention", {}).get("name")
    artifact_name = f"{spec.name}_{attention_name}" if attention_name else spec.name
    run_id = args.run_id or datetime.now().strftime(f"%Y%m%d-%H%M%S-{artifact_name}-seed{seed}")
    experiment_dir = PROJECT_ROOT / "experiments" / args.experiment_stage / artifact_name / run_id
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / args.experiment_stage / artifact_name / run_id
    if experiment_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError(f"Run already exists: {spec.name}/{run_id}")
    (experiment_dir / "logs").mkdir(parents=True)
    (experiment_dir / "metrics").mkdir()
    checkpoint_dir.mkdir(parents=True)
    shutil.copy2(spec.config_path, experiment_dir / "model.yaml")
    shutil.copy2(args.config, experiment_dir / "training.yaml")
    if args.attention_config:
        shutil.copy2(args.attention_config, experiment_dir / "attention.yaml")
    metadata = {
        "run_id": run_id,
        "model": spec.name,
        "artifact_name": artifact_name,
        "stage": args.experiment_stage,
        "attention": spec.model.get("attention"),
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "seed": seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dataset_cases": len(cases),
        "dataset_manifest": str(args.manifest.resolve()),
        "train_cases": len(train_cases),
        "validation_cases": len(validation_cases),
        "formal_dataset_complete": len(cases) >= expected,
    }
    run_path = experiment_dir / "run.json"
    run_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    model = adapter.build().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    metrics_path = experiment_dir / "metrics/history.jsonl"
    best_dice = -1.0
    started = time.perf_counter()
    try:
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            for epoch in range(1, int(settings["epochs"]) + 1):
                train_loss = run_epoch(
                    model, optimizer, scaler, train_cases, spec.model, settings, rng, device
                )
                val_dice = validate(
                    model, validation_cases, spec.model, settings, seed + epoch, device
                )
                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_dice": val_dice,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
                metrics_file.write(json.dumps(row) + "\n")
                metrics_file.flush()
                print(json.dumps(row), flush=True)
                if val_dice > best_dice:
                    best_dice = val_dice
                    save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, best_dice)
                save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, best_dice)
    except BaseException:
        metadata["status"] = "failed"
        metadata["finished_at"] = datetime.now(UTC).isoformat()
        run_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        raise

    metadata.update(
        {
            "status": "completed",
            "finished_at": datetime.now(UTC).isoformat(),
            "best_val_dice": best_dice,
        }
    )
    run_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
