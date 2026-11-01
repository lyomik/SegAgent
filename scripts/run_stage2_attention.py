#!/usr/bin/env python3
"""Train, infer, evaluate, and summarize Stage 2 attention experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
MANIFEST = ROOT / "data/training_cache/metadata/cases.json"
LABELS = ROOT / "data/training_cache/labels"
ATTENTIONS = ("se", "cbam", "eca", "ca", "mhsa")


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"COMMAND: {' '.join(command)}\n")
        handle.flush()
        subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--attentions",
        nargs="+",
        choices=ATTENTIONS,
        default=list(ATTENTIONS),
        help="Subset to run; completed training runs are reused.",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    for attention in args.attentions:
        run_id = f"20261101-segformer-{attention}-formal-seed42"
        experiment = ROOT / f"experiments/stage2/segformer_{attention}/{run_id}"
        checkpoint = ROOT / f"checkpoints/stage2/segformer_{attention}/{run_id}/best.pt"
        run_json = experiment / "run.json"
        completed = run_json.is_file() and json.loads(run_json.read_text(encoding="utf-8")).get("status") == "completed"
        if not completed:
            run(
                [
                    str(PYTHON), "scripts/train.py", "segformer", "--run-id", run_id,
                    "--experiment-stage", "stage2", "--attention-config", f"configs/attention/{attention}.yaml",
                    "--manifest", str(MANIFEST), "--device", "cuda",
                ],
                output / attention / "training.log",
            )
        run(
            [
                str(PYTHON), "scripts/inference.py", "--model", "segformer",
                "--attention-config", f"configs/attention/{attention}.yaml",
                "--checkpoint", str(checkpoint), "--manifest", str(MANIFEST),
                "--split", "validation", "--output", str(output / attention),
                "--device", "cuda", "--slice-batch-size", "4",
            ],
            output / attention / "inference.log",
        )
        run(
            [
                str(PYTHON), "scripts/evaluate.py", "--predictions", str(output / attention / "predictions"),
                "--labels", str(LABELS), "--output", str(output / attention / "evaluation"),
                "--model", f"segformer_{attention}", "--run-id", run_id,
                "--manifest", str(MANIFEST), "--split", "validation",
            ],
            output / attention / "evaluation.log",
        )

    run([str(PYTHON), "scripts/summarize_stage2.py", "--root", str(output)], output / "summary.log")
    (output / "COMPLETED").write_text(datetime.now(UTC).isoformat() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
