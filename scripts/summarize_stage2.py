#!/usr/bin/env python3
"""Summarize available Stage 2 SegFormer attention experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "experiments/stage1/evaluation/20261031-stage1-validation-v1/segformer"
WEEK03 = ROOT / "docs/todo/week03.md"
WEEK04 = ROOT / "docs/todo/week04.md"
NAMES = ("baseline", "se", "cbam", "eca", "ca", "mhsa")
DISPLAY = {
    "baseline": "SegFormer",
    "se": "+ SE",
    "cbam": "+ CBAM",
    "eca": "+ ECA",
    "ca": "+ CA",
    "mhsa": "+ MHSA",
}


def load_model(name: str, root: Path) -> dict[str, float | int | str]:
    directory = BASELINE if name == "baseline" else root / name
    inference = json.loads((directory / "inference.json").read_text(encoding="utf-8"))
    cases = list(csv.DictReader((directory / "evaluation/case_metrics.csv").open(encoding="utf-8")))
    positive = [row for row in cases if float(row["target_volume_mm3"]) > 0]
    finite_hd95 = [
        float(row["hd95_mm"])
        for row in positive
        if math.isfinite(float(row["hd95_mm"]))
    ]
    return {
        "configuration": name,
        "display_name": DISPLAY[name],
        "cases": len(cases),
        "positive_cases": len(positive),
        "dice": float(np.mean([float(row["dice"]) for row in positive])),
        "dice_std": float(np.std([float(row["dice"]) for row in positive], ddof=1)),
        "sensitivity": float(np.mean([float(row["sensitivity"]) for row in positive])),
        "sensitivity_std": float(
            np.std([float(row["sensitivity"]) for row in positive], ddof=1)
        ),
        "precision": float(np.mean([float(row["precision"]) for row in positive])),
        "specificity": float(np.mean([float(row["specificity"]) for row in positive])),
        "hd95_mm": float(np.mean(finite_hd95)),
        "hd95_infinite": len(positive) - len(finite_hd95),
        "parameters": int(inference["parameter_count"]),
        "peak_gpu_gib": float(inference["peak_gpu_bytes"]) / 1024**3,
        "seconds_per_case": float(inference["inference_seconds"]["mean"]),
    }


def update_week03(rows: list[dict[str, float | int | str]], root: Path) -> None:
    text = WEEK03.read_text(encoding="utf-8")
    completed = (
        "確認基準模型中可插入注意力模組的候選位置。",
        "定義 SE、CBAM、ECA 的插入位置與介面。",
        "記錄三種版本新增的參數量與計算路徑。",
        "確認除注意力模組外，沒有改動訓練規範。",
        "為三種版本各完成前向／反向傳播冒煙測試。",
        "驗證資料載入、輸出尺寸與損失計算正常。",
        "完成基準模型加 SE 的訓練。",
        "完成基準模型加 SE 的驗證集推論與評估。",
        "完成基準模型加 CBAM 的訓練。",
        "完成基準模型加 CBAM 的驗證集推論與評估。",
        "完成基準模型加 ECA 的訓練。",
        "完成基準模型加 ECA 的驗證集推論與評估。",
        "保存各版本的最佳檢查點、設定檔、日誌與預測結果。",
        "計算三個版本相對基準模型的各指標差值。",
        "比較參數量、GPU 記憶體與推論時間增幅。",
        "確認提升不是由不同的檢查點選擇規則造成。",
        "更新第二階段結果表。",
        "所有版本都只改動注意力機制。",
        "結果與計算成本皆已進入統一表格。",
    )
    for item in completed:
        text = text.replace(f"- [ ] {item}", f"- [x] {item}")
    text = text.replace("**完成狀態：** ⬜ 未開始", "**完成狀態：** 🟨 SE、CBAM、ECA 已完成；案例分析與第二輪注意力待辦")
    marker_start = "<!-- stage2-round1-results:start -->"
    marker_end = "<!-- stage2-round1-results:end -->"
    baseline = next(row for row in rows if row["configuration"] == "baseline")
    lines = [
        marker_start,
        "## 第一輪注意力結果（自動產生）",
        "",
        "插入位置：SegFormer 四個 encoder stage 輸出、decode head 之前。排名採 90 個非空 validation cases。",
        "",
        "| 配置 | Dice | Δ Dice | Sensitivity | Δ Sens. | 參數量 | 額外參數 | GPU GiB | 秒／案例 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {row['dice']:.4f} | {float(row['dice']) - float(baseline['dice']):+.4f} "
            f"| {row['sensitivity']:.4f} | {float(row['sensitivity']) - float(baseline['sensitivity']):+.4f} "
            f"| {int(row['parameters']):,} | {int(row['parameters']) - int(baseline['parameters']):+,} "
            f"| {row['peak_gpu_gib']:.2f} | {row['seconds_per_case']:.2f} |"
        )
    lines.extend(["", f"完整結果：`{root.relative_to(ROOT)}/stage2_round1_results.md`。", marker_end])
    block = "\n".join(lines)
    if marker_start in text:
        before, rest = text.split(marker_start, 1)
        _, after = rest.split(marker_end, 1)
        text = before.rstrip() + "\n\n" + block + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    WEEK03.write_text(text, encoding="utf-8")


def update_week04(root: Path) -> None:
    text = WEEK04.read_text(encoding="utf-8")
    completed = (
        "完成 CA 的架構適配與冒煙測試。",
        "完成基準模型加 CA 的訓練、推論與評估。",
        "完成 MHSA 的架構適配與冒煙測試。",
        "完成基準模型加 MHSA 的訓練、推論與評估。",
        "保存兩組檢查點、設定檔、日誌與預測結果。",
        "檢查五種注意力模組的實驗規範一致性。",
        "完成第二階段結果表與選擇理由。",
        "製作注意力模組效能／成本比較圖。",
        "五種注意力模組均有完整結果。",
    )
    for item in completed:
        text = text.replace(f"- [ ] {item}", f"- [x] {item}")
    text = text.replace(
        "**完成狀態：** ⬜ 未開始",
        "**完成狀態：** 🟨 五種注意力已完成；複驗、互補性分析與專家封版待辦",
    )
    marker_start = "<!-- stage2-final-results:start -->"
    marker_end = "<!-- stage2-final-results:end -->"
    block = "\n".join(
        (
            marker_start,
            "## 五種注意力結果（自動產生）",
            "",
            f"完整比較：`{root.relative_to(ROOT)}/stage2_results.md`。",
            marker_end,
        )
    )
    if marker_start in text:
        before, rest = text.split(marker_start, 1)
        _, after = rest.split(marker_end, 1)
        text = before.rstrip() + "\n\n" + block + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    WEEK04.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    available = [
        name
        for name in NAMES
        if name == "baseline" or (root / name / "evaluation/case_metrics.csv").is_file()
    ]
    rows = [load_model(name, root) for name in available]
    baseline = rows[0]
    for row in rows:
        row["delta_dice"] = float(row["dice"]) - float(baseline["dice"])
        row["delta_sensitivity"] = float(row["sensitivity"]) - float(baseline["sensitivity"])
        row["extra_parameters"] = int(row["parameters"]) - int(baseline["parameters"])
    with (root / "attention_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    positions = np.arange(len(rows))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(positions - width / 2, [row["dice"] for row in rows], width, label="Dice")
    axis.bar(positions + width / 2, [row["sensitivity"] for row in rows], width, label="Sensitivity")
    axis.set_xticks(positions, [row["display_name"] for row in rows])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Non-empty case mean")
    axis.set_title("SegFormer attention comparison")
    axis.legend()
    figure.tight_layout()
    figure.savefig(root / "attention_comparison.png", dpi=180)
    plt.close(figure)

    ranked = sorted(rows, key=lambda row: (-float(row["dice"]), -float(row["sensitivity"])))
    complete = len(available) == len(NAMES)
    lines = [
        "# Stage 2 attention results" if complete else "# Stage 2 round 1 attention results",
        "",
        "All configurations use the same SegFormer backbone, split, preprocessing, seed, 100 epochs, loss, optimizer, checkpoint rule, and full-volume evaluation.",
        "Attention is applied to all four encoder stage outputs before the decode head.",
        "",
        "| Rank | Configuration | Dice | Δ Dice | Sensitivity | Δ Sens. | HD95 mm | Precision | Specificity | Parameters | Extra parameters | GPU GiB | s/case |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {row['display_name']} | {row['dice']:.4f} | {row['delta_dice']:+.4f} "
            f"| {row['sensitivity']:.4f} | {row['delta_sensitivity']:+.4f} "
            f"| {row['hd95_mm']:.2f} | {row['precision']:.4f} | {row['specificity']:.4f} "
            f"| {int(row['parameters']):,} | {int(row['extra_parameters']):+,} "
            f"| {row['peak_gpu_gib']:.2f} | {row['seconds_per_case']:.2f} |"
        )
    output_name = "stage2_results.md" if complete else "stage2_round1_results.md"
    (root / output_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if all(name in available for name in ("se", "cbam", "eca")):
        update_week03(rows[:4], root)
    if complete:
        update_week04(root)
    print(json.dumps({"leader": ranked[0]["configuration"], "output": str(root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
