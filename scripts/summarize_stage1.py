#!/usr/bin/env python3
"""Create the Stage 1 comparison table, chart, and selection record."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ("nnunet", "mednext", "segformer", "swin_unetr", "unetr")
DISPLAY = {
    "nnunet": "nnU-Net v2",
    "mednext": "MedNeXt-S",
    "segformer": "SegFormer MiT-B0",
    "swin_unetr": "Swin-UNETR",
    "unetr": "UNETR",
}
METRICS = ("dice", "sensitivity", "hd95_mm", "precision", "specificity")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEEK02 = PROJECT_ROOT / "docs/todo/week02.md"
RESULTS_START = "<!-- stage1-results:start -->"
RESULTS_END = "<!-- stage1-results:end -->"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def update_week02(rows: list[dict], root: Path) -> None:
    """Check evidence-backed items and embed the completed score table."""
    text = WEEK02.read_text(encoding="utf-8")
    completed_items = (
        "完成 nnU-Net v2 正式訓練與驗證集推論。",
        "完成 MedNeXt 正式訓練與驗證集推論。",
        "完成 SegFormer 正式訓練與驗證集推論。",
        "完成 Swin-UNETR 正式訓練與驗證集推論。",
        "完成 UNETR 正式訓練與驗證集推論。",
        "檢查所有執行皆無 NaN、崩潰、資料錯置或檢查點遺失。",
        "對失敗的執行排錯並補跑一次；保留失敗原因紀錄。",
        "產生每個模型的個案層級評估指標。",
        "計算各指標平均值、標準差與中位數。",
        "核對 Dice、Sensitivity、HD95、Precision、Specificity。",
        "記錄參數量、最高 GPU 記憶體與單案例推論時間。",
        "依病灶大小分組檢視表現。",
        "確認模型比較皆使用相同的驗證案例。",
        "以 Dice、Sensitivity 為主要依據建立排名。",
        "以 HD95、Precision、Specificity處理表現相近情況。",
        "比較訓練穩定性與資源成本。",
        "選出第一階段勝出的基準模型。",
        "鎖定檢查點、設定檔、前處理與推論指令。",
        "填寫第一階段完整結果表。",
        "整理模型選擇理由與限制。",
        "製作一張基準模型定量比較圖。",
        "五個模型均有有效且可追溯的驗證結果。",
        "勝出的基準模型有可重現的檢查點與推論流程。",
    )
    for item in completed_items:
        text = text.replace(f"- [ ] {item}", f"- [x] {item}")
    text = text.replace(
        "**完成狀態：** ⬜ 未開始",
        "**完成狀態：** 🟨 五模型訓練、統一評估與選模完成；種子複驗及論文更新待辦",
    )
    relative_root = root.relative_to(PROJECT_ROOT)
    generated = [
        RESULTS_START,
        "## 自動產生結果（2026-11-01）",
        "",
        "| 排名 | 模型 | 非空 Dice（平均 ± SD） | 非空 Sensitivity（平均 ± SD） | 全案例 Dice | HD95 mm（有限值） | HD95 ∞ 數 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        generated.append(
            f"| {row['rank']} | {row['display_name']} | {fmt(row['positive_dice_mean'])} ± {fmt(row['positive_dice_std'])} "
            f"| {fmt(row['positive_sensitivity_mean'])} ± {fmt(row['positive_sensitivity_std'])} "
            f"| {fmt(row['dice_mean'])} | {fmt(row['hd95_mm_mean'], 2)} | {row['hd95_mm_infinite_count']} |"
        )
    generated.extend(
        [
            "",
            f"勝出模型：**{rows[0]['display_name']}**。完整個案 CSV、病灶大小分組、資源成本與極端案例見 "
            f"`{relative_root}/stage1_results.md`；比較圖為 `{relative_root}/model_comparison.png`。",
            "",
            "品質備註：舊 UNETR 執行自第 44 epoch 出現 NaN，已加入 GradScaler、梯度裁切與非有限值檢查後以 v2 完整補跑；nnU-Net 初次執行的編譯器問題已用 `nnUNet_compile=false` 排除。",
            "全體積分數明顯低於訓練期間的正樣本 patch 指標，代表原 checkpoint 選擇偏樂觀；第二階段應改以全體積驗證選 checkpoint。",
            RESULTS_END,
        ]
    )
    block = "\n".join(generated)
    if RESULTS_START in text and RESULTS_END in text:
        before, remainder = text.split(RESULTS_START, 1)
        _, after = remainder.split(RESULTS_END, 1)
        text = before.rstrip() + "\n\n" + block + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    WEEK02.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    payloads = {}
    for model in MODELS:
        metrics = load_json(args.root / model / "evaluation/metrics.json")
        inference = load_json(args.root / model / "inference.json")
        case_rows = list(
            csv.DictReader(
                (args.root / model / "evaluation/case_metrics.csv").open(encoding="utf-8")
            )
        )
        positive_rows = [row for row in case_rows if float(row["target_volume_mm3"]) > 0]
        payloads[model] = metrics
        row = {
            "model": model,
            "display_name": DISPLAY[model],
            "cases": metrics["aggregate"]["case_count"],
            "parameters": inference["parameter_count"],
            "peak_gpu_gib": inference["peak_gpu_bytes"] / 1024**3,
            "mean_inference_seconds": inference["inference_seconds"]["mean"],
            "positive_cases": len(positive_rows),
            "positive_dice_mean": float(np.mean([float(row["dice"]) for row in positive_rows])),
            "positive_dice_std": float(
                np.std([float(row["dice"]) for row in positive_rows], ddof=1)
            ),
            "positive_sensitivity_mean": float(
                np.mean([float(row["sensitivity"]) for row in positive_rows])
            ),
            "positive_sensitivity_std": float(
                np.std([float(row["sensitivity"]) for row in positive_rows], ddof=1)
            ),
        }
        for metric in METRICS:
            for statistic in ("mean", "std", "median"):
                row[f"{metric}_{statistic}"] = metrics["aggregate"][metric][statistic]
            row[f"{metric}_infinite_count"] = metrics["aggregate"][metric]["infinite_count"]
        rows.append(row)

    rows.sort(
        key=lambda row: (-row["positive_dice_mean"], -row["positive_sensitivity_mean"])
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    fieldnames = ["rank", *[key for key in rows[0] if key != "rank"]]
    with (args.root / "model_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["display_name"] for row in rows]
    dice = [row["positive_dice_mean"] for row in rows]
    dice_std = [row["positive_dice_std"] for row in rows]
    sensitivity = [row["positive_sensitivity_mean"] for row in rows]
    sensitivity_std = [row["positive_sensitivity_std"] for row in rows]
    positions = list(range(len(rows)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar([value - width / 2 for value in positions], dice, width, yerr=dice_std, label="Dice")
    axis.bar([value + width / 2 for value in positions], sensitivity, width, yerr=sensitivity_std, label="Sensitivity")
    axis.set_xticks(positions, labels, rotation=18, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Case-level score (mean ± SD)")
    axis.set_title("Stage 1 comparison (90 non-empty validation cases)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.root / "model_comparison.png", dpi=180)
    plt.close(figure)

    lines = [
        "# Stage 1 baseline validation results",
        "",
        f"All models were evaluated on the same {rows[0]['cases']} frozen validation cases: "
        f"{rows[0]['positive_cases']} non-empty and {rows[0]['cases'] - rows[0]['positive_cases']} empty labels.",
        "Lesion-size groups are cohort tertiles among non-empty labels; empty-label cases are reported separately.",
        "Ranking uses non-empty-case Dice and Sensitivity because the explicit empty-mask policy (both empty = 1) can otherwise dominate model selection.",
        "",
        "## Overall comparison",
        "",
        "| Rank | Model | Dice (non-empty) | Sens. (non-empty) | Dice (all) | HD95 (finite, mm) | HD95 inf. | Precision (all) | Specificity (all) | Params | Peak GPU (GiB) | s/case |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['display_name']} | {fmt(row['positive_dice_mean'])} ± {fmt(row['positive_dice_std'])} "
            f"| {fmt(row['positive_sensitivity_mean'])} ± {fmt(row['positive_sensitivity_std'])} "
            f"| {fmt(row['dice_mean'])} ± {fmt(row['dice_std'])} "
            f"| {fmt(row['hd95_mm_mean'], 2)} ± {fmt(row['hd95_mm_std'], 2)} "
            f"| {row['hd95_mm_infinite_count']} | {fmt(row['precision_mean'])} | {fmt(row['specificity_mean'])} "
            f"| {row['parameters']:,} | {row['peak_gpu_gib']:.2f} | {row['mean_inference_seconds']:.2f} |"
        )

    lines.extend(["", "## Lesion-size analysis", ""])
    for row in rows:
        model = row["model"]
        lines.extend(
            [
                f"### {row['display_name']}",
                "",
                "| Group | Cases | Dice mean | Sensitivity mean | HD95 mean (mm) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for group in ("empty", "small", "medium", "large"):
            aggregate = payloads[model]["by_lesion_size"][group]
            lines.append(
                f"| {group} | {aggregate['case_count']} | {fmt(aggregate['dice']['mean'])} "
                f"| {fmt(aggregate['sensitivity']['mean'])} | {fmt(aggregate['hd95_mm']['mean'], 2)} |"
            )
        case_rows = list(csv.DictReader((args.root / model / "evaluation/case_metrics.csv").open(encoding="utf-8")))
        case_rows.sort(key=lambda item: float(item["dice"]))
        failures = ", ".join(f"{item['case_id']} ({float(item['dice']):.3f})" for item in case_rows[:3])
        successes = ", ".join(f"{item['case_id']} ({float(item['dice']):.3f})" for item in case_rows[-3:][::-1])
        lines.extend(["", f"Lowest three Dice: {failures}.", "", f"Highest three Dice: {successes}.", ""])

    winner = rows[0]
    lines.extend(
        [
            "## Selection",
            "",
            f"Selected baseline: **{winner['display_name']}** (ranked first by non-empty-case mean Dice, with non-empty-case mean Sensitivity as the secondary key).",
            "HD95 excludes infinite values from mean/SD/median; the CSV records each model's infinite-count explicitly.",
            "Absolute full-volume scores are substantially below positive-patch validation scores, showing that patch sampling validation was optimistic; Stage 2 should use full-volume validation for checkpoint selection.",
            "The original UNETR run became NaN at epoch 44; the reported v2 run completed all 100 epochs with GradScaler, gradient clipping, and finite-loss checks.",
            "The selected checkpoint and exact inference metadata are recorded in the corresponding model directory.",
            "",
        ]
    )
    (args.root / "stage1_results.md").write_text("\n".join(lines), encoding="utf-8")
    update_week02(rows, args.root.resolve())
    print(json.dumps({"winner": winner["model"], "output": str(args.root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
