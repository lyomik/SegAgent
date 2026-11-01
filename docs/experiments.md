# 實驗管理

納入版本控制的實驗定義存放於 `configs/experiments/`。產生的日誌、預測、
表格與圖形存放於 `experiments/`，且不納入 Git。

第一階段每次實驗以 `experiments/stage1/<model>/<run_id>/` 保存鎖定的設定快照
與輸出；模型參數則分開存放於 `checkpoints/stage1/<model>/<run_id>/`。使用
`python scripts/manage_models.py init-run <model> --run-id <id>` 建立兩個目錄。

資料相關實驗也歸屬第一階段，但與模型訓練分開保存：

- `experiments/stage1/data_audit/<run_id>/`：下載完整性、DICOM/XML 對應、空間資訊與排除清單。
- `experiments/stage1/dataset_preparation/<run_id>/`：標註解析、病人切分與標準資料建構。
- `experiments/stage1/preprocessing/<run_id>/`：設定快照、試跑日誌、品質檢查、處理時間與輸出摘要。
- `experiments/stage1/evaluation/<run_id>/`：逐案例指標、彙總結果與評估設定。

每個資料實驗目錄固定包含 `run.json`、`logs/`、`reports/` 與 `artifacts/`。每次執行使用新的 `run_id`，避免覆寫先前結果。

## 第一階段資料執行紀錄

| run_id | 類型 | 狀態 | 主要結果 |
|---|---|---|---|
| `20261018-data-audit-v1` | 資料稽核 | 完成 | 1,018 series 中納入 888；887 位病人 |
| `20261018-dataset-preparation-v1` | 標註與切分 | 完成 | 1,188 個合格結節；病人 621/133/133 |
| `20261018-preprocessing-smoke-v1` | 前處理冒煙測試 | 完成 | 1 個陽性案例完成 RAS、1 mm、HU 正規化 |
| `20261018-metrics-smoke-v1` | 評估冒煙測試 | 完成 | 1 個 identity 案例，五項指標符合預期 |
| `20261019-split-smoke-v1` | 跨切分資料冒煙測試 | 完成 | 7 例；涵蓋三個 split、4 例陽性與 3 例空標註 |
| `20261019-metrics-smoke-v1` | 評估冒煙測試 | 完成 | 7 個 identity 案例，包含空標註政策 |
| `20261019-runtime-gate-v1` | 模型環境檢查 | 阻塞 | 缺訓練依賴與資料相依參數；尚未執行模型 forward |
| `20261026-model-smoke-v3` | 五模型冒煙測試 | 完成 | 5/5 通過真實陽性 patch 的 forward、loss、backward 與 optimizer step |
| `20261026-mednext-development-v1` | MedNeXt 訓練迴圈 | 完成 | 使用 7 例 smoke 資料驗證訓練、驗證與 checkpoint 流程；不納入正式比較 |
| `20261026-stage1-all-baselines-v1` | 五模型正式訓練佇列 | 已排程 | 完整前處理後依序執行四個共用 trainer 模型與 nnU-Net v2 外部流程 |

## 模型冒煙測試

2026-10-26 已在專案 `.venv`（Python 3.12.15、PyTorch 2.14.1、CUDA 13.0）通過
runtime strict gate。五個 baseline 均以預處理後的陽性案例完成一次真實資料最佳化步驟，
並驗證輸入、兩類 logits、標籤尺寸、有限輸出與有限梯度。可重跑命令如下：

```bash
.venv/bin/python scripts/manage_models.py validate --runtime --strict
.venv/bin/python scripts/smoke_test_models.py
```

nnU-Net 的 smoke 使用其相依套件所提供的網路元件建立 compact 3D PlainConvUNet，並
驗證 `nnUNetv2_train 501 3d_fullres 0` 命令可解析。這不取代 Dataset501 的正式匯出、
fingerprint extraction 與 experiment planning；三項仍是正式訓練前置作業。
