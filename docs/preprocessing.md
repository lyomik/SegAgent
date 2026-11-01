# 前處理

正式的前處理契約維護於
[`Stage1/Preprocess_Spec.md`](Stage1/Preprocess_Spec.md)。進行受控實驗前，
必須在 `configs/dataset/lung_nodule.yaml` 中鎖定所有實際數值。

## 執行順序

前處理分成四個可獨立重跑且可稽核的步驟：

1. **原始資料稽核：** 驗證下載完成標記、DICOM 可讀性、XML 壓縮檔、UID 對應、切片連續性與空間資訊。
2. **標準資料建構：** 依 XML 的 `SeriesInstanceUID` 選取 CT，建立多位醫師共識遮罩，輸出原始空間的 NIfTI 與中繼資料。
3. **病人層級切分：** 先完成標註與品質檢查，再以病人為單位產生固定的 70%／15%／15% 訓練、驗證與測試切分。
4. **共用前處理：** 統一方向、體素間距、HU 範圍與強度尺度；模型轉接器只處理張量排列、分塊尺寸與批次大小。

## 第一版固定政策

- 只納入切片厚度不超過 2.5 mm、切片連續且間距一致的 CT series。
- 只將直徑至少 3 mm，且獲至少三位放射科醫師接受的結節列為目標。
- 同一結節的遮罩採 50% 體素共識；另外保留各結節的實例遮罩，語意分割標籤則取所有合格結節的聯集。
- 結節先依輪廓最短距離聚類；若相鄰結節形成超過四份標註的連通群，改用受限單連結聚類，保證同一位醫師在一個結節群中最多只有一份標註。
- 影像統一為 RAS、1 mm 等向性間距，HU 截切為 `[-1000, 400]`，再固定縮放至 `[0, 1]`。
- 訓練可使用真實標註抽取正樣本分塊；驗證與測試必須對完整 CT 進行滑動視窗推論，不得使用真實標註定位 ROI。
- 評估前將預測結果還原至標準資料的原始幾何，並以毫米計算 HD95。

這項納入政策與 LUNA16 的主要資料條件一致，方便後續與公開研究比較；實際納入案例數仍以本專案品質檢查報告為準。

2026-10-18 的完整幾何稽核確認：1,018 個 CT series 中有 888 個通過上述條件，對應 887 位病人。其餘 130 個 series 因切片厚度超標、切片位置重複或間距不連續而排除。完整紀錄位於 `experiments/stage1/data_audit/20261018-data-audit-v1/`。

標註盤點得到 1,188 個符合至少三位醫師條件的結節，分布於 602 個 series；另有 286 個合格 CT series 沒有符合條件的結節。固定切分使用 seed 42，病人數為訓練 621、驗證 133、測試 133，series 數為 621、133、134。切分檔為 `data/canonical/splits/patient_split.json`。

## 執行方式

```bash
# 解析標註並建立固定病人切分
PYTHONPATH=src python scripts/prepare_dataset.py split

# 建構全部標準資料；可用 --split 或 --limit 分批續跑
PYTHONPATH=src python scripts/prepare_dataset.py build

# 執行固定前處理；正式重跑既有案例時明確加上 --overwrite
PYTHONPATH=src python scripts/preprocess.py
```

病人切分、標註摘要與 XML 版本決策分別保存在 `data/canonical/splits/patient_split.json`、`data/canonical/metadata/annotation_summary.csv` 與 `data/canonical/metadata/annotation_revisions.json`。標準資料及前處理資料皆以 `metadata/cases.json` 作為可續跑的 manifest。

2026-10-19 的跨切分冒煙測試共處理 7 例：train 3、validation 2、test 2，包含 4 例陽性及 3 例空標註。所有輸出皆為 RAS、1 mm 等向性、影像值域 `[0, 1]`，且影像與標籤幾何一致。

## 評估指標政策

`scripts/evaluate.py` 由 NIfTI 預測與標籤計算 Dice、Sensitivity、HD95（mm）、Precision 與 Specificity，並輸出逐案例 CSV 與彙總 JSON。預測與標註都為空時，Dice、Sensitivity、Precision 設為 1，HD95 設為 0；只有其中一者為空時 Dice 設為 0、HD95 設為無限大，彙總時另外計數而不納入有限值平均。

## 實驗紀錄

資料稽核記錄存放於：

```text
experiments/stage1/data_audit/<run_id>/
├── run.json
├── logs/
├── reports/
└── artifacts/
```

前處理規劃與試跑記錄存放於：

```text
experiments/stage1/preprocessing/<run_id>/
├── run.json
├── logs/
├── reports/
└── artifacts/
```

資料建構及評估分別存放於 `experiments/stage1/dataset_preparation/<run_id>/` 與 `experiments/stage1/evaluation/<run_id>/`，結構相同。

每次重跑必須使用新的 `run_id`，並保存資料設定快照、Git commit、輸入資料摘要、排除清單與品質檢查結果，不得覆寫既有紀錄。
