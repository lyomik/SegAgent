# Stage 3－Level 2：Rule-Based Ensemble

## 實驗目的

Level 2 根據單一 expert 的預測品質資訊，以及 experts 之間的一致性，為不同個案選擇或融合較可靠的結果。此階段使用預先定義的規則，不進行自主決策或線上學習。

## 規則所需資訊

### Per-expert 特徵

針對每個 expert 計算：

- **Confidence：** 病灶區域內的平均／分位數預測機率，或模型整體信心摘要。
- **Lesion volume：** 預測病灶的總體積。
- **Connected components：** 預測病灶的連通區塊數量及各區塊大小。
- **Probability statistics：** 最大值、平均值、標準差、entropy 等機率分布資訊。

### Cross-expert 特徵

比較不同 experts 的預測結果：

- **Pairwise Dice：** 任兩個 expert segmentation mask 的重疊程度。
- **Pairwise IoU：** 任兩個 expert 的 intersection over union。
- **Union／intersection ratio：** 所有預測聯集與交集的體積比例。
- **Disagreement：** experts 在 voxel／pixel 或病灶層級上的分歧程度。

## 規則設計原則

1. **高一致性案例：** 當 experts 間的 Pairwise Dice／IoU 高且 disagreement 低時，採用 Level 1 表現最佳的固定融合結果。
2. **低信心案例：** 當整體 confidence 過低時，標記為不確定案例，避免直接接受單一 expert 的結果。
3. **體積異常案例：** 當某個 expert 的 lesion volume 明顯偏離其他 experts 時，降低其權重或排除該預測。
4. **離散假陽性案例：** 當某個 expert 產生大量微小 connected components 時，可降低其權重或先進行規範化後處理。
5. **高分歧案例：** 當 union／intersection ratio 或 disagreement 過高時，選擇 validation set 上較可靠的 expert、使用保守融合策略，或標記為需進一步仲裁。

所有 threshold、權重與規則優先順序皆須由 training／validation data 決定，test set 僅用於最終評估。

## 規則流程

```text
輸入四個 experts 的 probability maps 與 masks
                    │
                    ▼
計算 per-expert 與 cross-expert 特徵
                    │
                    ▼
檢查信心、體積、連通區塊與一致性規則
                    │
                    ▼
選擇 expert／調整權重／套用固定融合／標記不確定案例
                    │
                    ▼
輸出最終 segmentation 與規則觸發紀錄
```

## 評估方式

- 與最佳單一 expert 及 Level 1 最佳固定融合方法比較。
- 使用 Dice、Sensitivity、HD95、Precision 與 Specificity 評估最終 segmentation。
- 額外統計各規則的觸發次數、觸發後效能變化與不確定案例比例。
- 進行 rule ablation，逐一移除規則以分析各項特徵的貢獻。

## 結果紀錄

| 方法／規則版本 | Dice ↑ | Sensitivity ↑ | HD95 ↓ | Precision ↑ | Specificity ↑ | 不確定案例數 | 備註 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Level 1 最佳方法 | TBD | TBD | TBD | TBD | TBD | N/A | Baseline |
| Rule-based v1 | TBD | TBD | TBD | TBD | TBD | TBD | |
| Rule-based v2 | TBD | TBD | TBD | TBD | TBD | TBD | |

**Level 2 最佳規則配置：** TBD
