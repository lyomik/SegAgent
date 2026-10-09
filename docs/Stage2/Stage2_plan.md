# Stage 2：Attention Module 比較

## 實驗目的

Stage 2 以 Stage 1 勝出的模型作為固定 baseline，分別加入五種 attention module，比較不同注意力機制對分割表現的影響，並選出適合進入 Stage 3 ensemble 的模型版本。

## Attention 候選配置

| Attention module | 類型 | 主要作用 |
| --- | --- | --- |
| SE | Channel attention | 重新校準各 channel 的重要性 |
| CBAM | Channel + spatial attention | 依序建模 channel 與 spatial 資訊 |
| ECA | Lightweight channel attention | 以低額外成本建立局部跨 channel 互動 |
| CA | Channel + spatial attention | 同時保留通道關係與位置資訊 |
| MHSA | Global self-attention | 建模長距離與全域空間關係 |

另保留一組未加入 attention 的 baseline 作為 control group。

## 實驗設計

- **固定 backbone：** 使用 Stage 1 的勝出模型。
- **固定訓練流程：** 沿用 Stage 1 的 dataset split、前處理、資料增強、epoch、loss、optimizer、learning-rate schedule、推論及評估方式。
- **單一變因：** 每次實驗只替換 attention module。
- **插入位置：** 依 backbone 架構選擇合理位置；同一 attention module 在所有重複實驗中必須保持一致。
- **重複實驗：** 使用相同 random seeds 或相同次數的重複實驗，回報平均值與標準差。

> 不同 attention module 的結構與計算方式不同，因此插入位置不必完全相同，但必須事先定義並完整記錄，避免依測試結果反覆調整。

## 評估方式

- 主要指標：Dice、Sensitivity。
- 次要指標：HD95、Precision、Specificity。
- 效率指標：參數量、FLOPs、推論時間與 GPU 記憶體用量。

除整體測試結果外，亦比較每個 attention 版本相對於 baseline 的變化量，確認效能提升是否足以抵銷額外計算成本。

## Stage 3 Expert 選擇原則

1. `expert 0` 固定為未加入 attention 的 baseline。
2. 依 validation set 的 Dice 與 Sensitivity，從五個 attention 版本中選出三個互補且表現穩定的版本，作為 `expert 1`～`expert 3`。
3. 若模型表現接近，優先考慮錯誤型態具有互補性者，而非只依單一平均分數排序。
4. Expert 組合確認後，不得依 test set 結果更換成員或調整設定。

## 預期產出

- Baseline 與五種 attention 配置的完整比較結果。
- Attention 插入位置與額外計算成本紀錄。
- Stage 3 使用的四個 experts：一個 baseline 與三個 attention-enhanced models。

## 結果紀錄

| 配置 | Attention 類型 | Dice ↑ | Sensitivity ↑ | HD95 ↓ | Precision ↑ | Specificity ↑ | 參數量 | 備註 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | None | TBD | TBD | TBD | TBD | TBD | TBD | expert 0 |
| Baseline + SE | Channel | TBD | TBD | TBD | TBD | TBD | TBD | |
| Baseline + CBAM | Channel + spatial | TBD | TBD | TBD | TBD | TBD | TBD | |
| Baseline + ECA | Lightweight channel | TBD | TBD | TBD | TBD | TBD | TBD | |
| Baseline + CA | Channel + spatial | TBD | TBD | TBD | TBD | TBD | TBD | |
| Baseline + MHSA | Global self-attention | TBD | TBD | TBD | TBD | TBD | TBD | |

**入選 Stage 3 的 attention 配置：** TBD、TBD、TBD
