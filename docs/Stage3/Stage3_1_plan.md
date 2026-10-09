# Stage 3－Level 1：固定式 Ensemble

## 實驗目的

Level 1 建立不依個案動態調整的固定式 ensemble baseline，用來評估多個 experts 的基礎融合效果，並作為 Level 2 rule-based system 與 Level 3 agent 的比較基準。

## Expert 組成

| Expert | 模型配置 |
| --- | --- |
| expert 0 | Stage 1 勝出的 baseline |
| expert 1 | baseline + attention 1 |
| expert 2 | baseline + attention 2 |
| expert 3 | baseline + attention 3 |

其中 attention 1～3 為 Stage 2 選出的三種配置。所有 experts 必須使用相同的資料切分、類別定義、輸入格式與輸出空間。

## 固定融合方法

### 1. Hard voting

先將每個 expert 的輸出轉為離散 segmentation mask，再對每個 voxel／pixel 進行多數決。若票數相同，需預先定義 tie-breaking 規則，例如採用平均機率或指定優先 expert。

### 2. Probability average

對四個 experts 的 class probability map 取算術平均，再以固定 threshold 或 `argmax` 產生最終 segmentation：

$$
p_{ensemble}(x)=\frac{1}{N}\sum_{i=1}^{N}p_i(x)
$$

其中 $N=4$，$p_i(x)$ 為第 $i$ 個 expert 對位置 $x$ 的預測機率。

### 3. Weighted probability average

依據 validation set 表現為每個 expert 指定固定權重：

$$
p_{ensemble}(x)=\sum_{i=1}^{N}w_i p_i(x), \qquad
w_i \ge 0, \qquad \sum_{i=1}^{N}w_i=1
$$

權重只能使用 training／validation data 決定，不得根據 test set 調整。

## 評估方式

- 使用 Dice、Sensitivity、HD95、Precision 與 Specificity 比較三種融合方法。
- 同時與四個單一 experts 的結果比較，確認 ensemble 是否帶來穩定增益。
- 記錄不同病灶大小下的表現，觀察固定融合是否偏向特定案例。
- 保存各 expert 的 probability map、離散 mask 與最終 ensemble mask，供 Level 2 與 Level 3 使用。

## 結果紀錄

| 方法 | Dice ↑ | Sensitivity ↑ | HD95 ↓ | Precision ↑ | Specificity ↑ | 備註 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| expert 0 | TBD | TBD | TBD | TBD | TBD | Single model |
| expert 1 | TBD | TBD | TBD | TBD | TBD | Single model |
| expert 2 | TBD | TBD | TBD | TBD | TBD | Single model |
| expert 3 | TBD | TBD | TBD | TBD | TBD | Single model |
| Hard voting | TBD | TBD | TBD | TBD | TBD | |
| Probability average | TBD | TBD | TBD | TBD | TBD | |
| Weighted probability average | TBD | TBD | TBD | TBD | TBD | |

**Level 1 最佳固定融合方法：** TBD
