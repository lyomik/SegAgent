# Stage 1：Baseline 模型篩選

## 實驗目的

Stage 1 的目標是在統一的資料與訓練條件下，比較五種不同架構家族的醫學影像分割模型，選出整體表現最佳的模型，作為 Stage 2 的 baseline backbone。

## 資料集與任務定義

- **資料集：** LIDC-IDRI
- **影像模態：** 胸部 CT
- **目標任務：** 肺結節分割
- **資料切分：** 以病人為單位建立 training／validation／test split；同一病人的影像、結節與標註不得跨 split。
- **標註處理：** LIDC-IDRI 的同一結節可能包含多位放射科醫師的輪廓標註。正式實驗前需固定標註整合規則，並對所有模型使用同一版本的 ground truth。

尚待 Week 01 固定的項目包括：結節納入／排除條件、多位標註的共識生成方式、CT intensity window、spacing、crop／patch 規則、2D／3D 輸入形式，以及 split 比例與 random seed。

## 候選模型

| 模型 | 架構家族 | 架構特性 |
| --- | --- | --- |
| nnU-Net v2 | Self-configuring CNN | 可依資料集特性自動配置前處理、網路與訓練策略 |
| MedNeXt | CNN | 採用現代化卷積設計，強調多尺度特徵建模 |
| SegFormer | Transformer | 使用階層式 Transformer encoder 與輕量化 decoder |
| Swin-UNETR | Hierarchical Transformer | 結合 Swin Transformer encoder 與 U-shaped decoder |
| PVFormer | Hybrid | 結合卷積的局部特徵與 Transformer 的全域資訊 |

## 控制變因

為確保比較公平，所有候選模型應盡可能採用一致的實驗設定：

- 使用相同的 LIDC-IDRI training、validation 與 test split。
- 使用相同的影像前處理流程。
- 使用相同的資料增強策略。
- 使用相同的訓練 epoch 數與 early-stopping 規則。
- 使用相同的模型選擇、推論及後處理方式。
- 使用相同的評估程式與指標計算方式。

若特定模型因架構限制而必須使用不同設定，需在實驗紀錄中明確註記，並說明差異與原因。

## 評估指標

### 主要指標

- **Dice score**：衡量預測區域與標註區域的重疊程度。
- **Sensitivity**：衡量模型找出實際病灶的能力，降低漏診風險。

### 次要指標

- **HD95**：衡量預測邊界與真實邊界之間的距離，數值越低越好。
- **Precision**：衡量模型預測為病灶的區域中，實際為病灶的比例。
- **Specificity**：衡量模型正確排除非病灶區域的能力。

## 模型選擇原則

1. 以 validation set 的 Dice 與 Sensitivity 作為主要排名依據。
2. 當主要指標表現接近時，依序參考 HD95、Precision 與 Specificity。
3. 同時記錄模型參數量、推論時間與 GPU 記憶體用量，作為效能相近時的選擇依據。
4. 僅使用 test set 對最終選定模型進行一次獨立評估，避免以 test set 反覆調整模型。

## 預期產出

- 五種候選模型在相同資料切分下的完整量化結果。
- 各模型的優缺點與失敗案例分析。
- 一個勝出的 baseline 模型，供 Stage 2 加入不同 attention module。

## 結果紀錄

| 模型 | Dice ↑ | Sensitivity ↑ | HD95 ↓ | Precision ↑ | Specificity ↑ | 備註 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| nnU-Net v2 | TBD | TBD | TBD | TBD | TBD | |
| MedNeXt | TBD | TBD | TBD | TBD | TBD | |
| SegFormer | TBD | TBD | TBD | TBD | TBD | |
| Swin-UNETR | TBD | TBD | TBD | TBD | TBD | |
| PVFormer | TBD | TBD | TBD | TBD | TBD | |

**Stage 1 勝出模型：** TBD
