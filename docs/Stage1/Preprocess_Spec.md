# SegAgent —  Preprocessing Specification

## 1. 必須統一的項目（Shared Preprocessing）

所有模型必須遵守相同的資料前處理規範，以確保 Stage 1 比較的公平性。

| 項目 | 統一規範 |
|---|---|
| Dataset | 使用相同的 LIDC-IDRI 資料 |
| CT Series Selection | 每個病例使用相同的 CT Series |
| DICOM Conversion | 統一轉換為 NIfTI，保留正確 HU 與空間資訊 |
| Ground Truth | 使用相同的結節標註整合方式與 Mask |
| Data Split | 相同 Patient-level Train / Val / Test |
| Orientation | 統一為 RAS |
| Target Spacing | 使用相同的 Voxel Spacing |
| Resampling | 使用相同的影像及 Mask 插值規則 |
| HU Clipping | 使用相同的 HU 範圍 |
| Intensity Normalization | 使用相同的 Normalization 方法與統計量 |
| Crop / ROI Policy | 使用相同的影像搜尋範圍與裁切規則 |

**原則：** 各模型可以使用自己的 Preprocessor 實作，但最終必須符合上述共同規範，避免重複 Resampling 或 Normalization。

## 2. 可自行設定的項目（Model-specific Configuration）

以下項目允許模型依自身 Architecture 與 Hardware Requirements 調整。

| 項目 | 自訂範圍 |
|---|---|
| Tensor Layout | 2D / 2.5D / 3D，依模型需求 |
| Patch Size | 優先統一，架構或記憶體限制時可調整 |
| Batch Size | 依 GPU Memory 調整 |
| Internal Padding | 依 Network Downsampling 需求 |
| Network Architecture | 保留模型原生設計 |
| Positional Embedding | Transformer 模型自行處理 |
| Gradient Checkpointing | 依記憶體需求使用 |
| Mixed Precision | FP32 / FP16 / BF16，依模型相容性 |

**限制：**

- Model-specific Adapter 不得自行改變已固定的 HU、Spacing、Normalization 或 Ground Truth。
- Patch Size 若不同，必須記錄實際 Physical Field of View。
- 不得在測試階段使用 Ground Truth 定位結節 ROI。
- 各模型自行設定的項目必須記錄於 Configuration，以確保實驗可重現。

## 3. 前處理架構

```text
Raw Dataset (DICOM + XML)
          │
          ▼
   Data Construction
   ├── QC / Series Selection
   ├── DICOM → NIfTI
   ├── Ground Truth Generation
   └── Patient-level Split
          │
          ▼
    Canonical Dataset
    (Image / Mask / Split / Metadata)
          │
          ▼
   Shared Preprocessing
   ├── Orientation
   ├── Target Spacing
   ├── Resampling
   ├── HU Clipping
   ├── Normalization
   └── Crop / ROI Policy
          │
          ▼
      Model Adapters
   ├── nnU-Net
   ├── MedNeXt
   ├── SegFormer
   ├── Swin-UNETR
   └── PVFormer
```

## 4. 核心原則

**Shared Preprocessing 決定資料的一致性；Model Adapter 只負責滿足不同架構的輸入與運算需求。**

nnU-Net 可以保留自適應的 Network Architecture、Batch Size、Patch Size 等配置，但在 Controlled Benchmark 中，不得覆寫共同定義的 Spacing、Intensity Normalization 與 Crop Policy。

所有模型最終接受相同的資料與標註，並遵守一致的前處理規範。