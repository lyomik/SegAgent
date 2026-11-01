# SegAgent

SegAgent 是一個以 LIDC-IDRI 進行肺結節分割的可重現研究專案。專案將共用的
資料集建構與前處理流程，和各模型專屬的轉接器分離，確保所有受評模型都使用
等價的輸入與真實標註。

## 專案結構

- `configs/`：資料集、模型、注意力模組與各階段實驗設定。
- `data/`：原始、標準化與處理後的資料集（不納入 Git）。
- `src/segagent/`：可重用的資料、模型、訓練、評估、融合與代理程式碼。
- `scripts/`：命令列入口與各階段流程控制程式。
- `experiments/`：實驗執行輸出（不納入 Git）。
- `checkpoints/`：模型權重（不納入 Git）。
- `notebooks/`：探索性分析與視覺化筆記本。
- `docs/`：規格、計畫與論文筆記。
- `system/`：後續的 SegAgent 應用層。

前處理契約記錄於
[`docs/Stage1/Preprocess_Spec.md`](docs/Stage1/Preprocess_Spec.md)。
基準模型註冊、設定歸屬與檢查點目錄結構記錄於
[`docs/model_management.md`](docs/model_management.md)。

## 環境設定

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

可使用下列命令續傳 LIDC-IDRI：

```bash
python scripts/download_lidc_idri.py \
  --metadata data/raw/LIDC-IDRI/LIDC-IDRI_MetaData.csv \
  --output data/raw/LIDC-IDRI/dicom
```
