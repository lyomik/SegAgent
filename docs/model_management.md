# 基準模型管理

第一階段以 `configs/models/baselines.yaml` 作為基準模型的唯一正式清單。
各模型的 YAML 檔負責架構與初始化設定；資料集建構和共用前處理則統一保留在
`configs/dataset/lung_nodule.yaml`。

## 已註冊的基準模型

| 固定代號 | 執行後端 | 輸入形式 | 初始化方式 |
| --- | --- | --- | --- |
| `nnunet` | nnU-Net v2 外部流程 | 3D | 完成資料規劃後從頭訓練 |
| `mednext` | MONAI MedNeXt-S | 3D | 從頭訓練 |
| `segformer` | Transformers MiT-B0 | 2D | 本機 ImageNet 編碼器權重 |
| `swin_unetr` | MONAI Swin-UNETR | 3D | 本機自監督 3D CT 編碼器權重 |
| `unetr` | MONAI UNETR | 3D | 從頭訓練 |

每個模型都接收單通道 CT，並輸出背景與結節兩個類別的 logits。SegFormer
會在模型內部複製灰階通道，以保留預訓練的三通道分塊嵌入層。這屬於模型
轉接器操作，不是共用的強度前處理。

## 設定歸屬

- `configs/models/baselines.yaml`：模型成員、固定代號與任務輸出契約。
- `configs/models/<name>.yaml`：執行後端、架構、初始化與檢查點政策。
- `configs/dataset/lung_nodule.yaml`：標準資料集建構與所有共用前處理。
- `experiments/stage1/<model>/<run_id>/`：鎖定的設定快照、日誌、指標與預測。
- `checkpoints/stage1/<model>/<run_id>/`：訓練產生的檢查點。
- `checkpoints/pretrained/<model>/`：唯讀的上游初始化權重。

不得覆寫實驗目錄，也不得以不同設定重複使用同一個 `run_id`。一般實驗只保留
`best` 和 `last` 檢查點；鎖定供正式比較的檢查點時，必須記錄 SHA-256 摘要。

## 管理命令

```bash
python scripts/manage_models.py list
python scripts/manage_models.py validate
python scripts/manage_models.py validate --runtime
python scripts/manage_models.py validate --strict
python scripts/smoke_test_models.py
python scripts/train_all.py --queue-id 20261026-stage1-all-baselines-v1
python scripts/manage_models.py init-run mednext --run-id stage1-mednext-seed42
```

預設驗證會檢查模型註冊表、轉接器、設定檔和本機初始化權重，尚未安裝的後端
會標記為待處理。加入 `--runtime` 後，會依模型要求 PyTorch、MONAI、
Transformers 與 nnU-Net。非嚴格驗證允許 `patch_size`、`batch_size` 和
nnU-Net 資料集代號暫時未決定。模型冒煙測試和正式訓練前，必須以
`--runtime --strict` 作為檢查關卡。資料相依的數值只能在完成清洗後資料的
間距、尺寸分布與 GPU 記憶體測試後鎖定。

nnU-Net 維持獨立的外部流程，因為它必須先對匯出的標準資料集擷取特徵並進行
實驗規劃。其餘四個模型則由各自的轉接器在程式內建構。
