# Week 02 TODO｜10/16～10/22

**階段：** Stage 1 Baseline 篩選  
**週目標：** 完成五種 baseline 比較並鎖定 Stage 2 backbone。  
**輸入：** Week 01 共用 pipeline、五個 baseline runs  
**完成狀態：** ⬜ 未開始

## 1. 訓練完成與品質檢查

- [ ] 完成 nnU-Net v2 正式訓練與 validation inference。
- [ ] 完成 MedNeXt 正式訓練與 validation inference。
- [ ] 完成 SegFormer 正式訓練與 validation inference。
- [ ] 完成 Swin-UNETR 正式訓練與 validation inference。
- [ ] 完成 PVFormer 正式訓練與 validation inference。
- [ ] 檢查所有 run 無 NaN、崩潰、資料錯置或 checkpoint 遺失。
- [ ] 對失敗 run 排錯並補跑一次；保留失敗原因紀錄。

## 2. 統一評估

- [ ] 產生每個模型的 case-level metrics。
- [ ] 計算各指標平均值、標準差與中位數。
- [ ] 核對 Dice、Sensitivity、HD95、Precision、Specificity。
- [ ] 記錄參數量、最高 GPU 記憶體與單案例推論時間。
- [ ] 依病灶大小分組檢視表現。
- [ ] 各抽查至少三個成功與三個失敗案例。
- [ ] 確認模型比較皆使用相同 validation cases。

## 3. 選模與封版

- [ ] 以 Dice、Sensitivity 為主要依據建立排名。
- [ ] 以 HD95、Precision、Specificity處理表現相近情況。
- [ ] 比較訓練穩定性與資源成本。
- [ ] 對前二名進行必要的 seed 複驗或結果確認。
- [ ] 選出 Stage 1 勝出 baseline。
- [ ] 鎖定 checkpoint、config、preprocessing 與 inference command。
- [ ] 將未入選模型標記為封存，不再占用 Stage 2 資源。

## 4. 文件與紀錄

- [ ] 填寫 Stage 1 完整結果表。
- [ ] 整理模型選擇理由與限制。
- [ ] 製作一張 baseline 定量比較圖。
- [ ] 更新論文 Stage 1 實驗設定與初步結果。

## 週末驗收

- [ ] 五個模型均有有效且可追溯的 validation 結果。
- [ ] 勝出 baseline 有可重現的 checkpoint 與 inference 流程。
- [ ] Stage 2 可直接沿用固定 protocol 啟動。

**本週產物：** Stage 1 結果表、case-level CSV、比較圖、勝出模型 checkpoint 與選模紀錄。  
**阻塞／備註：**

