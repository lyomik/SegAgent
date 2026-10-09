# Week 01 TODO｜10/09～10/15

**階段：** 共用 Pipeline 與 Stage 1 啟動  
**週目標：** 固定公平比較規格，讓五個 baseline 全部通過 smoke test 並開始正式訓練。  
**輸入：** LIDC-IDRI 原始 CT、結節標註、五個候選模型實作  
**完成狀態：** ⬜ 未開始

## 1. 資料與切分

- [x] 確定使用 LIDC-IDRI，任務為肺結節分割。
- [ ] 確認資料下載完整性、版本與原始檔案清單。
- [ ] 盤點影像數量、標註數量、類別比例與缺失資料。
- [ ] 固定結節納入／排除條件，以及多位放射科醫師標註的整合規則。
- [ ] 以病人為單位建立 training／validation／test split。
- [ ] 固定 split 檔案並記錄 random seed。
- [ ] 檢查同一病人的資料未跨 split。
- [ ] 抽查影像方向、spacing、尺寸、label mapping 與空標註。
- [ ] 記錄資料排除條件及最終納入數量。

## 2. 共用訓練與評估 Pipeline

- [ ] 固定前處理、normalization、resampling 與 crop／patch 規則。
- [ ] 固定資料增強、loss、optimizer、learning rate 與 epoch。
- [ ] 實作 Dice、Sensitivity、HD95、Precision、Specificity。
- [ ] 為空預測與空標註案例定義一致的 metric 處理方式。
- [ ] 統一 config、checkpoint、log、prediction 與 metrics 目錄結構。
- [ ] 在輸出中保存模型名稱、seed、資料版本與完整設定。

## 3. Baseline 啟動

- [ ] 為 nnU-Net v2 建立設定並完成 smoke test。
- [ ] 為 MedNeXt 建立設定並完成 smoke test。
- [ ] 為 SegFormer 建立設定並完成 smoke test。
- [ ] 為 Swin-UNETR 建立設定並完成 smoke test。
- [ ] 為 PVFormer 建立設定並完成 smoke test。
- [ ] 檢查五個模型輸入、輸出尺寸及類別數一致。
- [ ] 排定正式訓練 queue 並啟動可執行的模型。

## 4. 文件與紀錄

- [ ] 更新 Stage 1 的固定設定與資料切分。
- [ ] 建立實驗追蹤表及 run ID。
- [ ] 撰寫論文的 Dataset、Preprocessing 與 Metrics 初稿。
- [ ] 記錄本週錯誤、修正方式與尚未解決的風險。

## 週末驗收

- [ ] 五個模型皆完成 end-to-end smoke test。
- [ ] 共用評估程式能由預測檔產生結果表。
- [ ] 正式訓練已啟動，且剩餘時間可在 Week 02 內完成。
- [ ] 所有設定與 split 皆可重現。

**本週產物：** split 檔、共用 config、評估程式、五份 smoke-test log、實驗追蹤表。  
**阻塞／備註：**

