# Week 03 TODO｜10/23～10/29

**階段：** Stage 2 第一輪 Attention 實驗  
**週目標：** 完成 SE、CBAM、ECA 三種 attention 配置。  
**輸入：** 鎖定的 Stage 1 baseline  
**完成狀態：** ⬜ 未開始

## 1. 架構適配

- [ ] 確認 baseline 中可插入 attention 的候選位置。
- [ ] 定義 SE、CBAM、ECA 的插入位置與介面。
- [ ] 記錄三種版本新增的參數量與計算路徑。
- [ ] 確認除 attention 外沒有改動 training protocol。
- [ ] 為三種版本各完成 forward／backward smoke test。
- [ ] 驗證載入資料、輸出尺寸與 loss 計算正常。

## 2. 正式實驗

- [ ] 完成 baseline + SE 訓練。
- [ ] 完成 baseline + SE validation inference 與 metrics。
- [ ] 完成 baseline + CBAM 訓練。
- [ ] 完成 baseline + CBAM validation inference 與 metrics。
- [ ] 完成 baseline + ECA 訓練。
- [ ] 完成 baseline + ECA validation inference 與 metrics。
- [ ] 保存各版本最佳 checkpoint、config、log 與 prediction。

## 3. 初步比較

- [ ] 計算三個版本相對 baseline 的各指標差值。
- [ ] 比較參數量、GPU 記憶體與推論時間增幅。
- [ ] 確認提升不是由不同 checkpoint 選擇規則造成。
- [ ] 抽查新增 attention 後改善與退步的案例。
- [ ] 記錄三個版本的主要錯誤型態。

## 4. 文件與紀錄

- [ ] 畫出 attention 插入位置示意。
- [ ] 更新 Stage 2 結果表。
- [ ] 撰寫 SE、CBAM、ECA 的方法描述。
- [ ] 安排 Week 04 的 CA、MHSA 與複驗 queue。

## 週末驗收

- [ ] 3/5 attention 版本皆有完整 validation 結果。
- [ ] 所有版本只改動 attention mechanism。
- [ ] 結果與計算成本皆已進入統一表格。

**本週產物：** 三組 checkpoint、三組 predictions、初步比較表、attention 插入位置圖。  
**阻塞／備註：**

