# Week 05 TODO｜11/06～11/12

**階段：** Stage 3 Level 1－固定式 Ensemble  
**週目標：** 完成三種固定融合方法並選出 Level 1 baseline。  
**輸入：** expert 0～3 checkpoints 與 prediction manifest  
**完成狀態：** ⬜ 未開始

## 1. Expert 輸出準備

- [ ] 對完整 validation set 執行四個 experts。
- [ ] 保存每個案例的 probability maps 與 discrete masks。
- [ ] 檢查四組輸出的 shape、spacing、orientation 與 case ID。
- [ ] 檢查 probability range、softmax／sigmoid 與 threshold 定義。
- [ ] 建立缺檔、損毀檔與推論失敗檢查。
- [ ] 驗證單一 expert metrics 可重現 Stage 2 結果。

## 2. 固定融合實作

- [ ] 實作 hard voting。
- [ ] 定義票數相同時的 tie-breaking 規則。
- [ ] 實作 probability average。
- [ ] 實作 weighted probability average。
- [ ] 使用 validation 資料搜尋或估計固定權重。
- [ ] 驗證所有權重非負且總和為 1。
- [ ] 統一 ensemble threshold 與後處理方式。

## 3. 評估與選擇

- [ ] 產生三種 ensemble 的 case-level metrics。
- [ ] 與四個單一 experts 比較主要及次要指標。
- [ ] 依病灶大小分析 ensemble 是否穩定改善。
- [ ] 比較 inference 時間與儲存需求。
- [ ] 抽查 experts 一致與分歧時的輸出。
- [ ] 選出 Level 1 最佳固定融合方法。
- [ ] 鎖定方法、權重、threshold 與程式版本。

## 4. 文件與紀錄

- [ ] 完成 Level 1 結果表。
- [ ] 製作單一 expert 與 ensemble 比較圖。
- [ ] 撰寫三種融合公式、設定與選擇理由。
- [ ] 保存後續 Level 2 所需的輸出 manifest。

## 週末驗收

- [ ] 三種固定融合可重現執行。
- [ ] 最佳 Level 1 方法已鎖定。
- [ ] 每個案例均有四個 expert 與 ensemble 輸出。

**本週產物：** 融合程式、權重設定、Level 1 結果表、所有 validation ensemble outputs。  
**阻塞／備註：**

