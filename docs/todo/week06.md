# Week 06 TODO｜11/13～11/19

**階段：** Stage 3 Level 2－Rule-Based System  
**週目標：** 完成特徵計算與 Rule-based v1。  
**輸入：** 四個 experts 與 Level 1 validation outputs  
**完成狀態：** ⬜ 未開始

## 1. Per-expert 特徵

- [ ] 定義 confidence 的計算方式。
- [ ] 計算病灶內平均、分位數與最大 probability。
- [ ] 計算 lesion volume。
- [ ] 計算 connected components 數量與各元件體積。
- [ ] 計算 probability mean、standard deviation 與 entropy。
- [ ] 對空預測定義一致且不會報錯的特徵值。
- [ ] 將每個 expert 的特徵輸出成結構化表格。

## 2. Cross-expert 特徵

- [ ] 計算所有 expert pairs 的 Dice。
- [ ] 計算所有 expert pairs 的 IoU。
- [ ] 計算全體 masks 的 union／intersection ratio。
- [ ] 定義並計算 voxel-level disagreement。
- [ ] 定義 case-level disagreement summary。
- [ ] 驗證特徵對 expert 順序不敏感。
- [ ] 視覺化特徵分布與異常值。

## 3. Rule-Based v1

- [ ] 由 validation 分布制定高／低 confidence 範圍。
- [ ] 制定 lesion volume 異常規則。
- [ ] 制定過多微小 connected components 規則。
- [ ] 制定高一致性與高分歧規則。
- [ ] 定義規則優先順序與衝突處理。
- [ ] 定義 fallback 與 review required 條件。
- [ ] 為每個案例輸出觸發規則、選擇來源與理由。
- [ ] 完成 v1 validation 評估並與 Level 1 比較。

## 4. 文件與紀錄

- [ ] 建立 feature dictionary，記錄名稱、公式與範圍。
- [ ] 繪製規則流程圖。
- [ ] 保存 threshold 與規則設定檔。
- [ ] 整理 v1 成功與失敗案例。

## 週末驗收

- [ ] 每個 validation case 均有完整特徵。
- [ ] Rule-based v1 能穩定輸出結果與觸發紀錄。
- [ ] v1 與 Level 1 可使用同一評估流程比較。

**本週產物：** feature table、feature dictionary、rules v1 config、觸發紀錄與 v1 結果。  
**阻塞／備註：**

