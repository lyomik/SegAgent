# Week 09 TODO｜12/04～12/10

**階段：** Final Stage－效果診斷與第一輪最佳化  
**週目標：** 找出系統瓶頸，完成有明確改善證據的 Demo v1。  
**輸入：** Stage 3 初版、所有 validation case-level 結果  
**完成狀態：** ⬜ 未開始

## 1. 系統化錯誤分析

- [ ] 依病灶大小將 validation cases 分組。
- [ ] 依病灶數量與 connected components 分組。
- [ ] 依 expert disagreement 程度分組。
- [ ] 將錯誤分類為漏檢、假陽性、邊界錯誤與仲裁錯誤。
- [ ] 統計各錯誤類型的數量與指標影響。
- [ ] 找出 Level 1、Level 2、Level 3 各自最常失敗的條件。
- [ ] 建立按影響程度排序的改善清單。
- [ ] 選出本週最多三項優先改善目標。

## 2. 第一輪最佳化

- [ ] 重新檢查 Level 1 權重搜尋範圍。
- [ ] 調整 Level 2 threshold 與規則優先順序。
- [ ] 檢查 confidence 與實際正確率的關係。
- [ ] 評估是否需要 confidence calibration。
- [ ] 調整 Agent case summary 或 reason codes。
- [ ] 對每項改動建立獨立 config，不直接覆寫 Stage 3 初版。
- [ ] 每次只改一類因素並記錄相對改善。
- [ ] 確認改善在完整 validation set 上成立。
- [ ] 檢查改善是否造成特定子群明顯退步。

## 3. Demo v1

- [ ] 建立單案例輸入到最終結果的展示流程。
- [ ] 顯示原圖、各 expert mask、融合結果與 Agent 最終選擇。
- [ ] 顯示 confidence、disagreement、觸發規則與選擇理由。
- [ ] 選擇成功、改善與仍失敗的代表性案例。
- [ ] 統一影像色彩、切面、縮放與 mask overlay。
- [ ] 測量完整 demo 的執行時間。
- [ ] 準備預先產生的結果作為初步備援。

## 4. 文件與紀錄

- [ ] 完成瓶頸分析表。
- [ ] 更新最佳化前後的指標比較。
- [ ] 記錄保留與捨棄的調整及理由。
- [ ] 更新論文 Error Analysis 初稿。

## 週末驗收

- [ ] 至少一項主要瓶頸在 validation set 上獲得改善。
- [ ] Demo v1 能完整展示 Stage 1～3 流程。
- [ ] 所有調整均未使用 test set。

**本週產物：** 錯誤分類表、優先改善清單、最佳化 configs、Demo v1、前後比較表。  
**阻塞／備註：**

