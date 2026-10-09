# Week 07 TODO｜11/20～11/26

**階段：** Level 2 定案與 Level 3 Agent 建立  
**週目標：** 鎖定規則系統並完成可執行的 Agent 初版。  
**輸入：** Rule-based v1、validation features、Level 1 outputs  
**完成狀態：** ⬜ 未開始

## 1. Level 2 改善與 Ablation

- [ ] 依 v1 失敗案例調整規則優先順序。
- [ ] 只使用 validation set 調整 threshold 與權重。
- [ ] 實作 Rule-based v2。
- [ ] 比較 v1、v2、Level 1 與最佳單一 expert。
- [ ] 分別移除 confidence、volume、component 與 disagreement 規則。
- [ ] 記錄每項規則的觸發次數與觸發後效能。
- [ ] 檢查規則是否過度針對少數案例。
- [ ] 選出並鎖定 Level 2 最終版本。

## 2. Agent 輸入與輸出設計

- [ ] 定義 Agent 可讀取的四類資訊。
- [ ] 排除 ground truth 及其衍生資訊。
- [ ] 定義候選來源：expert 0～3、Level 1、Level 2。
- [ ] 定義 case-level summary 欄位與單位。
- [ ] 定義 JSON output schema。
- [ ] 建立有限且可統計的 reason codes。
- [ ] 定義 confidence 與 review required 的意義。
- [ ] 準備正常、空病灶與高分歧測試案例。

## 3. Agent 初版

- [ ] 完成 Agent prompt／決策流程初版。
- [ ] 實作輸入組裝與 schema validation。
- [ ] 實作非法輸出重試或 fallback。
- [ ] 確保 Agent 不覆寫原始候選結果。
- [ ] 保存輸入、輸出、版本與執行時間。
- [ ] 對小批次案例完成端到端測試。
- [ ] 人工檢查選擇來源與理由是否一致。

## 4. 文件與紀錄

- [ ] 更新 Level 2 最終結果表。
- [ ] 撰寫 Agent 輸入、輸出與限制。
- [ ] 保存 rules final config 與 Agent v1 config。
- [ ] 列出 Week 08 必做的 Agent 評估項目。

## 週末驗收

- [ ] Level 2 規則與 threshold 已鎖定。
- [ ] Agent 能對小批次案例輸出合法 JSON。
- [ ] 每次選擇均有可追溯的輸入與 reason codes。

**本週產物：** Level 2 final config、rule ablation、Agent schema、prompt／決策設定與小批次輸出。  
**阻塞／備註：**

