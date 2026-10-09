# Week 08 TODO｜11/27～12/03

**階段：** Stage 3 Level 3－Agent 初步評估  
**週目標：** 完成 Agent 全 validation 評估與 Stage 3 初版。  
**輸入：** 鎖定的 Level 1、Level 2 與 Agent v1  
**完成狀態：** ⬜ 未開始

## 1. 全資料執行

- [ ] 對完整 validation set 產生 Agent 輸入摘要。
- [ ] 執行 Agent 並驗證每筆輸出符合 schema。
- [ ] 對失敗、timeout 或非法輸出套用固定 fallback。
- [ ] 保存每筆 Agent 輸入、輸出、reason codes 與版本。
- [ ] 重新執行指定案例以檢查 decision stability。
- [ ] 計算 Agent 額外執行時間與失敗率。

## 2. 仲裁品質評估

- [ ] 計算 Level 3 分割 metrics。
- [ ] 計算 selection accuracy。
- [ ] 計算 per-case oracle 與 oracle gap。
- [ ] 計算 review rate。
- [ ] 計算 failure capture rate。
- [ ] 計算重複執行的一致率。
- [ ] 與最佳 expert、Level 1、Level 2 比較。
- [ ] 依病灶大小與 disagreement 分組分析。

## 3. Agent Ablation

- [ ] 移除單一 expert 統計資訊後重跑。
- [ ] 移除 Level 1 資訊後重跑。
- [ ] 移除 Level 2 規則紀錄後重跑。
- [ ] 移除 case-level summary 後重跑。
- [ ] 比較各類資訊對選擇與最終 segmentation 的影響。
- [ ] 分析 Agent 選錯來源的案例。
- [ ] 確認 Agent 是否過度偏好特定候選。

## 4. 整合與文件

- [ ] 建立單一指令的 Stage 1～3 inference pipeline。
- [ ] 完成 Stage 3 主結果表與 ablation 表。
- [ ] 完成方法流程圖。
- [ ] 整理至少三類代表性案例：成功、無改善、失敗。
- [ ] 列出 Final Stage 的效果瓶頸與優先處理順序。

## 週末驗收

- [ ] Agent 全 validation 結果可重現。
- [ ] Stage 3 三個 levels 有完整比較。
- [ ] 已清楚列出 Final Stage 要改善的問題。

**本週產物：** Agent validation outputs、Level 3 metrics、ablation 表、流程圖、瓶頸清單。  
**阻塞／備註：**

