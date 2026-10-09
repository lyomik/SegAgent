# Week 11 TODO｜12/18～12/24

**階段：** Final Stage－Demo 與實驗封版  
**週目標：** 完成教授 Demo、鎖定方法並取得一次最終 test 結果。  
**輸入：** Final candidate、Demo v2、完整 validation 證據  
**完成狀態：** ⬜ 未開始

## 1. Final Candidate 鎖定

- [ ] 確認 config A 的 validation 結果可重現。
- [ ] 確認所有模型 checkpoint checksum／版本。
- [ ] 鎖定 preprocessing、threshold、權重與規則。
- [ ] 鎖定 Agent prompt／決策設定與輸出 schema。
- [ ] 固定 random seeds 與軟體環境版本。
- [ ] 建立 final release tag／版本編號。
- [ ] 將所有鎖定檔案設為唯讀備份或建立不可覆寫副本。

## 2. 最終 Test Evaluation

- [ ] 僅使用鎖定版本執行 test inference。
- [ ] 不根據 test 結果更換模型、權重或 threshold。
- [ ] 產生 expert 0～3 的 test metrics。
- [ ] 產生 Level 1、Level 2、Level 3 test metrics。
- [ ] 產生 case-level test 結果與信賴區間／分布摘要。
- [ ] 檢查 metric 計算與案例數量是否一致。
- [ ] 保存原始 predictions、metrics 與執行 log。
- [ ] 若發現程式錯誤，先記錄並區分「修 bug」與「改方法」。

## 3. 最終展示材料

- [ ] 更新主要結果表與 ablation 表。
- [ ] 完成研究架構圖與推論流程圖。
- [ ] 完成代表性案例圖與必要註解。
- [ ] 準備研究動機、方法、結果、限制與下一步講稿。
- [ ] 將 Demo 控制在 10～15 分鐘。
- [ ] 進行最終 rehearsal 並錄下實際時間。
- [ ] 準備預先錄製或靜態結果的備援方案。
- [ ] 準備教授可能詢問的五個問題與回答。

## 4. 教授 Demo 與回饋

- [ ] 執行教授 Demo。
- [ ] 記錄教授對方法、實驗與論文主張的回饋。
- [ ] 將回饋分成必要修正、建議補強與投稿後工作。
- [ ] 只安排不會造成 test-set 調參的必要修正。
- [ ] 建立 Week 12 論文修改清單。

## 週末驗收

- [ ] Demo 完成且有備援材料。
- [ ] 最終 test 結果已保存並停止方法調整。
- [ ] 教授回饋已轉為明確的論文修改事項。

**本週產物：** final release、test results、教授 Demo、展示材料、回饋與修改清單。  
**阻塞／備註：**

