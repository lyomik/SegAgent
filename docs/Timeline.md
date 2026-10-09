# 論文實驗時程（2026/10/09～2026/12/31）

## 核心期限

- **2026/12/24：** 完成可向教授展示的穩定 Demo 版本。
- **2026/12/31：** 完成具有方法、結果、圖表與初步結論的論文初版。
- **2027 年 1 月：** 依教授意見修改、排版並投稿研討會。

模型選擇與系統調整只使用 training／validation set；test set 僅供鎖定版本的最終評估。

本研究固定使用 **LIDC-IDRI** 進行肺結節分割；所有階段沿用相同的病人層級資料切分與 ground-truth 生成規則。

## 階段配置

| 階段 | 日期 | 週數 | 主要產出 |
| --- | --- | ---: | --- |
| 前置作業與 Stage 1 | 10/09～10/22 | 2 週 | 共用 pipeline、五種 baseline 結果、勝出模型 |
| Stage 2 | 10/23～11/05 | 2 週 | 五種 attention 結果、三個入選版本 |
| Stage 3 | 11/06～12/03 | 4 週 | 固定融合、規則融合與 Agent 仲裁初版 |
| Final Stage | 12/04～12/24 | 3 週 | 效果最佳化、系統穩定化與教授 Demo |
| 論文初版封版 | 12/25～12/31 | 1 週 | 最終結果核對與論文初版 |

## 每週摘要

| Week | 日期 | 主題 | 週末交付物 | 詳細清單 |
| ---: | --- | --- | --- | --- |
| 01 | 10/09～10/15 | 共用 Pipeline 與 Stage 1 啟動 | 五個 baseline 通過 smoke test | [Week 01](todo/week01.md) |
| 02 | 10/16～10/22 | Stage 1 完成 | 勝出 baseline 與 Stage 1 結果 | [Week 02](todo/week02.md) |
| 03 | 10/23～10/29 | Stage 2 第一輪 | SE、CBAM、ECA 結果 | [Week 03](todo/week03.md) |
| 04 | 10/30～11/05 | Stage 2 完成 | 四個 Stage 3 experts | [Week 04](todo/week04.md) |
| 05 | 11/06～11/12 | Stage 3 Level 1 | 最佳固定式 ensemble | [Week 05](todo/week05.md) |
| 06 | 11/13～11/19 | Stage 3 Level 2 建立 | Rule-based system v1 | [Week 06](todo/week06.md) |
| 07 | 11/20～11/26 | Level 2 定案與 Agent | 鎖定規則與 Agent 初版 | [Week 07](todo/week07.md) |
| 08 | 11/27～12/03 | Stage 3 Level 3 | Agent 初步結果與 ablation | [Week 08](todo/week08.md) |
| 09 | 12/04～12/10 | Final：效果診斷 | 瓶頸報告與 Demo v1 | [Week 09](todo/week09.md) |
| 10 | 12/11～12/17 | Final：最佳化 | 最終候選設定與 Demo v2 | [Week 10](todo/week10.md) |
| 11 | 12/18～12/24 | Final：Demo 封版 | 教授 Demo 與最終測試結果 | [Week 11](todo/week11.md) |
| 12 | 12/25～12/31 | 論文初版封版 | 論文初版 | [Week 12](todo/week12.md) |

## 主要里程碑

- **M1－10/22：** Stage 1 baseline 鎖定。
- **M2－11/05：** Stage 2 experts 鎖定。
- **M3－11/12：** Level 1 固定式 ensemble 鎖定。
- **M4－11/26：** Level 2 規則鎖定，Level 3 Agent 可執行。
- **M5－12/03：** Stage 3 初版完成。
- **M6－12/24：** Demo 版本與最終實驗封版。
- **M7－12/31：** 論文初版完成。

## 執行限制

- 每週保留約 10%～15% 時間處理失敗實驗與環境問題。
- Stage 1、Stage 2 優先確保公平 protocol；完整複驗集中在排名靠前的模型。
- 12/04 後不新增大型模型或主要研究方向，只最佳化既有系統。
- 12/17 後原則上不新增決策特徵，只處理穩定性、必要補跑與 Demo。
- 12/24 後不得依 test set 修改模型、規則、權重或 Agent 決策。

## 2027 年 1 月投稿銜接

- **1 月第 1 週：** 依教授意見修改內容，確認研討會格式與頁數限制。
- **1 月第 2 週：** 完成英文潤稿、圖表重繪、引用與格式檢查。
- **投稿截止前：** 完成最終校稿、作者資訊確認、PDF 檢查與投稿系統上傳。
