# 實驗 TODO 總覽

詳細工作已依週拆分至 [todo](todo/) 資料夾。本頁用於快速查看進度；實際完成狀態請在各週檔案中勾選。

## 已固定的實驗決策

- **資料集：** LIDC-IDRI
- **任務：** 肺結節分割
- **切分原則：** 以病人為單位建立 training／validation／test split，避免資料洩漏。
- **待定項目：** 多位放射科醫師標註的整合規則、納入／排除條件、前處理與資料切分比例。

| Week | 日期 | 階段 | 詳細清單 | 狀態 |
| ---: | --- | --- | --- | --- |
| 01 | 10/09～10/15 | Pipeline／Stage 1 | [開啟](todo/week01.md) | 🟨 進行中 |
| 02 | 10/16～10/22 | Stage 1 | [開啟](todo/week02.md) | ⬜ 未開始 |
| 03 | 10/23～10/29 | Stage 2 | [開啟](todo/week03.md) | ⬜ 未開始 |
| 04 | 10/30～11/05 | Stage 2 | [開啟](todo/week04.md) | ⬜ 未開始 |
| 05 | 11/06～11/12 | Stage 3 Level 1 | [開啟](todo/week05.md) | ⬜ 未開始 |
| 06 | 11/13～11/19 | Stage 3 Level 2 | [開啟](todo/week06.md) | ⬜ 未開始 |
| 07 | 11/20～11/26 | Stage 3 Level 2／3 | [開啟](todo/week07.md) | ⬜ 未開始 |
| 08 | 11/27～12/03 | Stage 3 Level 3 | [開啟](todo/week08.md) | ⬜ 未開始 |
| 09 | 12/04～12/10 | Final Stage | [開啟](todo/week09.md) | ⬜ 未開始 |
| 10 | 12/11～12/17 | Final Stage | [開啟](todo/week10.md) | ⬜ 未開始 |
| 11 | 12/18～12/24 | Final Stage／Demo | [開啟](todo/week11.md) | ⬜ 未開始 |
| 12 | 12/25～12/31 | 論文初版 | [開啟](todo/week12.md) | ⬜ 未開始 |

## 狀態標記

- ⬜ 未開始
- 🟨 進行中
- ✅ 已完成
- 🟥 阻塞

## 使用方式

1. 每週開始時確認該週的輸入條件與計算資源。
2. 執行過程直接勾選當週檔案中的工作項目。
3. 每次實驗均保存 config、seed、commit／版本、log、checkpoint 與 metrics。
4. 每週結束前完成驗收；未完成項目需註記原因並移入下一週。
5. 同步更新本頁的狀態欄及 [Timeline](Timeline.md) 中的里程碑。
