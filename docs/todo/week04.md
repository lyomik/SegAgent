# Week 04 TODO｜10/30～11/05

**階段：** Stage 2 完成與 Experts 定案  
**週目標：** 完成 CA、MHSA，選出三個 attention experts。  
**輸入：** Baseline 與 Week 03 三個 attention 版本  
**完成狀態：** ⬜ 未開始

## 1. 剩餘 Attention 實驗

- [ ] 完成 CA 的架構適配與 smoke test。
- [ ] 完成 baseline + CA 訓練、推論與評估。
- [ ] 完成 MHSA 的架構適配與 smoke test。
- [ ] 完成 baseline + MHSA 訓練、推論與評估。
- [ ] 保存兩組 checkpoint、config、log 與 predictions。
- [ ] 檢查五種 attention 的 protocol 一致性。

## 2. 複驗與互補性分析

- [ ] 對 validation 表現排名靠前的版本進行 seed 複驗。
- [ ] 計算平均值、標準差及相對 baseline 的提升。
- [ ] 建立五個版本的 case-level 指標矩陣。
- [ ] 計算模型預測間的 Pairwise Dice／IoU。
- [ ] 找出各模型獨有的成功與失敗案例。
- [ ] 比較效能增益與額外計算成本。
- [ ] 排除表現不穩定或與其他模型高度重複的版本。

## 3. Expert 選擇與封版

- [ ] 固定 expert 0 為未加入 attention 的 baseline。
- [ ] 選出三個兼具表現與互補性的 attention experts。
- [ ] 指定 expert 1～3 的固定順序與名稱。
- [ ] 鎖定四個 experts 的 checkpoint 與 config。
- [ ] 以相同 inference command 產生小批次結果。
- [ ] 確認四個 probability maps 位於相同座標與數值尺度。

## 4. 文件與紀錄

- [ ] 完成 Stage 2 結果表與選擇理由。
- [ ] 製作 attention 效能／成本比較圖。
- [ ] 更新論文 Stage 2 方法與結果。
- [ ] 建立 Stage 3 所需的 prediction manifest。

## 週末驗收

- [ ] 五種 attention 均有完整結果。
- [ ] expert 0～3 已鎖定且可批次推論。
- [ ] Stage 3 可直接讀取四組 probability maps 與 masks。

**本週產物：** Stage 2 最終表、互補性分析、四個 expert manifests、鎖定 checkpoints。  
**阻塞／備註：**

