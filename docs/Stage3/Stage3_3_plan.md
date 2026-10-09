# Stage 3－Level 3：Agent 仲裁

## 實驗目的

Level 3 建立 case-level agent，整合單一 experts、固定式 ensemble 與 rule-based ensemble 的資訊，針對每個案例進行仲裁，選擇適合的候選結果並留下可追蹤的決策依據。

Agent 不直接修改模型參數，而是在既有候選 segmentation 中進行選擇或仲裁。

## Agent 輸入

每個案例包含下列四類資訊：

1. **Expert outputs**
   - 四個 experts 的 probability maps 與 segmentation masks。
   - 各 expert 的 confidence、lesion volume、connected components 與 probability statistics。
2. **Level 1：Static ensemble result**
   - Hard voting、probability average 及 weighted probability average 的結果。
   - Level 1 最佳固定融合候選及其摘要資訊。
3. **Level 2：Rule-based ensemble result**
   - Rule-based system 的最終 segmentation。
   - 被觸發的規則、調整後權重與不確定性標記。
4. **Case-level summary**
   - 病灶大小、病灶候選數、預測一致性、模型信心及其他可解釋的個案統計。

## Agent 任務

- 比較各候選 segmentation 的可信度與一致性。
- 根據 case-level summary 選擇單一 expert、Level 1 或 Level 2 的結果。
- 當所有候選結果分歧過大或信心不足時，將案例標記為 uncertain／review required。
- 輸出結構化的選擇結果與理由，確保決策可重現、可稽核。

## 建議輸出格式

```json
{
  "case_id": "case_xxx",
  "selected_source": "expert_0 | expert_1 | expert_2 | expert_3 | level_1 | level_2",
  "confidence": 0.0,
  "review_required": false,
  "reason_codes": [],
  "summary": ""
}
```

`confidence` 的定義與校正方式須在 validation set 上固定；`reason_codes` 應使用預先定義的有限集合，避免只留下無法統計的自由文字。

## 實驗與限制

- Agent 可使用的資訊必須在推論時實際取得，不可使用 ground truth 或由 ground truth 衍生的特徵。
- Prompt、決策規則、候選結果與輸出 schema 在測試前固定。
- 若 Agent 具有隨機性，需固定 seed／temperature，或進行多次執行並回報穩定性。
- 所有案例應保存輸入摘要、選擇結果、理由與執行版本，以便重現。
- Agent 的選擇不應覆寫原始 expert 與 ensemble 輸出。

## 評估方式

### 分割效能

- Dice、Sensitivity、HD95、Precision、Specificity。
- 與最佳單一 expert、Level 1 及 Level 2 結果比較。

### 仲裁品質

- **Oracle gap：** Agent 結果與「每個案例事後最佳候選」之間的差距。
- **Selection accuracy：** Agent 是否選到該案例表現最佳的候選結果。
- **Review rate：** 被標記為需人工複核的案例比例。
- **Failure capture rate：** 低品質案例中被成功標記為需複核的比例。
- **Decision stability：** 相同輸入重複執行時，選擇結果的一致程度。

## Ablation Study

依序移除不同資訊來源，以分析各類輸入的貢獻：

- 不提供單一 expert 統計資訊。
- 不提供 Level 1 結果。
- 不提供 Level 2 規則與觸發紀錄。
- 不提供 case-level summary。

## 結果紀錄

| 方法 | Dice ↑ | Sensitivity ↑ | HD95 ↓ | Precision ↑ | Specificity ↑ | Review rate | 備註 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 最佳單一 expert | TBD | TBD | TBD | TBD | TBD | N/A | |
| Level 1 | TBD | TBD | TBD | TBD | TBD | N/A | |
| Level 2 | TBD | TBD | TBD | TBD | TBD | TBD | |
| Level 3 Agent | TBD | TBD | TBD | TBD | TBD | TBD | |

**Level 3 最終設定：** TBD
