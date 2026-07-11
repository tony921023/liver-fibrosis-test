# 實驗結果

⚠️ **所有數字都是樂觀上限**,不是真實表現 —— patient-level leakage 拆不掉,
來源混淆也還沒排除。見 [leakage.md](leakage.md)。

全部 resnet18、TTA、`SEED=42`。

---

## 正式基準線:perceptual dedup + 5-fold CV

1388 張影像,每張剛好被「沒看過它的那一折」預測一次(out-of-fold)。

### multiclass(F0~F4)

| 指標 | mean ± std |
|---|---|
| macro AUROC | 0.9355 ± 0.0104 |
| **balanced accuracy** | **0.7624 ± 0.0226** |
| QWK | 0.7316 ± 0.0430 |

per-class recall:

| F0 | F1 | F2 | F3 | F4 |
|---|---|---|---|---|
| 0.960 ± 0.038 | 0.702 ± 0.076 | 0.711 ± 0.107 | 0.692 ± 0.062 | 0.747 ± 0.046 |

- **balanced accuracy 才是真正的難度**(隨機 = 0.20)。macro AUROC 在 5 分類 OvR 下偏寬鬆
- **兩端好、中間差**:F0/F4 好認,F1/F2/F3 難分 —— 與臨床上判讀者間差異最大的區間一致
- 錯誤有 ordinal 結構:正確約 75% / 錯 1 期約 15% / 錯 ≥2 期約 10%。模型「錯也錯得近」

### binary ≥F2(顯著纖維化)

| 指標 | mean ± std |
|---|---|
| macro AUROC | 0.9306 ± 0.0100 |
| balanced accuracy | 0.8593 ± 0.0209 |
| **sensitivity** | **0.8624 ± 0.0394** |
| **specificity** | **0.8563 ± 0.0529** |

高 sensitivity 的方向是 `CLASS_WEIGHTS="auto"` 推的,臨床上正確
(漏掉顯著纖維化的代價高)。要別的取捨點**調閾值即可,不必重訓** —— 見 `calibrate.py`
的閾值表。

---

## 為什麼所有「A 比 B 好」的結論都不可信

單一 split 的 test 只有 231 張、每類約 46 張,**recall 的 95% CI 約 ±0.13**。

同設定跑兩次(同 seed、同 split,差異僅來自 cuDNN / mixup 的非決定性),QWK 就差了
**0.0132** —— 這只是雜訊的**下限**。

所以下列改動的效果**全部淹在雜訊裡,證明不了**:

| 改動 | 效果 | 判定 |
|---|---|---|
| 正則化(dropout/wd/mixup) | +0.018 balanced accuracy | ❌ std 就是 ±0.026,證不出 |
| resnet18 vs resnet50 | +0.017 | ❌ 同上 |
| 專訓二元 vs 5 分類摺疊 | 單一 split 看似 +0.10 sens | ❌ CV 後兩者相同(0.855 vs 0.855) |

**唯一站得住的結論**:正則化讓 `best_epoch` 從 13 推遲到 20~30 → 模型撐更久才過擬合。
這是機制上的證據,不是指標上的。

---

## 過擬合曾是瓶頸(已緩解)

首輪 baseline:`train_loss 1.44 → 0.30`,但 `val_loss` 從 epoch 10 起卡在 0.87,
val AUROC plateau 在 0.92 —— 之後每一輪都只是在背 training set。

而且 **backbone 越大越差**(皆含正則化):

| backbone | AUROC | balanced acc | QWK |
|---|---|---|---|
| **resnet18** | **0.9474** | **0.7680** | **0.8300** |
| resnet50 | 0.9308 | 0.7333 | 0.8213 |
| convnext_tiny | 0.9352 | 0.7295 | 0.7293 |

去重後只剩約 1100 張訓練影像 → **瓶頸是資料量不是模型容量**。
別再往上換 backbone。加正則化(`WEIGHT_DECAY` / `DROPOUT` / `MIXUP_ALPHA`)才是對的方向。

⚠️ 但這三個 backbone 的差距同樣落在雜訊範圍內,只能說「沒證據更大的比較好」。

---

## 遮罩消融(尚未定案)

第一輪用了 `MASK_FRAC=0.7`,**事後才發現組織還剩 16%**(200 張隨機影像實測,
一半的圖剩 >20%)→ 那一輪的結果**不能定案**:

| MASK | bal.acc | F0 recall | 組織剩餘 |
|---|---|---|---|
| none | 0.742 | 0.967 | 100% |
| center (frac=0.7) | 0.642 | 0.900 | ~16% ⚠️ |
| periphery (frac=0.7) | 0.700 | 0.967 | ~84% |

暗示很強(只留 16% 組織就拿到 0.642,隨機是 0.20),但 16% ≠ 0,證據不夠硬。

**已改成 `center` 用 frac=0.9(組織剩 1.9%)、`periphery` 用 frac=0.6,待重跑。**

---

## 歷史:去重前的「灌水」數字

留作對照,說明 leakage 有多致命。`DEDUP=False` 可重現。

| 指標 | 值 |
|---|---|
| test macro AUROC | 0.9975 |
| balanced accuracy | 0.9714 |
| QWK | 0.9927 |
| F0 / F4 recall | 1.000 / 1.000 |

**這不是模型強,是 train ∩ test 有 636 張位元組完全相同的影像。**
