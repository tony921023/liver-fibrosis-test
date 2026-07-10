# Liver Fibrosis Ultrasound Classification — 專案脈絡

## 專案性質
這是**驗證 / 練習用**的 testbed,目的是把 pipeline 跑通、練熟目標架構,
**不是**產出論文成績。最終會搬到真實臨床資料(碩論研究),所以程式碼要可移植、模組化。

## 現況
- 資料:`data/Dataset/F0` ~ `F4`(METAVIR 五分期,ImageFolder 格式)
  - 檔案 6323 個(5397 `.jpg` + 926 `.png`),但**只有 1536 張不重複影像**(見下方 Leakage)
  - 去重後各類約 300 張:F0=317, F1=296, F2=308, F3=308, F4=307 → 其實相當平衡
- 環境:venv,已裝 torch 2.8(arm64/MPS)、torchvision、scikit-learn、pandas、matplotlib、kaggle
- 開發機:MacBook Pro M2(MPS GPU);訓練也會在 Colab(CUDA)跑 → 程式須同時支援
- 已有 git repo + GitHub remote;`.gitignore` 已擋掉 data/、venv/、憑證

## 重要約束(務必遵守)
- **device 自動偵測**:cuda → mps → cpu,同一份 code 在 Mac 和 Colab 都能跑
- **不要**把 data/、venv/、模型權重、kaggle.json commit 進 git
- requirements.txt 用寬鬆版本(Mac arm64 ↔ Colab x86 相容)
- **Leakage 警告(兩層,務必分清楚)**:
  1. **完全重複的影像**(已處理):6323 個檔案只有 1536 張不重複的圖,平均每張被複製約 4 次
     (最多 18 次)。隨機 split 會讓同一張圖的複製品同時落在 train/test —— 實測 train∩test
     有 **636 張位元組相同**的影像,模型用背的就滿分(舊 test macro AUROC = 0.9975,
     F0/F4 recall = 1.000,正是複製倍率最高的兩類)。
     → `config.DEDUP=True` 依內容 hash 去重後再 split,已消除此層。
     附註:1536 個 hash 全部只對應單一類別,**沒有跨類別重複**,標籤本身沒有矛盾。
  2. **patient-level leakage**(無解):檔名是「字母前綴 + 流水號」(`a1000.jpg`/`I2079.jpg`/
     `z9945.jpg`…),前綴不對應病人,**無病人 ID**,做不到 patient-level split。
     即使去重,同一病人的不同切面仍可能分散在不同 split。
  → 因此**所得 AUROC 仍偏樂觀、不可當真實表現**,程式註解須標明。
  真實 patient-level 評估留給未來臨床資料(屆時把 `dataset._dedup_indices` 換成依病人 ID 的
  `GroupShuffleSplit` 即可)。

## 目標結構(請重構成這樣)
- `dataset.py`:資料載入 / transforms / stratified split
- `model.py`:模型定義(transfer learning,backbone 可換)
- `train.py`:訓練迴圈 + 評估(單一 split);`run_one()` 供 crossval 重用
- `crossval.py`:k-fold cross-validation,輸出 mean ± std 與 out-of-fold confusion matrix
- `metrics.py`:評估指標與結果輸出(與訓練流程解耦)
- `config.py`:集中超參數與選項;`DATA_DIR`/`TASK`/`RESULTS_DIR`/`CV_RESULTS_DIR` 可用環境變數覆蓋

## 基準線(2026-07-10,dedup 後,單一 split)
**這是後續改動要比較的對象。** 全部 resnet18、TTA、SEED=42。

### multiclass(5 分期)

| 指標 | 無正則化 | +正則化 |
|---|---|---|
| macro AUROC | 0.9398 | 0.9474 |
| **balanced accuracy** | **0.7506** | **0.7680** |
| QWK | 0.8195 | 0.8300 |
| best_epoch | 13 | 20~30 |

- **balanced accuracy 才是真正的難度**;macro AUROC 在 5 分類 OvR 下偏寬鬆(F0 太好分)
- per-class recall(無正則化):F0=1.000 / F1=0.644 / F2=0.609 / F3=0.739 / F4=0.761
  → **F1/F2/F3 中間分期是戰場**,與臨床上判讀者間差異最大的區間一致
- 錯誤有 ordinal 結構:正確 77.1% / 錯 1 期 13.4% / 錯 ≥2 期 9.5%(正則化後)
- **過擬合曾是瓶頸**:train_loss 1.44→0.30,val_loss 從 epoch 10 起卡在 0.87。
  加 `WEIGHT_DECAY`/`DROPOUT`/`MIXUP_ALPHA` 後 best_epoch 推遲到 20~30,確認有效

### binary_geF2(≥F2 顯著纖維化)

| | sensitivity | specificity | balanced acc |
|---|---|---|---|
| 5 分類結果直接摺成二元 | 0.855 | 0.839 | 0.847 |
| **專門訓練的二元模型** | **0.957** | 0.804 | **0.881** |

專訓贏過摺疊。高 sens / 低 spec 是 `CLASS_WEIGHTS="auto"`(權重 1.20/0.80)推的,
臨床上方向正確(漏掉顯著纖維化的代價高)。要別的取捨點調閾值即可,不必重訓。

### backbone:越大越差(皆含正則化)

| backbone | AUROC | bal.acc | QWK |
|---|---|---|---|
| **resnet18** | **0.9474** | **0.7680** | **0.8300** |
| resnet50 | 0.9308 | 0.7333 | 0.8213 |
| convnext_tiny | 0.9352 | 0.7295 | 0.7293 |

去重後只剩 1074 張訓練影像,**瓶頸是資料量不是模型容量**。別再往上換 backbone。

### ⚠️ 這些數字的三個但書
1. **仍偏樂觀** —— patient-level leakage 還在(見上方 Leakage 警告)
2. **單一 split 的雜訊可能吞掉增益** —— test 只有 231 張、每類約 46 張,
   recall 95% CI 約 ±0.13。同設定跑兩次,QWK 就差 0.0132(同 seed、同 split,
   差異僅來自 cuDNN/mixup 非決定性 → 這只是雜訊**下限**)。
   → 用 `crossval.py` 做 k-fold 才能拿到帶誤差棒的結論
3. **F0 recall 三次都是 1.000(48/48)** —— 完美到需要警覺。可能只是正常肝好認,
   也可能 F0 影像來自不同機器/前處理,模型在認「來源」而非「病理」。尚未排除

## 本階段範圍
1. 先做**能跑的 5 分類 transfer learning baseline**(resnet,先小後大)
2. 指標:macro AUROC(one-vs-rest)+ train/val loss
3. 結構要預留擴充:**多模態(影像 + tabular)融合**、**attention-MIL(多視角)**
4. 預測目標目前用 5 分類,但設計成可切換(config 旗標),之後可能改二元門檻(如 ≥F2)