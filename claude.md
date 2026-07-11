# Liver Fibrosis Ultrasound Classification — 專案脈絡

## 專案性質
這是**驗證 / 練習用**的 testbed,目的是把 pipeline 跑通、練熟目標架構,
**不是**產出論文成績。最終會搬到真實臨床資料(碩論研究),所以程式碼要可移植、模組化。

## 現況
- 資料:`data/Dataset/F0` ~ `F4`(METAVIR 五分期,ImageFolder 格式)
  - 6323 個檔案,但去重後只有 **1536 張**(exact)/ **1388 張**(perceptual)
  - 各類相當平衡(perceptual 後:199/288/308/308/285)
- 環境:venv,torch 2.8(arm64/MPS)+ torchvision + scikit-learn + matplotlib
- 開發機 MacBook Pro M2(MPS);訓練在 Colab(CUDA)→ 程式須同時支援
- git repo + GitHub remote;`.gitignore` 已擋掉 data/、venv/、憑證、輸出

## 🚨 資料問題(動手前務必先讀 [docs/leakage.md](docs/leakage.md))
1. **完全重複的影像** — 6323 檔只有 1536 張不重複。不去重 AUROC 會灌到 0.9975
2. **近重複的影像** — exact 之上還有 148 張(F0 特別多)
3. **來源混淆** — F0 有 198/199 張是 `a` 前綴,`ct` 只在 F1~F3 → 模型可能在「認來源」
4. **patient-level leakage** — 無病人 ID,**無解**

→ **所有數字都是樂觀上限,不是真實表現。**

## 重要約束(務必遵守)
- **device 自動偵測**:cuda → mps → cpu,同一份 code 在 Mac 和 Colab 都能跑
- **不要**把 data/、venv/、模型權重、kaggle.json commit 進 git
- requirements.txt 用寬鬆版本(Mac arm64 ↔ Colab x86 相容)
- 新增實驗設定時,**記得加進 `train._run_metadata()`** —— 否則之後分不出結果是哪來的

## 結構
| 檔案 | 職責 |
|---|---|
| `config.py` | 超參數與選項;常用的可用環境變數覆蓋 |
| `dataset.py` | 載入 / transforms / dedup / split(單一 split 與 k-fold) |
| `model.py` | backbone + head;凍結/解凍、differential LR |
| `metrics.py` | 指標與結果輸出,與訓練流程解耦 |
| `train.py` | 訓練 + 評估。`run_one()` 是唯一入口,crossval 也用它 |
| `crossval.py` | k-fold,輸出 mean ± std 與 out-of-fold 機率。**可續跑** |
| `calibrate.py` | 校準 / 溫度縮放 / bootstrap CI / 閾值表(不用重訓) |
| `explain.py` | Grad-CAM 混淆稽核 |
| `colab_setup.ipynb` | Colab 執行流程 |

## 跑法
```bash
python train.py                                        # 單一 split
DEDUP=perceptual TASK=binary_geF2 python train.py      # 換設定
python crossval.py                                     # k-fold(結論要靠這個)
MASK=center MASK_FRAC=0.9 python train.py              # 遮罩消融
python calibrate.py outputs_cv/oof_predictions.npz     # 校準
```

## 結果
見 [docs/results.md](docs/results.md)。摘要(perceptual + 5-fold CV):

| | multiclass | binary ≥F2 |
|---|---|---|
| macro AUROC | 0.936 ± 0.010 | 0.931 ± 0.010 |
| balanced accuracy | **0.762 ± 0.023** | 0.859 ± 0.021 |
| sensitivity / specificity | — | 0.862 / 0.856 |

⚠️ **所有「A 比 B 好 0.0x」的比較都淹在雜訊裡**(正則化、backbone、專訓 vs 摺疊)。
單一 split 比不出小差異,結論一律用 `crossval.py`。

## 本階段範圍
1. ✅ 5 分類 transfer learning baseline
2. ✅ leakage 處理、k-fold、校準、混淆稽核
3. ⏳ 遮罩消融待重跑(frac 修正後)
4. 預留擴充:**多模態(影像 + tabular)**、**attention-MIL(多視角)**
   —— 兩者都需要臨床資料才做得動(這份公開資料無 tabular、無病人 ID)
