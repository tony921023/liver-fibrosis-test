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
- `train.py`:訓練迴圈 + 評估(macro AUROC)
- `config.py` 或 argparse:集中超參數與選項

## 本階段範圍
1. 先做**能跑的 5 分類 transfer learning baseline**(resnet,先小後大)
2. 指標:macro AUROC(one-vs-rest)+ train/val loss
3. 結構要預留擴充:**多模態(影像 + tabular)融合**、**attention-MIL(多視角)**
4. 預測目標目前用 5 分類,但設計成可切換(config 旗標),之後可能改二元門檻(如 ≥F2)