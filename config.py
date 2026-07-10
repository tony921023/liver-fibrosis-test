"""集中超參數與選項。

之所以集中放這裡:之後要搬到真實臨床資料 / 換 backbone / 切換預測目標時,
只改這一個檔案,train.py、dataset.py、model.py 都不用動。
"""

import os

# ---- 資料 ----
# ImageFolder root,底下是 F0~F4。
# Colab 上資料路徑不同,用環境變數覆蓋即可,不用改這個檔案:
#     %env DATA_DIR=/content/dl/Dataset
#     !python train.py
DATA_DIR = os.environ.get("DATA_DIR", "data/Dataset")
IMG_SIZE = 224
# ImageNet 統計值(transfer learning 用 pretrained backbone 時必須對齊)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ---- 去重(dedup)----
# ⚠️ 這份公開資料含大量「位元組完全相同」的重複影像:
#     6323 個檔案其實只有 1536 張不重複的圖(平均每張被複製約 4 次,最多 18 次)。
# 隨機 split 會讓同一張圖的複製品同時落在 train 與 test → 模型用背的就近乎滿分
#(去重前 test macro AUROC = 0.9975,F0/F4 recall = 1.000,正是複製倍率最高的兩類)。
# DEDUP=True 會先依「檔案內容 hash」分組,每組只留一張代表,再做 split。
# 去重後各類約 300 張(F0~F4 = 317/296/308/308/307),其實相當平衡。
# 設 False 可重現舊的「灌水」數字,用來對照。
DEDUP = True

# ---- augmentation 強度 ----
# "basic"  = resize + 水平翻轉(原本的設定)
# "strong" = 再加 RandomResizedCrop / 小角度旋轉 / 亮度對比抖動
# 超音波是灰階且方向有意義,所以不做垂直翻轉、不動色相(hue)。
# 去重後訓練集只剩約 1075 張,augmentation 是主要的過擬合防線。
AUG_STRENGTH = "strong"

# ---- split (train / val / test 三分)----
# val 只用來選模型(early stopping / 存 checkpoint),test 只在最後評一次。
# 這樣回報的數字才不會有「拿同一份 val 既調參又打分」的選模型樂觀偏差。
# 兩者皆為「占全體」的比例;train = 1 - VAL_SPLIT - TEST_SPLIT。
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
STRATIFY = True                # 依類別分層,避免小類別在某個 split 缺漏
SEED = 42                      # 固定亂數種子 → 同一份 split 可重現

# ---- cross-validation(crossval.py 用)----
# 單一 split 的 test 只有 231 張、每類約 46 張,recall 的 95% CI 約 ±0.13,
# 小幅改動(例如正則化帶來的 +0.018 balanced accuracy)分不出是真的還是雜訊。
# k-fold 讓每張影像剛好被評估一次,拿到的是 out-of-fold 估計 + 摺間標準差。
N_FOLDS = 5

# ---- 訓練(兩階段微調)----
# Phase 1「暖身」:凍結 backbone,只訓 head,跑 WARMUP_EPOCHS 輪。
# Phase 2「微調」:解凍 backbone,用 differential LR 微調,跑到 EPOCHS。
# 想退回「單階段只訓 head」→ 設 WARMUP_EPOCHS = EPOCHS;
# 想「從頭就全網路微調」→ 設 WARMUP_EPOCHS = 0。
# 去重後資料量掉到約 1/4,每輪步數變少 → 總輪數要拉高才收斂得完。
EPOCHS = 30                    # 總輪數(含暖身);配合 early stopping,不一定跑滿
WARMUP_EPOCHS = 3              # 暖身輪數(凍結 backbone)
HEAD_LR = 1e-3                 # head 的學習率(兩階段都用這個)
BACKBONE_LR = 1e-4             # 解凍後 backbone 的學習率(較小,避免破壞 pretrained 特徵)
BATCH_SIZE = 32
NUM_WORKERS = 2                # Mac/Colab 都安全;Colab 可調大

# ---- 正則化(對抗過擬合)----
# 首輪 baseline(resnet18,dedup 後 1074 張 train)量到:
#   train_loss 1.44 → 0.30,但 val_loss 從 epoch 10 起就卡在 0.87 上不去,
#   val AUROC 也 plateau 在 0.92 —— 之後每一輪都只是在背 training set。
# 而且 resnet50 表現不比 resnet18 好(AUROC 0.934 vs 0.940),
# 代表限制在「資料量」不在「模型容量」→ 該加的是正則化,不是更大的 backbone。
WEIGHT_DECAY = 1e-4            # 用 AdamW(decoupled weight decay);0 = 關閉
DROPOUT = 0.3                  # 分類頭前的 dropout;0 = 關閉
# mixup:把兩張影像與其標籤按 lam 線性混合,強迫模型在樣本之間平滑決策邊界。
# 對小資料集特別有效。alpha 越大混得越兇;0 = 關閉。
MIXUP_ALPHA = 0.2

# ---- 損失函數 ----
# label smoothing:避免模型對 5 分期過度自信。分期邊界(F1/F2)本來就有判讀者間差異,
# 硬標籤把它當成 100% 確定並不合理。0 = 關閉。
LABEL_SMOOTHING = 0.05
# "auto" = 依 train split 的類別頻率給權重(少數類權重高);"none" = 不加權。
# 這份資料去重後已相當平衡,auto 幾乎等於不加權;留著是為了未來臨床資料(通常很不平衡)。
CLASS_WEIGHTS = "auto"

# ---- test-time augmentation ----
# True = test/val 評估時平均「原圖」與「水平翻轉」兩次的機率,通常小幅穩定提升。
TTA = True

# ---- 學習率排程 ----
# "cosine"  = CosineAnnealingLR(在 Phase 2 期間退火)
# "plateau" = ReduceLROnPlateau(val 指標停滯時降 LR)
# "none"    = 不排程
SCHEDULER = "cosine"

# ---- early stopping / checkpoint ----
EARLY_STOP_PATIENCE = 7        # 連續幾輪 val 指標沒進步就停;<=0 關閉
MONITOR = "val_macro_auroc"    # 監看指標(越大越好),也用來決定存哪個 checkpoint
CKPT_DIR = "checkpoints"       # 最佳權重存這(已被 .gitignore 擋掉,不會誤 commit)

# ---- 結果輸出 ----
# 每輪指標(metrics.csv)、訓練曲線、test 集評估報告與 confusion matrix 都存這。
# 跑不同設定時用環境變數分開存,才不會互相覆蓋:
#     RESULTS_DIR=outputs_binary TASK=binary_geF2 python train.py
RESULTS_DIR = os.environ.get("RESULTS_DIR", "outputs")   # 已被 .gitignore 擋掉
# crossval.py 的輸出根目錄,底下是 fold_1/ ... fold_N/ 與 cv_summary.json
CV_RESULTS_DIR = os.environ.get("CV_RESULTS_DIR", "outputs_cv")

# ---- 模型 ----
# 可選:resnet18 / resnet34 / resnet50 / resnet101 / efficientnet_b0 / convnext_tiny
# 實測 resnet50 沒有比 resnet18 好(macro AUROC 0.934 vs 0.940,QWK 0.783 vs 0.820),
# 資料量才是瓶頸 → 別急著換更大的 backbone,先把正則化做好。
BACKBONE = "resnet18"
PRETRAINED = True

# ---- 預測目標 ----
# "multiclass"  = METAVIR 五分期 (F0~F4)
# "binary_geF2" = 二元門檻 (>=F2 為陽性),臨床上真正在問的「有沒有顯著纖維化」
# 兩者共用同一份 code,train/dataset/metrics 會自動跟著走。
# 用環境變數切換:  TASK=binary_geF2 python train.py
TASK = os.environ.get("TASK", "multiclass")
