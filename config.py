"""集中超參數與選項。

搬到臨床資料 / 換 backbone / 切換預測目標時只改這裡,其餘模組不用動。
常用選項可用環境變數覆蓋(Colab 上不必改檔):

    DEDUP=perceptual TASK=binary_geF2 RESULTS_DIR=outputs_binary python train.py
"""

import os

# ---- 資料 ----
DATA_DIR = os.environ.get("DATA_DIR", "data/Dataset")   # ImageFolder root,底下是 F0~F4
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]    # ImageNet 統計值:用 pretrained backbone 時必須對齊
STD = [0.229, 0.224, 0.225]

# ---- 去重 ----
# "exact"(=True)  依 md5 去位元組完全相同的重複    6323 -> 1536 張
# "perceptual"    再用 dHash 去近重複(重新壓縮過)  6323 -> 1388 張
# False           不去重,用來重現去重前的灌水數字
# ⚠️ 不去重的話 train/test 會有 636 張相同影像,AUROC 灌到 0.9975。見 docs/leakage.md
DEDUP = os.environ.get("DEDUP", "exact")

# ---- 遮罩消融 ----
# 來源與類別高度綁定(F0 有 198/199 張是 'a' 前綴),模型可能在「認來源」而非判讀纖維化。
# 塗黑一部分後重訓,可直接測出模型靠什麼吃飯。見 docs/leakage.md 的「來源混淆」。
# "none" / "center"(塗黑中央組織) / "periphery"(塗黑邊緣)
MASK = os.environ.get("MASK", "none")
# ⚠️ 兩種遮罩要用不同尺寸:center 用 0.9(組織剩 1.9%)、periphery 用 0.6(保留 69% 組織)。
# 共用一個值會讓其中一邊的對照失去意義。
MASK_FRAC = float(os.environ.get("MASK_FRAC", 0.9))

# ---- augmentation ----
# "basic" = resize + 水平翻轉;"strong" = 再加 RandomResizedCrop / 旋轉 / 亮度對比抖動。
# 超音波是灰階且方向有意義 → 不做垂直翻轉、不動色相。
AUG_STRENGTH = "strong"

# ---- split ----
# val 只用來選模型(early stopping / checkpoint),test 只在最後評一次
# → 回報的數字才沒有「拿同一份 val 既調參又打分」的樂觀偏差。
VAL_SPLIT = 0.15               # 占全體的比例;train = 1 - VAL_SPLIT - TEST_SPLIT
TEST_SPLIT = 0.15
STRATIFY = True
SEED = 42
N_FOLDS = 5                    # crossval.py:單一 split 的雜訊太大,結論要靠 CV

# ---- 訓練(兩階段微調)----
# Phase 1 凍結 backbone 只訓 head(WARMUP_EPOCHS 輪),Phase 2 解凍 + differential LR。
# WARMUP_EPOCHS = EPOCHS → 只訓 head;= 0 → 從頭全網路微調。
EPOCHS = 30
WARMUP_EPOCHS = 3
HEAD_LR = 1e-3
BACKBONE_LR = 1e-4             # 較小,避免破壞 pretrained 特徵
BATCH_SIZE = 32
NUM_WORKERS = 2                # Mac/Colab 都安全;Colab 可調大

# ---- 正則化 ----
# 資料量(約 1100 張 train)才是瓶頸,不是模型容量 → 加正則化而非換更大的 backbone。
WEIGHT_DECAY = 1e-4            # 用 AdamW:decoupled decay 才真的起正則化作用
DROPOUT = 0.3                  # 分類頭前
MIXUP_ALPHA = 0.2              # 兩張影像按 lam 線性混合;對小資料集特別有效

# ---- 損失函數 ----
# 分期邊界本來就有判讀者間差異,硬標籤當成 100% 確定並不合理。
LABEL_SMOOTHING = 0.05
CLASS_WEIGHTS = "auto"         # "auto" 依 train 頻率加權 / "none" 不加權

TTA = True                     # test 時平均「原圖 + 水平翻轉」兩次的機率

SCHEDULER = "cosine"           # "cosine" / "plateau" / "none"

# ---- early stopping / checkpoint ----
EARLY_STOP_PATIENCE = 7        # <=0 關閉
MONITOR = "val_macro_auroc"    # 越大越好;也用來決定存哪個 checkpoint
CKPT_DIR = "checkpoints"

# ---- 輸出 ----
# 跑不同設定時用環境變數分開存,才不會互相覆蓋。
RESULTS_DIR = os.environ.get("RESULTS_DIR", "outputs")
CV_RESULTS_DIR = os.environ.get("CV_RESULTS_DIR", "outputs_cv")

# ---- 模型 ----
# resnet18/34/50/101 / efficientnet_b0 / convnext_tiny
# ⚠️ 實測 backbone 越大越差(見 docs/results.md)—— 別急著往上換。
BACKBONE = "resnet18"
PRETRAINED = True

# ---- 預測目標 ----
# "multiclass" = METAVIR 五分期;"binary_geF2" = 臨床上真正在問的「有沒有顯著纖維化」
TASK = os.environ.get("TASK", "multiclass")
