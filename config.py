"""集中超參數與選項。

之所以集中放這裡:之後要搬到真實臨床資料 / 換 backbone / 切換預測目標時,
只改這一個檔案,train.py、dataset.py、model.py 都不用動。
"""

# ---- 資料 ----
DATA_DIR = "data/Dataset"      # ImageFolder root,底下是 F0~F4
IMG_SIZE = 224
# ImageNet 統計值(transfer learning 用 pretrained backbone 時必須對齊)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ---- split ----
VAL_SPLIT = 0.2
STRATIFY = True                # 依類別分層,避免小類別在 val 缺漏
SEED = 42                      # 固定亂數種子 → 同一份 split 可重現

# ---- 訓練(兩階段微調)----
# Phase 1「暖身」:凍結 backbone,只訓 head,跑 WARMUP_EPOCHS 輪。
# Phase 2「微調」:解凍 backbone,用 differential LR 微調,跑到 EPOCHS。
# 想退回「單階段只訓 head」→ 設 WARMUP_EPOCHS = EPOCHS;
# 想「從頭就全網路微調」→ 設 WARMUP_EPOCHS = 0。
EPOCHS = 20                    # 總輪數(含暖身);配合 early stopping,不一定跑滿
WARMUP_EPOCHS = 3              # 暖身輪數(凍結 backbone)
HEAD_LR = 1e-3                 # head 的學習率(兩階段都用這個)
BACKBONE_LR = 1e-4             # 解凍後 backbone 的學習率(較小,避免破壞 pretrained 特徵)
BATCH_SIZE = 32
NUM_WORKERS = 2                # Mac/Colab 都安全;Colab 可調大

# ---- 學習率排程 ----
# "cosine"  = CosineAnnealingLR(在 Phase 2 期間退火)
# "plateau" = ReduceLROnPlateau(val 指標停滯時降 LR)
# "none"    = 不排程
SCHEDULER = "cosine"

# ---- early stopping / checkpoint ----
EARLY_STOP_PATIENCE = 5        # 連續幾輪 val 指標沒進步就停;<=0 關閉
MONITOR = "val_macro_auroc"    # 監看指標(越大越好),也用來決定存哪個 checkpoint
CKPT_DIR = "checkpoints"       # 最佳權重存這(已被 .gitignore 擋掉,不會誤 commit)

# ---- 模型 ----
BACKBONE = "resnet18"          # 可換 "resnet34"/"resnet50" 等 torchvision 名稱
PRETRAINED = True

# ---- 預測目標(預留擴充)----
# "multiclass" = METAVIR 五分期 (F0~F4)
# "binary_geF2" = 二元門檻 (>=F2 為陽性),臨床上常用的顯著纖維化判讀
# 目前用 multiclass,之後改旗標即可切換,train/dataset 會自動跟著走。
TASK = "multiclass"
