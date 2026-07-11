"""訓練迴圈 + 評估。

訓練骨架:
  Phase 1「暖身」  凍結 backbone,只訓 head(config.WARMUP_EPOCHS 輪)
  Phase 2「微調」  解凍 backbone,differential LR + LR scheduler 微調到 config.EPOCHS
  全程       監看 val 指標 → early stopping + 存最佳 checkpoint(用 val,不碰 test)
  最後       載回最佳權重,在獨立 test 集評一次:macro AUROC + per-class + QWK + 混淆矩陣

結果輸出(config.RESULTS_DIR):metrics.csv、curves.png、confusion_matrix.png、test_report.json

跑法:  python train.py
超參數集中在 config.py;要換 backbone / 排程 / 預測目標,改 config 即可。

⚠️ 評估數字仍偏樂觀:即使 DEDUP=True 去掉了完全重複的影像,
這份資料沒有病人 ID,做不到 patient-level split。詳見 dataset.py 的 LEAKAGE 警告。
"""

import os

import numpy as np
import torch
import torch.nn as nn

from dataset import get_dataloaders            # 來自 dataset.py
from model import build_model, set_backbone_trainable, param_groups  # 來自 model.py
import metrics                                 # 評估指標與結果輸出
import config                                  # 超參數集中放這


def get_device():
    # device 自動偵測:同一份 code 在 Mac(MPS)和 Colab(CUDA)都能跑
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _mixup(x, y, alpha):
    """把 batch 內兩兩樣本按 lam 線性混合,回傳 (mixed_x, y_a, y_b, lam)。

    影像混合、標籤不混合 —— 改成 loss 端用 lam 加權兩個 CE
    (等價於混合 one-hot 標籤,但這樣寫可以直接沿用帶 class weight / label
    smoothing 的 CrossEntropyLoss,不必自己攤開 soft target)。
    """
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    # batch 只有 1 張時 mixup 等於沒混(randperm 只會拿到自己),直接跳過
    use_mixup = config.MIXUP_ALPHA > 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if use_mixup and x.size(0) > 1:
            x, y_a, y_b, lam = _mixup(x, y, config.MIXUP_ALPHA)
            logits = model(x)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
        else:
            loss = criterion(model(x), y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def collect(model, loader, criterion, device, tta=False):
    """跑一遍 loader,回傳 (avg_loss, probs[N,C], labels[N])。

    供 per-epoch 的 val 評估與最後的 test 完整報告共用。

    tta=True 時把「原圖」與「水平翻轉」兩次的機率平均(超音波左右翻轉不改變分期)。
    回報的 loss 一律取原圖那次,才跟訓練/early stopping 看的是同一個量。
    """
    model.eval()
    total_loss = 0.0
    probs, labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)

        p = torch.softmax(logits, dim=1)
        if tta:
            p_flip = torch.softmax(model(torch.flip(x, dims=[3])), dim=1)
            p = (p + p_flip) / 2

        probs.append(p.cpu().numpy())
        labels.append(y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.concatenate(probs), np.concatenate(labels)


def _build_optimizer(model):
    """param_groups 只收 requires_grad 的參數:
    暖身階段 backbone 凍結 → 只剩 head;微調階段含 backbone(differential LR)。

    用 AdamW 而非 Adam:AdamW 的 weight decay 是 decoupled 的,不會被 Adam 的
    自適應學習率縮放掉,才真的起到正則化作用。WEIGHT_DECAY=0 時兩者等價。
    """
    groups = param_groups(model, config, backbone_lr=config.BACKBONE_LR, head_lr=config.HEAD_LR)
    return torch.optim.AdamW(groups, weight_decay=config.WEIGHT_DECAY)


def _build_scheduler(optimizer, finetune_epochs):
    name = config.SCHEDULER
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, finetune_epochs))
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2)
    if name == "none":
        return None
    raise ValueError(f"未知的 SCHEDULER: {name!r}")


def run_one(data, device, results_dir, ckpt_path):
    """訓練一次 + 在 test 集評估一次,回傳 report dict。

    抽出來是為了讓 train.py(單一 split)與 crossval.py(k-fold)共用同一段邏輯,
    不會出現「CV 跑的其實是另一份訓練流程」這種對不起來的狀況。
    """
    num_classes, class_names = data.num_classes, data.class_names
    model = build_model(num_classes, config).to(device)

    # class weights 依 train split 頻率算(見 dataset._class_weights);label smoothing 見 config
    weights = data.class_weights.to(device) if data.class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=config.LABEL_SMOOTHING)

    print(f"[train] backbone={config.BACKBONE}  dropout={config.DROPOUT}  "
          f"weight_decay={config.WEIGHT_DECAY}  mixup_alpha={config.MIXUP_ALPHA}")
    if config.MIXUP_ALPHA > 0:
        # mixup 的 target 是混合的,loss 天生比 val 高;兩者不能直接比大小判斷過擬合,
        # 要看 val_loss 自己的走勢有沒有回升。
        print("[train] ⚠️ mixup 開啟 → train_loss 會偏高,不可直接與 val_loss 比較")

    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # --- Phase 1:暖身(凍結 backbone,只訓 head)---
    warmup = max(0, min(config.WARMUP_EPOCHS, config.EPOCHS))
    set_backbone_trainable(model, config, trainable=False)
    optimizer = _build_optimizer(model)
    scheduler = None  # 暖身階段不排程,LR 維持 HEAD_LR
    phase = "warmup"

    best_metric = -float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, config.EPOCHS + 1):
        # --- 進入 Phase 2:解凍 backbone,換 differential-LR optimizer + scheduler ---
        if epoch == warmup + 1:
            set_backbone_trainable(model, config, trainable=True)
            optimizer = _build_optimizer(model)
            scheduler = _build_scheduler(optimizer, finetune_epochs=config.EPOCHS - warmup)
            phase = "finetune"
            print(f"--- unfreeze backbone @ epoch {epoch} "
                  f"(backbone_lr={config.BACKBONE_LR}, head_lr={config.HEAD_LR}) ---")

        train_loss = train_one_epoch(model, data.train_loader, criterion, optimizer, device)
        # val 不開 TTA:每輪都跑兩次前向會拖慢訓練,而選模型只需要相對排序
        val_loss, val_probs, val_labels = collect(model, data.val_loader, criterion, device)
        auroc = metrics.macro_auroc(val_labels, val_probs, num_classes)

        if scheduler is not None:
            scheduler.step(auroc) if config.SCHEDULER == "plateau" else scheduler.step()

        history.append({"epoch": epoch, "phase": phase, "train_loss": train_loss,
                        "val_loss": val_loss, "val_macro_auroc": auroc})

        improved = auroc > best_metric
        flag = ""
        if improved:
            best_metric, best_epoch, epochs_no_improve = auroc, epoch, 0
            torch.save(
                # metric 轉 Python float:roc_auc_score 回傳 numpy 標量,
                # 否則 torch>=2.6 的 weights_only=True 預設載入會失敗。
                {"model_state": model.state_dict(), "epoch": epoch,
                 "metric": float(auroc), "num_classes": num_classes,
                 "class_names": class_names,
                 "backbone": config.BACKBONE, "task": config.TASK},
                ckpt_path,
            )
            flag = "  <- best (saved)"
        else:
            epochs_no_improve += 1

        print(f"epoch {epoch:2d} [{phase:8s}]: train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_macro_auroc={auroc:.4f}{flag}")

        # --- early stopping(只看 val)---
        if config.EARLY_STOP_PATIENCE > 0 and epochs_no_improve >= config.EARLY_STOP_PATIENCE:
            print(f"early stop @ epoch {epoch} "
                  f"(no improvement for {config.EARLY_STOP_PATIENCE} epochs)")
            break

    print(f"best val_macro_auroc={best_metric:.4f} @ epoch {best_epoch}  -> {ckpt_path}")

    # --- 存每輪指標與訓練曲線 ---
    metrics.save_history_csv(history, os.path.join(results_dir, "metrics.csv"))
    metrics.save_curves(history, os.path.join(results_dir, "curves.png"))

    # --- 載回最佳權重,在「獨立 test 集」評一次(這才是要回報的數字)---
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, test_probs, test_labels = collect(
        model, data.test_loader, criterion, device, tta=config.TTA)
    report = metrics.full_report(test_labels, test_probs, class_names)

    print("\n===== TEST 集評估(best checkpoint @ epoch "
          f"{ckpt['epoch']}{', TTA' if config.TTA else ''})=====")
    print(metrics.format_report(report))
    if config.DEDUP:
        print("⚠️ 已去除完全重複的影像,但仍非 patient-level split → 數字依舊偏樂觀。")
    else:
        print("🚨 DEDUP=False:重複影像橫跨 train/test,以下數字嚴重灌水,不具參考價值。")

    # 記下影響可信度與結果的設定,之後對照不同 run 才知道數字是哪來的
    extra = {"best_epoch": int(ckpt["epoch"]), "backbone": config.BACKBONE,
             "task": config.TASK, "val_macro_auroc_best": float(best_metric),
             "dedup": config.DEDUP, "tta": config.TTA,
             "aug_strength": config.AUG_STRENGTH,
             "weight_decay": config.WEIGHT_DECAY, "dropout": config.DROPOUT,
             "mixup_alpha": config.MIXUP_ALPHA,
             "label_smoothing": config.LABEL_SMOOTHING}

    metrics.save_report_json(
        report, os.path.join(results_dir, "test_report.json"), extra=extra)
    metrics.save_confusion_png(
        report, os.path.join(results_dir, "confusion_matrix.png"))
    print(f"\n結果已存到 {results_dir}/:"
          " metrics.csv, curves.png, confusion_matrix.png, test_report.json")

    # 回傳的 report 內容與存進 json 的完全一致(crossval.py 才拿得到 best_epoch 等欄位);
    # probs/labels 另外回傳供 crossval 彙總成 out-of-fold 預測(校準 / ROC 要用)。
    return {**report, **extra}, test_probs, test_labels


def main():
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    device = get_device()
    print("device:", device, "| torch:", torch.__version__)

    data = get_dataloaders(config)
    run_one(data, device, config.RESULTS_DIR,
            os.path.join(config.CKPT_DIR, "best.pt"))   # 單一 split 用不到 probs/labels


if __name__ == "__main__":
    main()
