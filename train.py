"""訓練迴圈 + 評估(單一 split)。

    python train.py
    DEDUP=perceptual TASK=binary_geF2 RESULTS_DIR=outputs_binary python train.py

輸出到 config.RESULTS_DIR:metrics.csv、curves.png、confusion_matrix.png、test_report.json

run_one() 是訓練 + 評估的單一入口,crossval.py 也用它 —— 兩邊的流程不會走鐘。

⚠️ 評估數字是樂觀上限,不是真實表現。見 docs/leakage.md。
"""

import os

import numpy as np
import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import build_model, set_backbone_trainable, param_groups
import metrics
import config


def get_device():
    """同一份 code 在 Mac(MPS)和 Colab(CUDA)都能跑。"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _mixup(x, y, alpha):
    """回傳 (mixed_x, y_a, y_b, lam)。

    只混影像、不混標籤 —— 改在 loss 端用 lam 加權兩個 CE。等價於混合 one-hot,
    但可以直接沿用帶 class weight + label smoothing 的 CrossEntropyLoss,
    不必自己攤開 soft target。
    """
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    use_mixup = config.MIXUP_ALPHA > 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if use_mixup and x.size(0) > 1:     # batch 只有 1 張時 randperm 只會拿到自己
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

    tta=True 時平均「原圖」與「水平翻轉」的機率(超音波左右翻轉不改變分期)。
    loss 一律取原圖那次,才跟 early stopping 看的是同一個量。
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
    return total_loss / len(loader.dataset), np.concatenate(probs), np.concatenate(labels)


def _build_optimizer(model):
    """AdamW 而非 Adam:decoupled weight decay 才真的起正則化作用(WEIGHT_DECAY=0 時等價)。

    param_groups 只收 requires_grad 的參數 → 暖身階段自動只剩 head。
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


def _build_criterion(data, device):
    weights = data.class_weights.to(device) if data.class_weights is not None else None
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=config.LABEL_SMOOTHING)


def _save_checkpoint(model, path, epoch, auroc, data):
    torch.save({
        "model_state": model.state_dict(),
        "epoch": epoch,
        # 轉成 Python float:roc_auc_score 回傳 numpy 標量,
        # 否則 torch>=2.6 的 weights_only=True 預設載入會失敗。
        "metric": float(auroc),
        "num_classes": data.num_classes,
        "class_names": data.class_names,
        "backbone": config.BACKBONE,
        "task": config.TASK,
    }, path)


def _fit(model, data, criterion, device, ckpt_path):
    """兩階段微調 + early stopping。回傳 (history, best_metric, best_epoch)。

    Phase 1 凍結 backbone 只訓 head;Phase 2 解凍 + differential LR + scheduler。
    只看 val 選模型,完全不碰 test。
    """
    warmup = max(0, min(config.WARMUP_EPOCHS, config.EPOCHS))
    set_backbone_trainable(model, config, trainable=False)
    optimizer = _build_optimizer(model)
    scheduler = None            # 暖身階段不排程,LR 維持 HEAD_LR
    phase = "warmup"

    best_metric, best_epoch, epochs_no_improve = -float("inf"), 0, 0
    history = []

    for epoch in range(1, config.EPOCHS + 1):
        if epoch == warmup + 1:
            set_backbone_trainable(model, config, trainable=True)
            optimizer = _build_optimizer(model)
            scheduler = _build_scheduler(optimizer, finetune_epochs=config.EPOCHS - warmup)
            phase = "finetune"
            print(f"--- unfreeze backbone @ epoch {epoch} "
                  f"(backbone_lr={config.BACKBONE_LR}, head_lr={config.HEAD_LR}) ---")

        train_loss = train_one_epoch(model, data.train_loader, criterion, optimizer, device)
        # val 不開 TTA:每輪跑兩次前向會拖慢訓練,而選模型只需要相對排序
        val_loss, val_probs, val_labels = collect(model, data.val_loader, criterion, device)
        auroc = metrics.macro_auroc(val_labels, val_probs, data.num_classes)

        if scheduler is not None:
            scheduler.step(auroc) if config.SCHEDULER == "plateau" else scheduler.step()

        history.append({"epoch": epoch, "phase": phase, "train_loss": train_loss,
                        "val_loss": val_loss, "val_macro_auroc": auroc})

        flag = ""
        if auroc > best_metric:
            best_metric, best_epoch, epochs_no_improve = auroc, epoch, 0
            _save_checkpoint(model, ckpt_path, epoch, auroc, data)
            flag = "  <- best (saved)"
        else:
            epochs_no_improve += 1

        print(f"epoch {epoch:2d} [{phase:8s}]: train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_macro_auroc={auroc:.4f}{flag}")

        if config.EARLY_STOP_PATIENCE > 0 and epochs_no_improve >= config.EARLY_STOP_PATIENCE:
            print(f"early stop @ epoch {epoch} "
                  f"(no improvement for {config.EARLY_STOP_PATIENCE} epochs)")
            break

    return history, best_metric, best_epoch


def _run_metadata(best_epoch, best_metric):
    """記下影響數字的所有設定 —— 沒有這些就無法分辨不同 run 的結果是哪來的。"""
    return {
        "best_epoch": int(best_epoch),
        "backbone": config.BACKBONE,
        "task": config.TASK,
        "val_macro_auroc_best": float(best_metric),
        "dedup": config.DEDUP,
        "tta": config.TTA,
        "aug_strength": config.AUG_STRENGTH,
        "weight_decay": config.WEIGHT_DECAY,
        "dropout": config.DROPOUT,
        "mixup_alpha": config.MIXUP_ALPHA,
        "label_smoothing": config.LABEL_SMOOTHING,
        # frac 一定要記:center 與 periphery 用的是不同尺寸,少了它無法還原實驗設定
        "mask": config.MASK,
        "mask_frac": config.MASK_FRAC,
    }


def run_one(data, device, results_dir, ckpt_path):
    """訓練一次 + 在 test 集評估一次。

    回傳 (report, test_probs, test_labels)。probs/labels 供 crossval.py 彙總成
    out-of-fold 預測(校準 / ROC 要用)。

    train.py(單一 split)與 crossval.py(k-fold)共用這一段,避免兩邊的訓練流程走鐘。
    """
    model = build_model(data.num_classes, config).to(device)
    criterion = _build_criterion(data, device)

    print(f"[train] backbone={config.BACKBONE}  dropout={config.DROPOUT}  "
          f"weight_decay={config.WEIGHT_DECAY}  mixup_alpha={config.MIXUP_ALPHA}")
    if config.MIXUP_ALPHA > 0:
        # mixup 的 target 是混合的 → train_loss 天生比 val_loss 高。
        # 不能直接比大小判斷過擬合,要看 val_loss 自己的走勢有沒有回升。
        print("[train] ⚠️ mixup 開啟 → train_loss 會偏高,不可直接與 val_loss 比較")

    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    history, best_metric, best_epoch = _fit(model, data, criterion, device, ckpt_path)
    print(f"best val_macro_auroc={best_metric:.4f} @ epoch {best_epoch}  -> {ckpt_path}")

    metrics.save_history_csv(history, os.path.join(results_dir, "metrics.csv"))
    metrics.save_curves(history, os.path.join(results_dir, "curves.png"))

    # 載回最佳權重,在獨立 test 集評一次 —— 這才是要回報的數字
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, test_probs, test_labels = collect(
        model, data.test_loader, criterion, device, tta=config.TTA)
    report = metrics.full_report(test_labels, test_probs, data.class_names)

    print(f"\n===== TEST 集評估(best checkpoint @ epoch {ckpt['epoch']}"
          f"{', TTA' if config.TTA else ''})=====")
    print(metrics.format_report(report))
    if config.DEDUP:
        print("⚠️ 已去除重複影像,但仍非 patient-level split → 數字依舊偏樂觀。")
    else:
        print("🚨 DEDUP=False:重複影像橫跨 train/test,以下數字嚴重灌水,不具參考價值。")

    meta = _run_metadata(ckpt["epoch"], best_metric)
    metrics.save_report_json(
        report, os.path.join(results_dir, "test_report.json"), extra=meta)
    metrics.save_confusion_png(
        report, os.path.join(results_dir, "confusion_matrix.png"))
    print(f"\n結果已存到 {results_dir}/:"
          " metrics.csv, curves.png, confusion_matrix.png, test_report.json")

    # 回傳的 report 與存進 json 的完全一致 → crossval 才拿得到 best_epoch 等欄位
    return {**report, **meta}, test_probs, test_labels


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
