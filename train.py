"""訓練迴圈 + 評估(macro AUROC)。

訓練骨架:
  Phase 1「暖身」  凍結 backbone,只訓 head(config.WARMUP_EPOCHS 輪)
  Phase 2「微調」  解凍 backbone,differential LR + LR scheduler 微調到 config.EPOCHS
  全程       監看 val 指標 → early stopping + 存最佳 checkpoint

跑法:  python train.py
超參數集中在 config.py;要換 backbone / 排程 / 預測目標,改 config 即可。

⚠️ 評估數字偏樂觀:split 非 patient-level,詳見 dataset.py 的 LEAKAGE 警告。
"""

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from dataset import get_dataloaders            # 來自 dataset.py
from model import build_model, set_backbone_trainable, param_groups  # 來自 model.py
import config                                  # 超參數集中放這


def get_device():
    # device 自動偵測:同一份 code 在 Mac(MPS)和 Colab(CUDA)都能跑
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, num_classes, device):
    """回傳 (val_loss, macro_auroc)。"""
    model.eval()
    total_loss = 0.0
    probs, labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(y.cpu().numpy())
    val_loss = total_loss / len(loader.dataset)

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    # 明確帶 labels=range(num_classes),避免 val 某類別剛好缺漏時報錯;
    # 二元任務取陽性類機率即可。
    if num_classes == 2:
        auroc = roc_auc_score(labels, probs[:, 1])
    else:
        auroc = roc_auc_score(
            labels, probs, multi_class="ovr", average="macro",
            labels=list(range(num_classes)),
        )
    return val_loss, auroc


def _build_optimizer(model):
    """param_groups 只收 requires_grad 的參數:
    暖身階段 backbone 凍結 → 只剩 head;微調階段含 backbone(differential LR)。
    """
    groups = param_groups(model, config, backbone_lr=config.BACKBONE_LR, head_lr=config.HEAD_LR)
    return torch.optim.Adam(groups)


def _build_scheduler(optimizer, finetune_epochs):
    name = config.SCHEDULER
    if name == "cosine":
        # T_max 設為 Phase 2 的輪數,讓 LR 在微調期間退火到接近 0
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, finetune_epochs))
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2)
    if name == "none":
        return None
    raise ValueError(f"未知的 SCHEDULER: {name!r}")


def main():
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    device = get_device()
    print("device:", device, "| torch:", torch.__version__)

    train_loader, val_loader, num_classes = get_dataloaders(config)
    model = build_model(num_classes, config).to(device)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(config.CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.CKPT_DIR, "best.pt")

    # --- Phase 1:暖身(凍結 backbone,只訓 head)---
    warmup = max(0, min(config.WARMUP_EPOCHS, config.EPOCHS))
    set_backbone_trainable(model, config, trainable=False)
    optimizer = _build_optimizer(model)
    scheduler = None  # 暖身階段不排程,LR 維持 HEAD_LR
    phase = "warmup"

    best_metric = -float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, config.EPOCHS + 1):
        # --- 進入 Phase 2:解凍 backbone,換 differential-LR optimizer + scheduler ---
        if epoch == warmup + 1:
            set_backbone_trainable(model, config, trainable=True)
            optimizer = _build_optimizer(model)
            scheduler = _build_scheduler(optimizer, finetune_epochs=config.EPOCHS - warmup)
            phase = "finetune"
            print(f"--- unfreeze backbone @ epoch {epoch} "
                  f"(backbone_lr={config.BACKBONE_LR}, head_lr={config.HEAD_LR}) ---")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, auroc = evaluate(model, val_loader, criterion, num_classes, device)

        if scheduler is not None:
            scheduler.step(auroc) if config.SCHEDULER == "plateau" else scheduler.step()

        improved = auroc > best_metric
        flag = ""
        if improved:
            best_metric, best_epoch, epochs_no_improve = auroc, epoch, 0
            torch.save(
                # metric 轉 Python float:roc_auc_score 回傳 numpy 標量,
                # 否則 torch>=2.6 的 weights_only=True 預設載入會失敗。
                {"model_state": model.state_dict(), "epoch": epoch,
                 "metric": float(auroc), "num_classes": num_classes,
                 "backbone": config.BACKBONE, "task": config.TASK},
                ckpt_path,
            )
            flag = "  <- best (saved)"
        else:
            epochs_no_improve += 1

        print(f"epoch {epoch:2d} [{phase:8s}]: train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_macro_auroc={auroc:.4f}{flag}")

        # --- early stopping ---
        if config.EARLY_STOP_PATIENCE > 0 and epochs_no_improve >= config.EARLY_STOP_PATIENCE:
            print(f"early stop @ epoch {epoch} "
                  f"(no improvement for {config.EARLY_STOP_PATIENCE} epochs)")
            break

    print(f"best val_macro_auroc={best_metric:.4f} @ epoch {best_epoch}  "
          f"-> {ckpt_path}")


if __name__ == "__main__":
    main()
