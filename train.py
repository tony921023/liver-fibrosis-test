"""訓練迴圈 + 評估(macro AUROC)。

跑法:  python train.py
超參數集中在 config.py;要換 backbone / epoch / 預測目標,改 config 即可。

⚠️ 評估數字偏樂觀:split 非 patient-level,詳見 dataset.py 的 LEAKAGE 警告。
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from dataset import get_dataloaders   # 來自 dataset.py
from model import build_model         # 來自 model.py
import config                         # 超參數集中放這


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


def main():
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    device = get_device()
    print("device:", device, "| torch:", torch.__version__)

    train_loader, val_loader, num_classes = get_dataloaders(config)
    model = build_model(num_classes, config).to(device)
    criterion = nn.CrossEntropyLoss()
    # 只把「需要梯度」的參數丟進 optimizer(凍結 backbone 時即只訓 head)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config.LR)

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, auroc = evaluate(model, val_loader, criterion, num_classes, device)
        print(f"epoch {epoch:2d}: train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_macro_auroc={auroc:.4f}")

    # 之後可加:torch.save(model.state_dict(), "checkpoints/best.pt")
    # (checkpoints/ 與 *.pt 已被 .gitignore 擋掉,不會誤 commit)


if __name__ == "__main__":
    main()
