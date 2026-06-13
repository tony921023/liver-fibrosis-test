"""資料載入 / transforms / stratified split。

對外只暴露 get_dataloaders(config) -> (train_loader, val_loader, num_classes)。

⚠️ LEAKAGE 警告(務必知道):
這份公開資料的檔名只有流水號(a<number>.jpg),沒有病人 ID,
無法做 patient-level split。這裡用 stratified RANDOM split,
同一位病人的多張影像可能同時落在 train 和 val,造成資訊洩漏。
→ 因此本專案算出的 macro AUROC 會「偏樂觀」,不可當成真實臨床表現。
真正的 patient-level 評估留給未來的臨床資料(碩論研究)。
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


def _build_transforms(config):
    """train 加輕量 augmentation,val 只做必要的 resize + normalize。"""
    normalize = transforms.Normalize(mean=config.MEAN, std=config.STD)
    train_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def _map_targets(targets, task):
    """依 config.TASK 把原始 5 分期 label 轉成目標 label。

    預留擴充點:目前支援 multiclass(直通)與 binary_geF2(>=F2 視為陽性)。
    回傳 (mapped_targets, num_classes)。
    """
    if task == "multiclass":
        return list(targets), 5
    if task == "binary_geF2":
        # F0,F1 -> 0(無顯著纖維化);F2,F3,F4 -> 1(顯著纖維化)
        return [int(t >= 2) for t in targets], 2
    raise ValueError(f"未知的 TASK: {task!r}")


class _RelabelDataset(Subset):
    """在 Subset 之上套用 label 轉換(供 binary 等任務切換用)。

    multiclass 時 mapped 與原 label 相同,等同透明包一層。
    """

    def __init__(self, dataset, indices, mapped_targets):
        super().__init__(dataset, indices)
        self._mapped = mapped_targets  # 與「原始 dataset」等長,用原始 index 取值

    def __getitem__(self, i):
        x, _ = super().__getitem__(i)
        original_index = self.indices[i]
        return x, self._mapped[original_index]


def get_dataloaders(config):
    """回傳 (train_loader, val_loader, num_classes)。

    用兩個共用同一份影像、但 transform 不同的 ImageFolder,
    再以同一組 stratified 索引各自 Subset → train/val 拿到各自的 augmentation。
    """
    train_tf, eval_tf = _build_transforms(config)

    # 同一個 root,兩種 transform;檔案掃描順序一致,故 index 對得起來
    train_base = datasets.ImageFolder(config.DATA_DIR, transform=train_tf)
    eval_base = datasets.ImageFolder(config.DATA_DIR, transform=eval_tf)

    mapped_targets, num_classes = _map_targets(train_base.targets, config.TASK)

    indices = list(range(len(train_base)))
    stratify = mapped_targets if config.STRATIFY else None
    train_idx, val_idx = train_test_split(
        indices,
        test_size=config.VAL_SPLIT,
        random_state=config.SEED,   # 固定 SEED → split 可重現
        shuffle=True,
        stratify=stratify,
    )

    train_set = _RelabelDataset(train_base, train_idx, mapped_targets)
    val_set = _RelabelDataset(eval_base, val_idx, mapped_targets)

    pin = torch.cuda.is_available()   # MPS 不支援 pin_memory,只在 CUDA 開
    train_loader = DataLoader(
        train_set, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_set, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=pin,
    )

    print(f"[dataset] task={config.TASK}  classes={train_base.classes}  "
          f"num_classes={num_classes}")
    print(f"[dataset] total={len(train_base)}  train={len(train_idx)}  val={len(val_idx)}")
    return train_loader, val_loader, num_classes
