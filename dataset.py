"""資料載入 / transforms / stratified split。

對外只暴露 get_dataloaders(config) ->
    (train_loader, val_loader, test_loader, num_classes, class_names)。

⚠️ LEAKAGE 警告(務必知道):
這份公開資料的檔名只有流水號(a<number>.jpg),沒有病人 ID,
無法做 patient-level split。這裡用 stratified RANDOM split,
同一位病人的多張影像可能同時落在不同 split,造成資訊洩漏。
→ 因此本專案算出的指標(含 test 集)都會「偏樂觀」,不可當成真實臨床表現。
切出獨立 test 集只能消除「用 val 同時選模型又打分」的偏差,
消不掉 patient-level leakage。真正的評估留給未來臨床資料(碩論研究)。
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


def _build_transforms(config):
    """train 加輕量 augmentation,val/test 只做必要的 resize + normalize。"""
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


def _map_targets(targets, classes, task):
    """依 config.TASK 把原始 5 分期 label 轉成目標 label。

    回傳 (mapped_targets, num_classes, class_names)。
    class_names 供 confusion matrix / 報告顯示用。
    """
    if task == "multiclass":
        return list(targets), len(classes), list(classes)
    if task == "binary_geF2":
        # F0,F1 -> 0(無顯著纖維化);F2,F3,F4 -> 1(顯著纖維化)
        return [int(t >= 2) for t in targets], 2, ["<F2", ">=F2"]
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


def _stratified_three_way(indices, labels, val_frac, test_frac, seed, stratify):
    """先切出 test,再從剩下的切 val,兩者比例皆相對「全體」。"""
    strat = labels if stratify else None
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_frac, random_state=seed,
        shuffle=True, stratify=strat,
    )
    # val 占全體的 val_frac → 占剩餘的 val_frac / (1 - test_frac)
    rel_val = val_frac / (1.0 - test_frac)
    strat_tv = [labels[i] for i in train_val_idx] if stratify else None
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=rel_val, random_state=seed,
        shuffle=True, stratify=strat_tv,
    )
    return train_idx, val_idx, test_idx


def get_dataloaders(config):
    """回傳 (train_loader, val_loader, test_loader, num_classes, class_names)。

    用兩個共用同一份影像、但 transform 不同的 ImageFolder:
    train 拿 augmentation 版,val/test 拿 eval 版;再以同一組 stratified
    索引各自 Subset。
    """
    train_tf, eval_tf = _build_transforms(config)

    train_base = datasets.ImageFolder(config.DATA_DIR, transform=train_tf)
    eval_base = datasets.ImageFolder(config.DATA_DIR, transform=eval_tf)

    mapped_targets, num_classes, class_names = _map_targets(
        train_base.targets, train_base.classes, config.TASK)

    indices = list(range(len(train_base)))
    train_idx, val_idx, test_idx = _stratified_three_way(
        indices, mapped_targets,
        val_frac=config.VAL_SPLIT, test_frac=config.TEST_SPLIT,
        seed=config.SEED, stratify=config.STRATIFY,
    )

    train_set = _RelabelDataset(train_base, train_idx, mapped_targets)   # augmentation
    val_set = _RelabelDataset(eval_base, val_idx, mapped_targets)        # eval transform
    test_set = _RelabelDataset(eval_base, test_idx, mapped_targets)      # eval transform

    pin = torch.cuda.is_available()   # MPS 不支援 pin_memory,只在 CUDA 開

    def _loader(ds, shuffle):
        return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=shuffle,
                          num_workers=config.NUM_WORKERS, pin_memory=pin)

    train_loader = _loader(train_set, shuffle=True)
    val_loader = _loader(val_set, shuffle=False)
    test_loader = _loader(test_set, shuffle=False)

    print(f"[dataset] task={config.TASK}  classes={class_names}  num_classes={num_classes}")
    print(f"[dataset] total={len(train_base)}  "
          f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    return train_loader, val_loader, test_loader, num_classes, class_names
