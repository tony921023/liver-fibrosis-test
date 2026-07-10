"""資料載入 / transforms / dedup / stratified split。

對外只暴露 get_dataloaders(config) -> DataBundle。

⚠️ LEAKAGE 警告(務必知道,有兩層):

【第一層:完全重複的影像】(DEDUP=True 可解決)
  這份公開資料的 6323 個檔案裡,只有 1536 張「位元組不重複」的影像
  ——平均每張被複製約 4 次,最多的一張出現 18 次。
  隨機 split 會讓同一張圖的複製品同時落在 train 與 test,模型用背的就滿分
  (去重前 test macro AUROC = 0.9975,F0/F4 recall = 1.000,正是複製倍率最高的兩類)。
  → DEDUP=True 依檔案內容 hash 分組,每組只留一張代表,再做 split。
  好消息:1536 個 hash 全部只對應單一類別,沒有跨類別重複,標籤本身沒有矛盾。

【第二層:patient-level leakage】(這份資料無解)
  檔名只有「字母前綴 + 流水號」(a1000.jpg / I2079.jpg / z9945.jpg …),沒有病人 ID,
  前綴字母經抽查也不對應病人,無法做 patient-level split。
  即使去重後,同一位病人的不同切面/不同幀仍可能分散在不同 split。
  → 因此本專案的指標仍然「偏樂觀」,不可當成真實臨床表現。
  切出獨立 test 集只消除「用 val 既選模型又打分」的偏差;
  DEDUP 只消除「完全重複」的洩漏;patient-level leakage 兩者都消不掉。
  真正的評估留給未來臨床資料(碩論研究)——屆時把 _dedup_indices 換成
  依病人 ID 分組的 GroupShuffleSplit 即可。
"""

import hashlib
from dataclasses import dataclass
from typing import List, Optional

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


@dataclass
class DataBundle:
    """get_dataloaders 的回傳值。用 dataclass 而非 tuple,之後加欄位不會弄亂呼叫端。"""
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    class_names: List[str]
    # 依 train split 頻率算;CLASS_WEIGHTS="none" 時為 None(專案跑在 Python 3.9,不能用 X | None)
    class_weights: Optional[torch.Tensor]


def _build_transforms(config):
    """train 加 augmentation,val/test 只做必要的 resize + normalize。

    超音波影像的限制:灰階(不動 hue/saturation)、上下方向有意義(不做垂直翻轉)、
    探頭角度會有小幅變化(允許小角度旋轉)、增益設定因機器而異(允許亮度/對比抖動)。
    """
    normalize = transforms.Normalize(mean=config.MEAN, std=config.STD)
    size = (config.IMG_SIZE, config.IMG_SIZE)

    if config.AUG_STRENGTH == "basic":
        train_ops = [
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
        ]
    elif config.AUG_STRENGTH == "strong":
        train_ops = [
            # scale 下限 0.8:再裁更兇會把病灶區域裁掉
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    else:
        raise ValueError(f"未知的 AUG_STRENGTH: {config.AUG_STRENGTH!r};可選 'basic'/'strong'")

    train_tf = transforms.Compose(train_ops + [transforms.ToTensor(), normalize])
    eval_tf = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def _content_hash(path, chunk_size=1 << 20):
    """檔案內容的 md5。用來抓「位元組完全相同」的重複影像。

    注意:只抓得到 exact duplicate。若同一張圖被重新壓縮 / 縮放過(near-duplicate),
    hash 會不同、抓不到 —— 那需要 perceptual hash,目前先不做。
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _dedup_indices(samples):
    """依內容 hash 分組,每組只保留第一個出現的 index。

    samples 是 ImageFolder.samples,已按路徑排序 → 「第一個」是決定性的,
    不受檔案系統列舉順序影響,同一份資料每次跑都得到同一組代表。
    """
    seen = {}
    for i, (path, _) in enumerate(samples):
        seen.setdefault(_content_hash(path), i)
    return sorted(seen.values())


def _class_weights(labels, num_classes, mode):
    """依 train split 的類別頻率算權重:w_c = N / (C * n_c),再正規化成平均 1。

    少數類權重高。這份資料去重後很平衡,權重接近全 1;
    留著是為了未來臨床資料(F4 通常遠少於 F0)。
    """
    if mode == "none":
        return None
    if mode != "auto":
        raise ValueError(f"未知的 CLASS_WEIGHTS: {mode!r};可選 'auto'/'none'")

    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    # 某類在 train 完全缺漏時,權重設 0 而非 inf(不會有樣本觸發它,但避免 NaN)
    weights = torch.where(counts > 0, len(labels) / (num_classes * counts.clamp(min=1)),
                          torch.zeros_like(counts))
    return weights / weights[weights > 0].mean()


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

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]


def _stratified_three_way(indices, labels, val_frac, test_frac, seed, stratify):
    """先切出 test,再從剩下的切 val,兩者比例皆相對「全體」。

    labels 與 indices 位置對齊(labels[k] 是 indices[k] 的 label)。
    dedup 後 indices 不再是 0..N-1,所以第二刀要用 index->label 的查表,
    不能寫成 labels[i]。
    """
    label_of = dict(zip(indices, labels))
    strat = labels if stratify else None
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_frac, random_state=seed,
        shuffle=True, stratify=strat,
    )
    # val 占全體的 val_frac → 占剩餘的 val_frac / (1 - test_frac)
    rel_val = val_frac / (1.0 - test_frac)
    strat_tv = [label_of[i] for i in train_val_idx] if stratify else None
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=rel_val, random_state=seed,
        shuffle=True, stratify=strat_tv,
    )
    return train_idx, val_idx, test_idx


def get_dataloaders(config):
    """回傳 DataBundle。

    用兩個共用同一份影像、但 transform 不同的 ImageFolder:
    train 拿 augmentation 版,val/test 拿 eval 版;再以同一組 stratified
    索引各自 Subset。

    DEDUP=True 時,split 只在「去重後的代表影像」上做 —— 被丟掉的複製品
    不會出現在任何 split,所以不可能有同一張圖橫跨 train/test。
    """
    train_tf, eval_tf = _build_transforms(config)

    train_base = datasets.ImageFolder(config.DATA_DIR, transform=train_tf)
    eval_base = datasets.ImageFolder(config.DATA_DIR, transform=eval_tf)

    mapped_targets, num_classes, class_names = _map_targets(
        train_base.targets, train_base.classes, config.TASK)

    if config.DEDUP:
        indices = _dedup_indices(train_base.samples)
        print(f"[dataset] dedup: {len(train_base)} 個檔案 -> "
              f"{len(indices)} 張不重複影像(丟掉 {len(train_base) - len(indices)} 張複製品)")
    else:
        indices = list(range(len(train_base)))
        print("[dataset] ⚠️ DEDUP=False:重複影像會橫跨 train/test,指標將嚴重灌水")

    # stratify 要拿「這些 index 對應的 label」,不能整份 mapped_targets 傳進去
    split_labels = [mapped_targets[i] for i in indices]
    train_idx, val_idx, test_idx = _stratified_three_way(
        indices, split_labels,
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

    train_labels = [mapped_targets[i] for i in train_idx]
    weights = _class_weights(train_labels, num_classes, config.CLASS_WEIGHTS)

    print(f"[dataset] task={config.TASK}  classes={class_names}  num_classes={num_classes}")
    print(f"[dataset] train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    print(f"[dataset] train 類別分布={torch.bincount(torch.tensor(train_labels), minlength=num_classes).tolist()}")
    if weights is not None:
        print(f"[dataset] class_weights={[round(w, 3) for w in weights.tolist()]}")

    return DataBundle(
        train_loader=_loader(train_set, shuffle=True),
        val_loader=_loader(val_set, shuffle=False),
        test_loader=_loader(test_set, shuffle=False),
        num_classes=num_classes,
        class_names=class_names,
        class_weights=weights,
    )
