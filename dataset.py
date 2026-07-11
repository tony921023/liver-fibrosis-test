"""資料載入 / transforms / dedup / split。

對外只暴露 get_dataloaders(config, fold=None) -> DataBundle。

⚠️ 這份資料有嚴重的 leakage 與來源混淆,所得指標是樂觀上限而非真實表現。
   去重(DEDUP)、遮罩消融(MASK)的來龍去脈見 docs/leakage.md。
   patient-level split 做不到(無病人 ID)—— 臨床資料到手後,把 _dedup_indices
   換成依病人 ID 分組的 StratifiedGroupKFold 即可,其餘流程不用動。
"""

import hashlib
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold


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


class _Mask:
    """把中央方塊(或其補集)塗黑,用來測模型靠組織還是靠來源假影。見 docs/leakage.md。

    "center"     塗黑中央肝實質,只留邊緣/背景
    "periphery"  塗黑邊緣,只留中央組織

    frac = 中央方塊的邊長佔全圖比例。塗黑 = 填 0 = 原始像素的黑色,
    所以要在 ToTensor 之後、normalize 之前套用。
    """

    def __init__(self, mode, frac):
        self.mode, self.frac = mode, frac

    def __call__(self, x):          # x: [C,H,W],已 ToTensor,值域 [0,1]
        _, h, w = x.shape
        ch, cw = round(h * self.frac), round(w * self.frac)
        top, left = (h - ch) // 2, (w - cw) // 2
        x = x.clone()
        if self.mode == "center":
            x[:, top:top + ch, left:left + cw] = 0.0
        elif self.mode == "periphery":
            keep = x[:, top:top + ch, left:left + cw].clone()
            x.zero_()
            x[:, top:top + ch, left:left + cw] = keep
        else:
            raise ValueError(f"未知的 MASK: {self.mode!r};可選 'center'/'periphery'/None")
        return x


def _build_transforms(config):
    """train 加 augmentation,val/test 只做 resize + normalize。

    超音波的限制:灰階(不動 hue)、上下方向有意義(不做垂直翻轉)、
    探頭角度會小幅變化(允許小角度旋轉)、增益因機器而異(允許亮度/對比抖動)。
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

    # 遮罩消融:接在 ToTensor 之後、normalize 之前;train/eval 套用同一個遮罩。
    # MASK 為 None/"none" 時是空 list,完全不影響原本的行為。
    mask_ops = []
    if getattr(config, "MASK", None) and config.MASK != "none":
        mask_ops = [_Mask(config.MASK, frac=config.MASK_FRAC)]
        # 遮罩區塊固定在中央,若 train 還做隨機裁切/旋轉,遮罩相對組織的位置會亂跑
        # → 消融時把 train augmentation 降到 basic,確保兩組實驗只差在「遮罩」。
        train_ops = [transforms.Resize(size), transforms.RandomHorizontalFlip()]

    train_tf = transforms.Compose(
        train_ops + [transforms.ToTensor()] + mask_ops + [normalize])
    eval_tf = transforms.Compose(
        [transforms.Resize(size), transforms.ToTensor()] + mask_ops + [normalize])
    return train_tf, eval_tf


def _content_hash(path, chunk_size=1 << 20):
    """檔案內容的 md5,抓位元組完全相同的重複。重新壓縮過的近重複抓不到 → _perceptual_hash。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _perceptual_hash(path):
    """dHash:縮到 9x8 灰階、比較相鄰像素亮度 → 64-bit 指紋。抓重新壓縮/縮放過的近重複。

    ⚠️ 超音波都是「黑背景 + 扇形區」,低頻結構相近,偶爾會讓兩張「其實不同組織」的圖
    (F0/F4)撞到同一個 dHash → 所以只在同類別內合併,見 _dedup_indices。
    """
    from PIL import Image  # 延後匯入:只有開 perceptual dedup 才需要
    img = Image.open(path).convert("L").resize((9, 8), Image.BILINEAR)
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.packbits(bits).tobytes()


def _dedup_indices(samples, mode="exact"):
    """依內容分組,每組只留第一個出現的 index。

    samples(= ImageFolder.samples)已按路徑排序 → 「第一個」是決定性的,
    不受檔案系統列舉順序影響,同一份資料每次跑都得到同一組代表。
    """
    seen = {}
    for i, (path, _) in enumerate(samples):
        seen.setdefault(_content_hash(path), i)
    exact_idx = sorted(seen.values())

    if mode == "exact":
        return exact_idx
    if mode != "perceptual":
        raise ValueError(f"未知的 DEDUP 模式: {mode!r};可選 True/'exact'/'perceptual'/False")

    # key 帶 label → 只在同類別內合併,避免 dHash 碰撞誤刪(見 _perceptual_hash)
    seen_p = {}
    for i in exact_idx:
        path, label = samples[i]
        seen_p.setdefault((label, _perceptual_hash(path)), i)
    return sorted(seen_p.values())


def _class_weights(labels, num_classes, mode):
    """w_c = N / (C * n_c),正規化成平均 1 → 少數類權重高。

    這份資料去重後很平衡(權重接近全 1);留著是為了未來臨床資料(通常很不平衡)。
    """
    if mode == "none":
        return None
    if mode != "auto":
        raise ValueError(f"未知的 CLASS_WEIGHTS: {mode!r};可選 'auto'/'none'")

    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    # 某類在 train 完全缺漏時權重設 0 而非 inf(避免 NaN)
    weights = torch.where(counts > 0, len(labels) / (num_classes * counts.clamp(min=1)),
                          torch.zeros_like(counts))
    return weights / weights[weights > 0].mean()


def _map_targets(targets, classes, task):
    """把原始 5 分期 label 轉成目標 label。回傳 (mapped, num_classes, class_names)。"""
    if task == "multiclass":
        return list(targets), len(classes), list(classes)
    if task == "binary_geF2":
        return [int(t >= 2) for t in targets], 2, ["<F2", ">=F2"]
    raise ValueError(f"未知的 TASK: {task!r}")


class _RelabelDataset(Subset):
    """Subset + label 轉換。multiclass 時 mapped 與原 label 相同,等同透明包一層。"""

    def __init__(self, dataset, indices, mapped_targets):
        super().__init__(dataset, indices)
        self._mapped = mapped_targets  # 與「原始 dataset」等長,用原始 index 取值

    def __getitem__(self, i):
        x, _ = super().__getitem__(i)
        return x, self._mapped[self.indices[i]]

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]


def _stratified_three_way(indices, labels, val_frac, test_frac, seed, stratify):
    """先切 test,再從剩下的切 val;兩者比例皆相對「全體」。

    labels 與 indices 位置對齊。⚠️ 去重後 indices 不再是 0..N-1,
    所以第二刀要用 index->label 查表,不能寫成 labels[i]。
    """
    label_of = dict(zip(indices, labels))
    strat = labels if stratify else None
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_frac, random_state=seed,
        shuffle=True, stratify=strat,
    )
    rel_val = val_frac / (1.0 - test_frac)   # val 占全體 val_frac → 占剩餘的這麼多
    strat_tv = [label_of[i] for i in train_val_idx] if stratify else None
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=rel_val, random_state=seed,
        shuffle=True, stratify=strat_tv,
    )
    return train_idx, val_idx, test_idx


def _kfold_split(indices, labels, fold, n_folds, val_frac, seed):
    """第 fold 折(0-based)當 test,其餘再切出 val。

    n_folds 折輪流當 test → 每張影像剛好被評估一次,合起來就是全體的 out-of-fold 估計。
    """
    label_of = dict(zip(indices, labels))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_val_pos, test_pos = list(skf.split(indices, labels))[fold]

    train_val_idx = [indices[p] for p in train_val_pos]
    test_idx = [indices[p] for p in test_pos]

    rel_val = val_frac / (1.0 - 1.0 / n_folds)   # val 仍占全體的 val_frac
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=rel_val, random_state=seed,
        shuffle=True, stratify=[label_of[i] for i in train_val_idx],
    )
    return train_idx, val_idx, test_idx


# 相容布林 True/False,也接受環境變數的字串。
# ⚠️ env 一律是字串 → "False"/"none" 都是 truthy,不正規化會反而關不掉去重。
_DEDUP_ALIAS = {
    True: "exact", "true": "exact", "exact": "exact",
    "perceptual": "perceptual",
    False: None, "false": None, "none": None, "": None,
}


def _dedup_mode(value):
    key = value.lower() if isinstance(value, str) else value
    if key not in _DEDUP_ALIAS:
        raise ValueError(f"未知的 DEDUP: {value!r};可選 'exact'/'perceptual'/False")
    return _DEDUP_ALIAS[key]


def _select_indices(train_base, config):
    """挑出要參與 split 的 index —— 去重後就只剩代表影像。

    因為被丟掉的複製品不會出現在任何 split,所以同一張圖不可能橫跨 train/test。
    """
    mode = _dedup_mode(config.DEDUP)
    if not mode:
        print("[dataset] ⚠️ DEDUP=False:重複影像會橫跨 train/test,指標將嚴重灌水")
        return list(range(len(train_base)))

    indices = _dedup_indices(train_base.samples, mode=mode)
    print(f"[dataset] dedup({mode}): {len(train_base)} 個檔案 -> "
          f"{len(indices)} 張不重複影像(丟掉 {len(train_base) - len(indices)} 張)")
    return indices


def get_dataloaders(config, fold=None):
    """fold=None → 單一 stratified 三分;fold=0..N-1 → 該折當 test 的 k-fold split。

    用兩個共用同一份影像、但 transform 不同的 ImageFolder(train 有 augmentation,
    val/test 沒有),再以同一組索引各自 Subset。
    """
    train_tf, eval_tf = _build_transforms(config)
    train_base = datasets.ImageFolder(config.DATA_DIR, transform=train_tf)
    eval_base = datasets.ImageFolder(config.DATA_DIR, transform=eval_tf)

    mapped_targets, num_classes, class_names = _map_targets(
        train_base.targets, train_base.classes, config.TASK)

    indices = _select_indices(train_base, config)
    split_labels = [mapped_targets[i] for i in indices]   # 只取這些 index 的 label

    if fold is None:
        train_idx, val_idx, test_idx = _stratified_three_way(
            indices, split_labels,
            val_frac=config.VAL_SPLIT, test_frac=config.TEST_SPLIT,
            seed=config.SEED, stratify=config.STRATIFY,
        )
    else:
        train_idx, val_idx, test_idx = _kfold_split(
            indices, split_labels, fold=fold, n_folds=config.N_FOLDS,
            val_frac=config.VAL_SPLIT, seed=config.SEED,
        )
        print(f"[dataset] cross-validation: fold {fold + 1}/{config.N_FOLDS}")

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
