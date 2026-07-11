"""模型定義:transfer learning,backbone 可換。

  build_model(num_classes, config) -> nn.Module
  set_backbone_trainable(model, config, trainable)   # 兩階段微調用
  param_groups(model, config, backbone_lr, head_lr)  # differential LR

backbone 與 head 刻意分開,之後擴充可直接重用 backbone:
  - 多模態融合:影像 feature 與 tabular feature concat 後再進 head
  - attention-MIL:同一病人多張影像各自過 backbone,attention pooling 後進 head
接入點在 _replace_head。
"""

import torch.nn as nn
from torchvision import models


# backbone -> (建構式, 預訓練權重, 分類頭的屬性名)。加新 backbone 就在這裡加一行。
# resnet 的 head 是單一 Linear;efficientnet/convnext 的是 Sequential(含 Dropout/
# LayerNorm/Flatten)→ 替換時要找出最後一層 Linear,見 _replace_head。
_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, "fc"),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, "fc"),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, "fc"),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2, "fc"),
    "efficientnet_b0": (models.efficientnet_b0,
                        models.EfficientNet_B0_Weights.IMAGENET1K_V1, "classifier"),
    "convnext_tiny": (models.convnext_tiny,
                      models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, "classifier"),
}


def _head_attr(config):
    if config.BACKBONE not in _BACKBONES:
        raise ValueError(f"未支援的 backbone: {config.BACKBONE!r};可選 {list(_BACKBONES)}")
    return _BACKBONES[config.BACKBONE][2]


def _head_module(model, config):
    return getattr(model, _head_attr(config))


def _new_head(in_features, num_classes, dropout):
    """分類頭:dropout > 0 時在 Linear 前加一層 Dropout(對抗過擬合)。"""
    if dropout <= 0:
        return nn.Linear(in_features, num_classes)
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))


def _replace_head(model, head_attr, num_classes, dropout):
    """把 head 裡的最後一層 Linear 換成輸出 num_classes 的新 head。

    head 本身就是 Linear(resnet)→ 直接整個換掉;
    head 是 Sequential(efficientnet/convnext)→ 只換裡面最後一層 Linear,
    保留前面的 Dropout / LayerNorm / Flatten(它們是 pretrained 架構的一部分)。
    """
    head = getattr(model, head_attr)
    if isinstance(head, nn.Linear):
        setattr(model, head_attr, _new_head(head.in_features, num_classes, dropout))
        return

    for i in range(len(head) - 1, -1, -1):
        if isinstance(head[i], nn.Linear):
            head[i] = _new_head(head[i].in_features, num_classes, dropout)
            return
    raise ValueError(f"在 {head_attr} 裡找不到 nn.Linear,無法替換分類頭")


def build_model(num_classes, config):
    if config.BACKBONE not in _BACKBONES:
        raise ValueError(f"未支援的 backbone: {config.BACKBONE!r};可選 {list(_BACKBONES)}")
    ctor, weights_enum, head_attr = _BACKBONES[config.BACKBONE]
    weights = weights_enum if config.PRETRAINED else None
    model = ctor(weights=weights)

    # === 擴充接入點 ===
    # 多模態:把 head 的 in_features 改成 in_features + tabular_dim,並在 forward 前
    #         將影像 feature 與 tabular 向量 concat 後丟進這個 head。
    # attention-MIL:head 之前先做 attention pooling 聚合多視角 feature。
    _replace_head(model, head_attr, num_classes, config.DROPOUT)  # 新 head 預設 requires_grad=True

    # 凍結/解凍由 train.py 依兩階段流程控制(見 set_backbone_trainable)
    return model


def set_backbone_trainable(model, config, trainable):
    """設定 backbone(head 以外所有參數)是否需要梯度。

    暖身階段 trainable=False(只訓 head);解凍微調階段 trainable=True。
    head 永遠保持可訓練。
    """
    head_param_ids = {id(p) for p in _head_module(model, config).parameters()}
    for p in model.parameters():
        if id(p) not in head_param_ids:
            p.requires_grad = trainable


def param_groups(model, config, backbone_lr, head_lr):
    """回傳 differential LR 的參數分組:backbone 用小 LR、head 用大 LR。

    只收 requires_grad=True 的參數,所以暖身階段(backbone 凍結)會自動
    只剩 head 那組。
    """
    head_param_ids = {id(p) for p in _head_module(model, config).parameters()}
    backbone_params, head_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (head_params if id(p) in head_param_ids else backbone_params).append(p)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    return groups
