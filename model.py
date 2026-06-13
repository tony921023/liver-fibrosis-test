"""模型定義:transfer learning,backbone 可換。

對外暴露:
  - build_model(num_classes, config) -> nn.Module
  - set_backbone_trainable(model, config, trainable)   # 凍結/解凍 backbone
  - param_groups(model, config, backbone_lr, head_lr)  # differential LR 參數分組

設計上把「特徵抽取(backbone)」與「分類頭(head)」分開,
之後要做下列擴充時,backbone 那段可以直接重用:
  - 多模態融合:把影像 feature 與 tabular feature concat 後再進 head
    (見下方 _replace_head 註解的接入點)
  - attention-MIL(多視角):同一病人多張影像各自過 backbone 取 feature,
    再用 attention pooling 聚合成單一向量後進 head
"""

import torch.nn as nn
from torchvision import models


# torchvision backbone 對應的 weights enum 與「分類頭屬性名」。
# 要支援新的 backbone 時,在這裡加一行即可。
_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, "fc"),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, "fc"),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, "fc"),
}


def _head_attr(config):
    if config.BACKBONE not in _BACKBONES:
        raise ValueError(f"未支援的 backbone: {config.BACKBONE!r};可選 {list(_BACKBONES)}")
    return _BACKBONES[config.BACKBONE][2]


def _head_module(model, config):
    return getattr(model, _head_attr(config))


def build_model(num_classes, config):
    if config.BACKBONE not in _BACKBONES:
        raise ValueError(f"未支援的 backbone: {config.BACKBONE!r};可選 {list(_BACKBONES)}")
    ctor, weights_enum, head_attr = _BACKBONES[config.BACKBONE]
    weights = weights_enum if config.PRETRAINED else None
    model = ctor(weights=weights)

    in_features = getattr(model, head_attr).in_features
    # === 擴充接入點 ===
    # 多模態:把 in_features 改成 in_features + tabular_dim,並在 forward 前
    #         將影像 feature 與 tabular 向量 concat 後丟進這個 head。
    # attention-MIL:head 之前先做 attention pooling 聚合多視角 feature。
    new_head = nn.Linear(in_features, num_classes)
    setattr(model, head_attr, new_head)  # 新 head 預設 requires_grad=True

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
