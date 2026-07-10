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
# resnet 的 head 是單一 nn.Linear;efficientnet/convnext 的 head 是 nn.Sequential
#(裡面還有 Dropout / LayerNorm / Flatten),所以替換要找出最後一層 Linear,見 _replace_head。
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


def _replace_head(model, head_attr, num_classes):
    """把 head 裡的最後一層 Linear 換成輸出 num_classes 的新 Linear。

    head 本身就是 Linear(resnet)→ 直接整個換掉;
    head 是 Sequential(efficientnet/convnext)→ 只換裡面最後一層 Linear,
    保留前面的 Dropout / LayerNorm / Flatten(它們是 pretrained 架構的一部分)。
    """
    head = getattr(model, head_attr)
    if isinstance(head, nn.Linear):
        setattr(model, head_attr, nn.Linear(head.in_features, num_classes))
        return

    for i in range(len(head) - 1, -1, -1):
        if isinstance(head[i], nn.Linear):
            head[i] = nn.Linear(head[i].in_features, num_classes)
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
    _replace_head(model, head_attr, num_classes)  # 新 head 預設 requires_grad=True

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
