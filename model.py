"""模型定義:transfer learning,backbone 可換。

對外只暴露 build_model(num_classes, config) -> nn.Module。

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


def _freeze(module):
    for p in module.parameters():
        p.requires_grad = False


def build_model(num_classes, config):
    name = config.BACKBONE
    if name not in _BACKBONES:
        raise ValueError(f"未支援的 backbone: {name!r};可選 {list(_BACKBONES)}")
    ctor, weights_enum, head_attr = _BACKBONES[name]

    weights = weights_enum if config.PRETRAINED else None
    model = ctor(weights=weights)

    # 先(選擇性)凍結整個 backbone,再換上新的、可訓練的 head。
    # baseline 階段 FREEZE_BACKBONE=True → 只訓 head,快又穩;
    # 放大階段設 False → 全網路微調。
    if config.FREEZE_BACKBONE:
        _freeze(model)

    in_features = getattr(model, head_attr).in_features
    # === 擴充接入點 ===
    # 多模態:把 in_features 改成 in_features + tabular_dim,並在 forward 前
    #         將影像 feature 與 tabular 向量 concat 後丟進這個 head。
    # attention-MIL:head 之前先做 attention pooling 聚合多視角 feature。
    new_head = nn.Linear(in_features, num_classes)
    setattr(model, head_attr, new_head)  # 新 head 預設 requires_grad=True

    return model
