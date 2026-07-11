"""Grad-CAM 混淆稽核:看模型在看「肝實質」還是「來源假影」。

    python explain.py                  # 從 DATA_DIR 每類抽樣
    python explain.py a1000.jpg        # 指定影像
    CKPT=... EXPLAIN_DIR=... PER_CLASS=6 python explain.py

這份資料沒有病人 ID,測試分數是被 leakage 與來源混淆灌水的上限,無法判斷模型有沒有
作弊(F0 有 198/199 張來自 'a' 前綴)。不能信分數時,唯一能驗證的是「模型在看哪裡」。
背景見 docs/leakage.md。

**邊緣佔比(border ratio)** = Grad-CAM 熱圖落在外框 20% 的能量比例。
肝實質在畫面中央,文字標註 / 探頭刻度 / 扇形邊緣多半在四周 →
佔比高代表模型盯著假影。重點看 F0 是否明顯高於其他類。

比遮罩消融(config.MASK)弱 —— 只說「看哪裡」,不說「沒有組織能不能活」,但便宜。
"""

import os
import sys
import re
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import transforms

from model import build_model
from train import get_device
import config


def _target_layer(model, backbone):
    """Grad-CAM 的目標層:最後一個還保有空間維度的 conv/feature block。"""
    if backbone.startswith("resnet"):
        return model.layer4
    if backbone.startswith("efficientnet") or backbone.startswith("convnext"):
        return model.features
    raise ValueError(f"未支援的 backbone Grad-CAM: {backbone!r}")


def _model_config(backbone, dropout):
    """給 build_model 用的區域設定。

    不直接改 config 的全域變數 —— 那會污染整個 process,之後任何讀 config 的地方
    都會拿到被偷改的值。
    """
    return SimpleNamespace(BACKBONE=backbone, DROPOUT=dropout,
                           PRETRAINED=False)   # 權重從 checkpoint 載,不必下載預訓練的


def _load_model(ckpt_path, device):
    """載 checkpoint 並重建模型,回傳 (model, ckpt)。

    checkpoint 沒存 dropout,而 head 有沒有 dropout 會改變 state_dict 的 key
    (fc.weight vs fc.1.weight)→ 兩種都試,哪個載得進去就用哪個。
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    last_err = None
    for dropout in (config.DROPOUT, 0.0 if config.DROPOUT > 0 else 0.3):
        model = build_model(ckpt["num_classes"],
                            _model_config(ckpt["backbone"], dropout))
        try:
            model.load_state_dict(ckpt["model_state"])
        except RuntimeError as e:
            last_err = e
            continue
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(True)      # Grad-CAM 要把梯度回傳到 activation
        return model, ckpt

    raise RuntimeError(f"無法載入 checkpoint(head 結構對不上):{last_err}")


class GradCAM:
    """對單一 conv 目標層做 Grad-CAM。用 forward hook 存 activation 並 retain_grad,
    backward 後從 activation.grad 取梯度,不依賴 backward hook 的版本差異。"""

    def __init__(self, model, target_layer):
        self.model = model
        self.activation = None
        self._handle = target_layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        out.retain_grad()
        self.activation = out

    def __call__(self, x, class_idx=None):
        """x: [1,C,H,W]。回傳 (cam[H,W] in [0,1], pred_idx, prob[num_classes])。"""
        logits = self.model(x)                       # 建圖
        prob = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        pred = int(logits.argmax(1).item())
        target = pred if class_idx is None else class_idx

        self.model.zero_grad(set_to_none=True)
        logits[0, target].backward()

        A = self.activation[0]                        # [C,h,w]
        grads = self.activation.grad[0]               # [C,h,w]
        weights = grads.mean(dim=(1, 2))              # [C] 每個 channel 的重要度
        cam = F.relu((weights[:, None, None] * A).sum(0))  # [h,w]

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)                # 正規化到 [0,1]
        cam = F.interpolate(cam[None, None], size=x.shape[2:],
                            mode="bilinear", align_corners=False)[0, 0]
        return cam.detach().cpu().numpy(), pred, prob

    def close(self):
        self._handle.remove()


def _border_ratio(cam, frac=0.2):
    """Grad-CAM 能量落在「外框 frac 寬」的佔比。高 = 盯著邊緣/假影。"""
    h, w = cam.shape
    bh, bw = max(1, round(h * frac)), max(1, round(w * frac))
    mask = np.ones_like(cam, dtype=bool)
    mask[bh:h - bh, bw:w - bw] = False               # 中央挖空,剩外框
    total = cam.sum()
    return float(cam[mask].sum() / total) if total > 0 else 0.0


def _eval_transform():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])


def _save_overlay(raw_img, cam, out_path, title):
    """把熱圖疊在原圖上存檔。"""
    base = np.asarray(raw_img.convert("RGB").resize(
        (config.IMG_SIZE, config.IMG_SIZE))) / 255.0
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(base)
    ax.imshow(cam, cmap="jet", alpha=0.45)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _sample_images(per_class=4):
    """從 DATA_DIR 每類抽樣(決定性)。F0 特別再標出 'a' 前綴與否,
    因為 'a'≈F0 是 source-confound 的主要嫌疑。"""
    root = config.DATA_DIR
    picks = []
    for ci, cname in enumerate(sorted(os.listdir(root))):
        cdir = os.path.join(root, cname)
        if not os.path.isdir(cdir):
            continue
        files = sorted(os.listdir(cdir))
        step = max(1, len(files) // per_class)
        for f in files[::step][:per_class]:
            picks.append((os.path.join(cdir, f), cname, f))
    return picks


def main():
    device = get_device()
    ckpt_path = os.environ.get("CKPT", os.path.join(config.CKPT_DIR, "best.pt"))
    out_dir = os.environ.get("EXPLAIN_DIR", "outputs_gradcam")
    os.makedirs(out_dir, exist_ok=True)

    model, ckpt = _load_model(ckpt_path, device)
    print(f"device: {device} | ckpt: {ckpt_path}")
    print(f"backbone={ckpt['backbone']} task={ckpt['task']} "
          f"classes={ckpt['class_names']} (train metric={ckpt.get('metric', float('nan')):.4f})")

    cam_engine = GradCAM(model, _target_layer(model, ckpt["backbone"]))
    tf = _eval_transform()
    class_names = ckpt["class_names"]

    # 指定影像 or 每類抽樣
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    if argv:
        items = [(p, "?", os.path.basename(p)) for p in argv]
    else:
        items = _sample_images(per_class=int(os.environ.get("PER_CLASS", 4)))

    per_class_border = {}   # true_class -> [border ratios]
    print(f"\n{'true':<6}{'pred':<6}{'conf':>7}{'border%':>9}  file")
    for path, true_cls, fname in items:
        raw = Image.open(path)
        # ImageFolder 預設 loader 會轉 RGB,這裡要對齊(有些超音波是灰階單通道)
        x = tf(raw.convert("RGB")).unsqueeze(0).to(device)
        cam, pred, prob = cam_engine(x, class_idx=None)
        br = _border_ratio(cam)
        pred_name = class_names[pred]

        per_class_border.setdefault(true_cls, []).append(br)
        tag = f"{true_cls}_{fname}".replace("/", "_")
        _save_overlay(raw, cam, os.path.join(out_dir, f"{tag}_cam.png"),
                      title=f"true={true_cls} pred={pred_name} ({prob[pred]:.2f}) border={br:.0%}")
        print(f"{true_cls:<6}{pred_name:<6}{prob[pred]:>7.2f}{br*100:>8.1f}%  {fname}")

    cam_engine.close()

    # --- 稽核重點:各類邊緣佔比 ---
    print(f"\n{'=' * 50}\n  各類 Grad-CAM 邊緣佔比(高 = 盯著邊緣/假影)\n{'=' * 50}")
    print(f"  {'class':<8}{'mean border%':>14}{'n':>5}")
    for cls in sorted(per_class_border):
        vals = per_class_border[cls]
        print(f"  {cls:<8}{np.mean(vals) * 100:>13.1f}%{len(vals):>5}")
    if "F0" in per_class_border:
        others = [v for c, vs in per_class_border.items() if c != "F0" for v in vs]
        f0 = np.mean(per_class_border["F0"])
        rest = np.mean(others) if others else float("nan")
        print(f"\n  F0 邊緣佔比 {f0:.0%} vs 其他類 {rest:.0%}")
        print("  → F0 明顯較高則支持 source-confound;相近則 F0 高分較可能是真的好認。")
    print(f"\n疊圖已存到 {out_dir}/  (人工檢視:熱區落在肝實質還是文字/邊緣?)")


if __name__ == "__main__":
    main()
