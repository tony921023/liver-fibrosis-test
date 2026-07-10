"""k-fold cross-validation:把單一 split 的結論換成帶誤差棒的結論。

跑法:  python crossval.py
       TASK=binary_geF2 CV_RESULTS_DIR=cv_binary python crossval.py

為什麼需要這個:
單一 split 的 test 只有 231 張、每類約 46 張,recall 的 95% CI 約 ±0.13。
正則化帶來的 +0.018 balanced accuracy、resnet18 vs resnet50 的 +0.017,
都可能只是那一份 split 的運氣。k-fold 讓每張影像剛好當一次 test,
回報 mean ± std(摺間標準差),才能判斷差異是不是真的。

輸出(config.CV_RESULTS_DIR):
  fold_1/ ... fold_N/   每折的 metrics.csv / curves.png / confusion_matrix.png / test_report.json
  cv_summary.json       彙總:每折的指標、mean ± std、彙總後的 confusion matrix

⚠️ 這裡的 std 是「摺間變異」,反映 split 的不確定性。
   它仍消不掉 patient-level leakage(見 dataset.py 的 LEAKAGE 警告)。
"""

import json
import os
import statistics

import numpy as np
import torch

from dataset import get_dataloaders
from train import get_device, run_one
import config


# 彙總時要看的純量指標。二元任務多出 sensitivity / specificity,
# 用 in report 判斷,不硬編死。
_SCALAR_KEYS = [
    "macro_auroc",
    "balanced_accuracy",
    "quadratic_weighted_kappa",
    "sensitivity",
    "specificity",
]


def _mean_std(values):
    """回傳 (mean, std)。只有一折時 std 給 0(statistics.stdev 會拋例外)。"""
    if len(values) < 2:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def _aggregate(reports):
    """把每折的 report 彙總成 mean ± std,並把 confusion matrix 相加。

    confusion matrix 直接相加是合法的:k-fold 讓每張影像剛好被預測一次,
    所以加總後就是「全體資料的 out-of-fold confusion matrix」,總數 = 全部影像數。
    """
    class_names = reports[0]["class_names"]

    scalars = {}
    for key in _SCALAR_KEYS:
        if key not in reports[0]:
            continue  # 例如 multiclass 沒有 sensitivity
        vals = [r[key] for r in reports]
        mean, std = _mean_std(vals)
        scalars[key] = {"mean": mean, "std": std, "per_fold": vals}

    per_class = {}
    for i, name in enumerate(class_names):
        for metric in ("precision", "recall", "f1"):
            vals = [r["per_class"][name][metric] for r in reports]
            mean, std = _mean_std(vals)
            per_class.setdefault(name, {})[metric] = {"mean": mean, "std": std}
        per_class[name]["support_total"] = sum(
            r["per_class"][name]["support"] for r in reports)

    total_cm = np.sum([np.asarray(r["confusion_matrix"]) for r in reports], axis=0)

    return {
        "n_folds": len(reports),
        "class_names": class_names,
        "scalars": scalars,
        "per_class": per_class,
        "confusion_matrix_total": total_cm.tolist(),
        "best_epoch_per_fold": [r["best_epoch"] for r in reports],
    }


def _print_summary(summary):
    names = summary["class_names"]
    n = summary["n_folds"]

    print(f"\n{'=' * 68}")
    print(f"  {n}-fold cross-validation 彙總")
    print(f"{'=' * 68}")

    print(f"\n{'指標':<26}{'mean':>9}{'std':>9}   每折")
    for key, s in summary["scalars"].items():
        folds = "  ".join(f"{v:.3f}" for v in s["per_fold"])
        print(f"{key:<26}{s['mean']:>9.4f}{s['std']:>9.4f}   {folds}")

    print(f"\nper-class recall")
    print(f"  {'class':<8}{'mean':>9}{'std':>9}{'support':>10}")
    for name in names:
        m = summary["per_class"][name]["recall"]
        sup = summary["per_class"][name]["support_total"]
        print(f"  {name:<8}{m['mean']:>9.3f}{m['std']:>9.3f}{sup:>10d}")

    cm = np.asarray(summary["confusion_matrix_total"])
    print(f"\nout-of-fold confusion matrix(全體 {cm.sum()} 張,row=true, col=pred)")
    print("  " + "".join(f"{x:>8}" for x in names) + "   <- pred")
    for name, row in zip(names, cm):
        print(f"  {name:<6}" + "".join(f"{v:>8d}" for v in row))

    print(f"\nbest_epoch 每折: {summary['best_epoch_per_fold']}")
    print("\n⚠️ std 是摺間變異,反映 split 的不確定性;")
    print("   它消不掉 patient-level leakage,數字整體仍偏樂觀。")


def main():
    device = get_device()
    print("device:", device, "| torch:", torch.__version__)
    print(f"task={config.TASK}  backbone={config.BACKBONE}  n_folds={config.N_FOLDS}")

    out_root = config.CV_RESULTS_DIR
    os.makedirs(out_root, exist_ok=True)

    reports = []
    for fold in range(config.N_FOLDS):
        # 每折重設種子:確保「折與折的差異」只來自 split 本身,
        # 而不是承接上一折留下的 RNG 狀態。
        torch.manual_seed(config.SEED)
        np.random.seed(config.SEED)

        print(f"\n{'=' * 68}\n  fold {fold + 1}/{config.N_FOLDS}\n{'=' * 68}")
        data = get_dataloaders(config, fold=fold)

        fold_dir = os.path.join(out_root, f"fold_{fold + 1}")
        # 每折存自己的 checkpoint,否則後面的折會覆蓋前面的
        ckpt_path = os.path.join(config.CKPT_DIR, f"cv_fold_{fold + 1}.pt")
        reports.append(run_one(data, device, fold_dir, ckpt_path))

    summary = _aggregate(reports)
    _print_summary(summary)

    summary_path = os.path.join(out_root, "cv_summary.json")
    with open(summary_path, "w") as f:
        json.dump({**summary,
                   "task": config.TASK, "backbone": config.BACKBONE,
                   "dedup": config.DEDUP, "tta": config.TTA,
                   "dropout": config.DROPOUT, "weight_decay": config.WEIGHT_DECAY,
                   "mixup_alpha": config.MIXUP_ALPHA},
                  f, indent=2, ensure_ascii=False)
    print(f"\n彙總已存到 {summary_path}")


if __name__ == "__main__":
    main()
