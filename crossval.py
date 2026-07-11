"""k-fold cross-validation:把單一 split 的結論換成帶誤差棒的結論。

    python crossval.py
    DEDUP=perceptual TASK=binary_geF2 CV_RESULTS_DIR=outputs_cv_binary python crossval.py

單一 split 的 test 只有 231 張、每類約 46 張,recall 的 95% CI 約 ±0.13 —— 小幅改動
分不出是真的還是雜訊(見 docs/results.md)。k-fold 讓每張影像剛好當一次 test。

**可續跑**:每折跑完就存檔,再次執行時已完成的折直接載回、不重訓
(Colab 斷線重跑只補跑未完成的)。設定只要有一項不同,該折就會重訓 —— 見 _RESUME_KEYS。

輸出到 config.CV_RESULTS_DIR:
  fold_k/               每折的 metrics.csv / curves.png / confusion_matrix.png / test_report.json
  cv_summary.json       mean ± std、每折指標、彙總的 out-of-fold confusion matrix
  oof_predictions.npz   全體影像的 out-of-fold 機率 + 標籤(calibrate.py 要用)

每張影像剛好被「沒看過它的那一折」預測一次 → oof 機率是全體資料上最誠實的一份。
校準與 ROC/閾值分析都必須用它,不能用單一 split 的 test(樣本太少,且與選模型的資料重疊)。

⚠️ std 是摺間變異,反映 split 的不確定性;消不掉 patient-level leakage。
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


# 續跑時必須一致的設定:任一項不同就代表這折的舊結果不能用,要重訓。
# (否則會把不同 dedup / task / backbone 的結果混進同一份彙總,得到無意義的數字)
_RESUME_KEYS = ("task", "backbone", "dedup", "mask", "tta",
                "dropout", "weight_decay", "mixup_alpha", "label_smoothing")


def _save_fold(fold_dir, probs, labels):
    """把這一折的 test 預測存起來,供續跑時載回(report 本來就存成 test_report.json)。"""
    np.savez(os.path.join(fold_dir, "fold_predictions.npz"),
             probs=probs, labels=labels)


def _load_fold(fold_dir, fold):
    """若這一折已完整跑過且設定與現在相同,回傳 (report, probs, labels);否則 None。

    只有「report + predictions 都在」且「關鍵設定全部吻合」才算數。
    設定不合就當作沒跑過 → 重訓,避免拿舊設定的結果去湊新的彙總。
    """
    report_path = os.path.join(fold_dir, "test_report.json")
    npz_path = os.path.join(fold_dir, "fold_predictions.npz")
    if not (os.path.exists(report_path) and os.path.exists(npz_path)):
        return None

    with open(report_path) as f:
        report = json.load(f)

    current = {k: getattr(config, k.upper()) for k in _RESUME_KEYS}
    mismatch = {k: (report.get(k), current[k])
                for k in _RESUME_KEYS if report.get(k) != current[k]}
    if mismatch:
        print(f"[crossval] fold {fold + 1} 有舊結果,但設定不同 → 重訓。"
              f"不合的項目(舊 vs 新):{mismatch}")
        return None

    d = np.load(npz_path)
    return report, d["probs"], d["labels"]


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
    oof_probs, oof_labels, oof_fold = [], [], []
    n_done, n_trained = 0, 0
    for fold in range(config.N_FOLDS):
        fold_dir = os.path.join(out_root, f"fold_{fold + 1}")

        cached = _load_fold(fold_dir, fold)
        if cached is not None:
            # 已跑過這一折 → 直接載回。Colab 斷線重跑時只會續跑未完成的折,
            # 不會把前面的心血重訓一次。
            report, probs, labels = cached
            n_done += 1
            print(f"\n{'=' * 68}\n  fold {fold + 1}/{config.N_FOLDS}  "
                  f"✅ 已完成,跳過(載回 {fold_dir})\n{'=' * 68}")
        else:
            # 每折重設種子:確保「折與折的差異」只來自 split 本身,
            # 而不是承接上一折留下的 RNG 狀態。
            torch.manual_seed(config.SEED)
            np.random.seed(config.SEED)

            print(f"\n{'=' * 68}\n  fold {fold + 1}/{config.N_FOLDS}\n{'=' * 68}")
            data = get_dataloaders(config, fold=fold)

            # 每折存自己的 checkpoint,否則後面的折會覆蓋前面的
            ckpt_path = os.path.join(config.CKPT_DIR, f"cv_fold_{fold + 1}.pt")
            report, probs, labels = run_one(data, device, fold_dir, ckpt_path)
            _save_fold(fold_dir, probs, labels)   # 供續跑時載回
            n_trained += 1

        reports.append(report)
        # 這一折的 test 是「模型沒看過的」→ 累積起來就是全體的 out-of-fold 預測
        oof_probs.append(probs)
        oof_labels.append(labels)
        oof_fold.append(np.full(len(labels), fold + 1))

    if n_done:
        print(f"\n(本次訓練 {n_trained} 折,跳過 {n_done} 折已完成的)")

    summary = _aggregate(reports)
    _print_summary(summary)

    # --- 存 out-of-fold 預測(校準 / ROC / 閾值分析用,見 calibrate.py)---
    oof_path = os.path.join(out_root, "oof_predictions.npz")
    np.savez(oof_path,
             probs=np.concatenate(oof_probs),
             labels=np.concatenate(oof_labels),
             fold=np.concatenate(oof_fold),
             class_names=np.array(summary["class_names"]),
             task=config.TASK)
    print(f"out-of-fold 預測已存到 {oof_path}"
          f"  ({len(np.concatenate(oof_labels))} 張,每張剛好被沒看過它的折預測一次)")

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
