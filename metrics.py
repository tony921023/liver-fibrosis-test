"""評估指標與結果輸出。與訓練流程解耦,搬到臨床資料時可直接重用。

  macro_auroc(labels, probs, num_classes) -> float
  full_report(labels, probs, class_names) -> dict
  save_history_csv / save_curves / save_confusion_png / save_report_json

分期是 ordinal(F0<F1<...<F4)→ 用 QWK:把 F4 誤判成 F3 罰得比誤判成 F0 輕,
比單看 AUROC 更貼合臨床意義。二元任務另外輸出 sensitivity / specificity。
"""

import csv
import json

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    cohen_kappa_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)

# 無頭環境(Colab / 背景跑)也能存圖
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def macro_auroc(labels, probs, num_classes):
    """二元取陽性類機率;多分類用 one-vs-rest macro 平均。

    明確帶 labels=range(num_classes),避免某 split 某類缺漏時報錯。
    """
    if num_classes == 2:
        return float(roc_auc_score(labels, probs[:, 1]))
    return float(roc_auc_score(
        labels, probs, multi_class="ovr", average="macro",
        labels=list(range(num_classes)),
    ))


def full_report(labels, probs, class_names):
    """回傳一份完整的評估報告 dict(供 print / 存 json / 畫圖)。"""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    num_classes = len(class_names)
    preds = probs.argmax(axis=1)
    idx = list(range(num_classes))

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=idx, zero_division=0)
    per_class = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in idx
    }

    cm = confusion_matrix(labels, preds, labels=idx)

    report = {
        "macro_auroc": macro_auroc(labels, probs, num_classes),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        # QWK 主要對 ordinal 多分類有意義;二元時等同一般 kappa,仍可參考
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(labels, preds, labels=idx, weights="quadratic")),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
    }

    if num_classes == 2:
        # 二元任務(如 >=F2)臨床上看的是 sensitivity / specificity,不是 accuracy。
        # 慣例:class 1 = 陽性。cm = [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = cm.ravel()
        report["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else 0.0
        report["specificity"] = float(tn / (tn + fp)) if (tn + fp) else 0.0

    return report


def format_report(report):
    """把 full_report 的 dict 排成可讀的多行字串,印在終端機。"""
    lines = []
    lines.append(f"macro AUROC        : {report['macro_auroc']:.4f}")
    lines.append(f"balanced accuracy  : {report['balanced_accuracy']:.4f}")
    lines.append(f"quadratic w. kappa : {report['quadratic_weighted_kappa']:.4f}")
    if "sensitivity" in report:
        lines.append(f"sensitivity (recall陽性): {report['sensitivity']:.4f}")
        lines.append(f"specificity            : {report['specificity']:.4f}")
    lines.append("per-class:")
    lines.append(f"  {'class':<8}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}")
    for name, m in report["per_class"].items():
        lines.append(f"  {name:<8}{m['precision']:>10.3f}{m['recall']:>9.3f}"
                     f"{m['f1']:>8.3f}{m['support']:>9d}")
    names = report["class_names"]
    lines.append("confusion matrix (row=true, col=pred):")
    lines.append("  " + "".join(f"{n:>8}" for n in names) + "   <- pred")
    for name, row in zip(names, report["confusion_matrix"]):
        lines.append(f"  {name:<6}" + "".join(f"{v:>8d}" for v in row))
    return "\n".join(lines)


def save_history_csv(history, path):
    """history: list of per-epoch dict。存成 metrics.csv 方便對照不同 run。"""
    if not history:
        return
    fields = list(history[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def save_curves(history, path):
    """畫 train/val loss(左軸)與 val macro AUROC(右軸)隨 epoch 變化。"""
    epochs = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(epochs, [h["train_loss"] for h in history], "o-", label="train_loss")
    ax1.plot(epochs, [h["val_loss"] for h in history], "s-", label="val_loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(epochs, [h["val_macro_auroc"] for h in history], "^-",
             color="tab:green", label="val_macro_auroc")
    ax2.set_ylabel("val macro AUROC")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_confusion_png(report, path):
    """把 test 集的 confusion matrix 畫成熱圖。"""
    cm = np.asarray(report["confusion_matrix"])
    names = report["class_names"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (test)")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_report_json(report, path, extra=None):
    """把 test 報告(+ 選用的 metadata)存成 json。"""
    payload = dict(report)
    if extra:
        payload.update(extra)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
