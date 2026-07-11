"""校準與 operating point 分析:回答「這個機率能不能信」「閾值該設多少」。

跑法:
  python calibrate.py outputs_cv_perceptual_multiclass/oof_predictions.npz
  python calibrate.py outputs_cv_perceptual_binary_geF2/oof_predictions.npz

吃的是 crossval.py 存的 out-of-fold 機率:每張影像都由「沒看過它的那一折」預測,
是全體資料上最誠實的一份機率。**不要**用單一 split 的 test 做校準
(樣本太少,且與選模型的資料重疊)。

--- 為什麼要看校準 ---
臨床模型光準確率不夠:模型說「80% 是 F4」時,實際上就該有 80% 真的是 F4。
準確率高但校準爛的模型,醫師沒辦法拿它的信心值做決策。
本專案有兩個理由懷疑校準不好:
  1. mixup + label smoothing 通常讓模型「低估」信心
  2. 但實測 Grad-CAM 時看到 confidence 幾乎都是 1.00 → 反而像「過度自信」
  兩股力量方向相反,不量就不知道。

ECE(Expected Calibration Error):把預測依信心分箱,算每箱「平均信心」與
「實際準確率」的差,再依箱內樣本數加權平均。0 = 完美校準。

--- 為什麼要看 operating point ---
二元 >=F2 現在死用閾值 0.5。臨床真正的問題是:
「我要 sensitivity 達到 0.90,閾值該設多少?代價是 specificity 掉到多少?」
→ 不用重訓,調閾值即可。輸出閾值表讓你挑取捨點。

輸出(與 npz 同目錄):
  reliability.png    校準曲線(對角線 = 完美校準)
  roc.png            ROC 曲線(僅二元)
  threshold_table    終端機列印(僅二元)
"""

import os
import sys

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _confidence_and_correct(probs, labels):
    """多分類:取 argmax 的機率當「信心」,是否猜中當「正確」。

    二元也走同一套(argmax 等同閾值 0.5),這樣 reliability diagram 兩種任務通用。
    """
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (pred == labels).astype(float)
    return conf, correct


def expected_calibration_error(conf, correct, n_bins=10):
    """ECE:依信心分箱,加權平均 |平均信心 - 實際準確率|。

    回傳 (ece, bins)。bins 供畫圖用,只保留非空的箱。
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # 最後一箱含右端點,否則 conf=1.0 會被漏掉(本專案很常見)
        m = (conf > lo) & (conf <= hi) if hi < 1.0 else (conf > lo) & (conf <= 1.0)
        if not m.any():
            continue
        acc = correct[m].mean()
        avg_conf = conf[m].mean()
        w = m.sum() / len(conf)
        ece += w * abs(avg_conf - acc)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                     "acc": float(acc), "conf": float(avg_conf)})
    return float(ece), bins


def save_reliability(bins, ece, conf, correct, path, title):
    """reliability diagram:x=平均信心, y=實際準確率。對角線 = 完美校準。

    點在對角線「下方」= 過度自信(說 90% 其實只有 70% 對)。

    圖上標籤一律用英文:matplotlib 預設字型(DejaVu Sans)沒有 CJK 字元,
    中文會變成豆腐方塊。中文說明留在終端機輸出。
    """
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.5, 6),
                                  gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    if bins:
        ax.plot([b["conf"] for b in bins], [b["acc"] for b in bins],
                "o-", color="tab:blue", label="model")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{title}\nECE = {ece:.4f}   "
                 f"mean conf {conf.mean():.3f} vs accuracy {correct.mean():.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # 下方:每箱的樣本數,看信心分布集中在哪
    if bins:
        ax2.bar([b["conf"] for b in bins], [b["n"] for b in bins],
                width=0.08, color="tab:gray")
    ax2.set_xlabel("predicted confidence")
    ax2.set_ylabel("count")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_roc(labels, pos_prob, path, title):
    fpr, tpr, _ = roc_curve(labels, pos_prob)
    auc = roc_auc_score(labels, pos_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="tab:blue", label=f"AUROC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("1 - specificity (FPR)")
    ax.set_ylabel("sensitivity (TPR)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return float(auc)


def threshold_table(labels, pos_prob, targets=(0.80, 0.85, 0.90, 0.95, 0.99)):
    """對每個目標 sensitivity,找出「達標的最高閾值」(spec 最好的那個)。

    也回報預設 0.5 當對照。臨床上漏掉顯著纖維化的代價高 → 通常要拉高 sensitivity。
    """
    fpr, tpr, thr = roc_curve(labels, pos_prob)
    rows = []
    for want in targets:
        ok = np.where(tpr >= want)[0]
        if len(ok) == 0:
            continue
        i = ok[0]   # roc_curve 的 tpr 遞增 → 第一個達標的點閾值最高、spec 最好
        rows.append({"target_sens": want, "threshold": float(thr[i]),
                     "sensitivity": float(tpr[i]), "specificity": float(1 - fpr[i])})

    pred = (pos_prob >= 0.5).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum()); fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum()); fp = int(((pred == 1) & (labels == 0)).sum())
    default = {"target_sens": None, "threshold": 0.5,
               "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
               "specificity": tn / (tn + fp) if tn + fp else 0.0}
    return rows, default


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("錯誤:請指定 oof_predictions.npz 的路徑")
        sys.exit(1)

    npz_path = sys.argv[1]
    d = np.load(npz_path, allow_pickle=True)
    probs, labels = d["probs"], d["labels"]
    class_names = [str(c) for c in d["class_names"]]
    task = str(d["task"])
    out_dir = os.path.dirname(npz_path) or "."

    print(f"來源: {npz_path}")
    print(f"task={task}  classes={class_names}  n={len(labels)} 張(out-of-fold)")

    # ---- 校準 ----
    conf, correct = _confidence_and_correct(probs, labels)
    ece, bins = expected_calibration_error(conf, correct)
    gap = conf.mean() - correct.mean()

    print(f"\n{'=' * 58}\n  校準(calibration)\n{'=' * 58}")
    print(f"  ECE(期望校準誤差)   {ece:.4f}    (0 = 完美)")
    print(f"  平均信心              {conf.mean():.4f}")
    print(f"  實際準確率            {correct.mean():.4f}")
    print(f"  差距                  {gap:+.4f}  → "
          + ("過度自信(信心 > 實力)" if gap > 0.02 else
             "低估信心(信心 < 實力)" if gap < -0.02 else "大致校準"))
    print(f"\n  {'信心區間':<14}{'n':>6}{'平均信心':>10}{'實際準確率':>12}")
    for b in bins:
        print(f"  [{b['lo']:.1f}, {b['hi']:.1f}]{'':<3}{b['n']:>6}"
              f"{b['conf']:>10.3f}{b['acc']:>12.3f}")

    rel_path = os.path.join(out_dir, "reliability.png")
    save_reliability(bins, ece, conf, correct, rel_path, f"Reliability diagram ({task})")

    # ---- operating point(僅二元)----
    roc_path = None
    if len(class_names) == 2:
        pos_prob = probs[:, 1]          # class 1 = 陽性(>=F2)
        roc_path = os.path.join(out_dir, "roc.png")
        auc = save_roc(labels, pos_prob, roc_path, f"ROC ({task}, out-of-fold)")

        rows, default = threshold_table(labels, pos_prob)
        print(f"\n{'=' * 58}\n  operating point(閾值取捨,不用重訓)\n{'=' * 58}")
        print(f"  AUROC = {auc:.4f}\n")
        print(f"  {'目標 sens':<12}{'閾值':>8}{'sensitivity':>14}{'specificity':>14}")
        print(f"  {'(預設)':<12}{default['threshold']:>8.2f}"
              f"{default['sensitivity']:>14.3f}{default['specificity']:>14.3f}   ← 目前")
        for r in rows:
            print(f"  {r['target_sens']:<12.2f}{r['threshold']:>8.2f}"
                  f"{r['sensitivity']:>14.3f}{r['specificity']:>14.3f}")
        print("\n  臨床上漏掉顯著纖維化(>=F2)的代價高 → 通常往高 sensitivity 那端挑。")

    print(f"\n已存:{rel_path}" + (f"、{roc_path}" if roc_path else ""))
    print("⚠️ 這些機率仍受 patient-level leakage 影響(無病人 ID),校準亦然。")


if __name__ == "__main__":
    main()
