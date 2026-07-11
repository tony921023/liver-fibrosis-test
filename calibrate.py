"""校準與 operating point:回答「這個機率能不能信」「閾值該設多少」。不需重訓。

    python calibrate.py outputs_cv_perceptual_multiclass/oof_predictions.npz

吃 crossval.py 存的 out-of-fold 機率(每張影像由沒看過它的那一折預測)。
**不要**用單一 split 的 test 做校準 —— 樣本太少,且與選模型的資料重疊。

**校準**:臨床上光準確率不夠,模型說「80% 是 F4」時實際就該有 80% 真的是 F4,
否則醫師無法拿它的信心值做決策。ECE = 依信心分箱,加權平均 |平均信心 − 實際準確率|。

**溫度縮放**:診斷完要能修。softmax(logit / T),T>1 拉平機率修正過度自信。
argmax 不變 → 準確率完全不受影響,只讓機率變可信。

**bootstrap CI**:摺間 std 只有 5 個點、本身雜訊大。改對全體 out-of-fold 預測重抽 2000 次,
得到反映樣本量的信賴區間 —— 論文該報的誤差棒。

輸出(與 npz 同目錄):reliability.png、reliability_temp_scaled.png、roc.png(僅二元)。
終端機另印 ECE、最佳 T、95% CI、閾值表。
"""

import os
import sys

import numpy as np
from sklearn.metrics import (
    roc_curve, roc_auc_score, balanced_accuracy_score, cohen_kappa_score,
)

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


def fit_temperature(probs, labels, grid=None):
    """溫度縮放(temperature scaling):找一個純量 T,讓 softmax(logit / T) 最校準。

    T > 1 → 把機率「拉平」,修正過度自信;T < 1 → 拉尖,修正低估信心。
    只有 1 個參數,不動模型權重,幾乎不會傷準確率(argmax 不變 → accuracy 完全不變),
    是臨床上最常用的後處理校準法。

    實作細節:標準做法要用 logits,但我們只存了 softmax 後的機率。
    因為 log(p) = logit - logsumexp(logit),兩者只差一個常數,而 softmax 平移不變:
        softmax(log(p) / T) == softmax((logit - C) / T) == softmax(logit / T)
    所以直接拿 log(p) 當 logit 用是正確的。

    以 NLL(負對數概似)為目標在網格上搜 T —— 資料量小,網格搜尋比梯度下降更穩。
    """
    if grid is None:
        grid = np.linspace(0.2, 10.0, 393)    # 0.2 ~ 10.0,步長 0.025
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    n = len(labels)

    best_T, best_nll = 1.0, np.inf
    for T in grid:
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)          # 數值穩定
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        nll = -logp[np.arange(n), labels].mean()
        if nll < best_nll:
            best_T, best_nll = float(T), float(nll)

    # 卡在網格邊界 = 真正的最佳值可能在界外,結果被靜默截斷 → 要講出來
    if best_T <= grid[0] + 1e-9 or best_T >= grid[-1] - 1e-9:
        print(f"  ⚠️ 最佳 T={best_T:.3f} 落在搜尋範圍 [{grid[0]}, {grid[-1]}] 的邊界,"
              f"真正的最佳值可能在界外")
    return best_T


def apply_temperature(probs, T):
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / T
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def bootstrap_ci(labels, probs, n_boot=2000, seed=42):
    """對 out-of-fold 預測做 bootstrap 重抽樣,得到指標的 95% 信賴區間。

    為什麼比「摺間 std」好:摺間 std 只有 N_FOLDS 個點(5 個),本身雜訊很大;
    bootstrap 直接在 1388 張預測上重抽,反映的是「樣本量」帶來的不確定性,
    是論文裡該報的那種誤差棒。

    ⚠️ bootstrap 假設樣本獨立。本資料仍有 patient-level leakage(無病人 ID),
    所以真實的不確定性會比這裡算出來的更大。
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    n_classes = probs.shape[1]
    idx_all = np.arange(n_classes)

    stats = {"balanced_accuracy": [], "macro_auroc": [], "quadratic_weighted_kappa": []}
    is_binary = n_classes == 2
    if is_binary:
        stats["sensitivity"] = []
        stats["specificity"] = []

    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        yb, pb = labels[b], probs[b]
        if len(np.unique(yb)) < n_classes:
            continue                                   # 重抽時某類全缺 → 跳過
        predb = pb.argmax(1)
        stats["balanced_accuracy"].append(balanced_accuracy_score(yb, predb))
        stats["quadratic_weighted_kappa"].append(
            cohen_kappa_score(yb, predb, labels=idx_all, weights="quadratic"))
        stats["macro_auroc"].append(
            roc_auc_score(yb, pb[:, 1]) if is_binary else
            roc_auc_score(yb, pb, multi_class="ovr", average="macro", labels=list(idx_all)))
        if is_binary:
            tp = ((predb == 1) & (yb == 1)).sum(); fn = ((predb == 0) & (yb == 1)).sum()
            tn = ((predb == 0) & (yb == 0)).sum(); fp = ((predb == 1) & (yb == 0)).sum()
            stats["sensitivity"].append(tp / (tp + fn) if tp + fn else 0.0)
            stats["specificity"].append(tn / (tn + fp) if tn + fp else 0.0)

    out = {}
    for k, v in stats.items():
        v = np.asarray(v)
        out[k] = {"point": float(v.mean()),
                  "lo": float(np.percentile(v, 2.5)),
                  "hi": float(np.percentile(v, 97.5))}
    return out


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

    # ---- 溫度縮放:把「診斷出過度自信」變成「修好它」----
    T = fit_temperature(probs, labels)
    probs_t = apply_temperature(probs, T)
    conf_t, correct_t = _confidence_and_correct(probs_t, labels)
    ece_t, bins_t = expected_calibration_error(conf_t, correct_t)

    print(f"\n{'=' * 58}\n  溫度縮放(temperature scaling)\n{'=' * 58}")
    print(f"  最佳溫度 T            {T:.3f}   "
          + ("(>1 → 把機率拉平,修正過度自信)" if T > 1.02 else
             "(<1 → 把機率拉尖,修正低估信心)" if T < 0.98 else "(≈1 → 本來就大致校準)"))
    print(f"  ECE   {ece:.4f}  →  {ece_t:.4f}   ({(ece_t - ece) / max(ece, 1e-9) * 100:+.0f}%)")
    print(f"  平均信心 {conf.mean():.4f}  →  {conf_t.mean():.4f}   "
          f"(實際準確率 {correct.mean():.4f},不變)")
    # argmax 不變 → 準確率完全不受影響,這正是溫度縮放的賣點
    assert (probs.argmax(1) == probs_t.argmax(1)).all(), "溫度縮放不該改變 argmax"
    print("  ✅ 準確率完全不變(溫度縮放不改 argmax),只修機率的可信度")

    rel_t_path = os.path.join(out_dir, "reliability_temp_scaled.png")
    save_reliability(bins_t, ece_t, conf_t, correct_t, rel_t_path,
                     f"Reliability after temperature scaling (T={T:.2f})")

    # ---- bootstrap 95% CI ----
    print(f"\n{'=' * 58}\n  bootstrap 95% 信賴區間(n=2000,重抽 out-of-fold 預測)\n{'=' * 58}")
    ci = bootstrap_ci(labels, probs)
    print(f"  {'指標':<26}{'點估計':>10}{'95% CI':>22}")
    for k, v in ci.items():
        print(f"  {k:<28}{v['point']:>8.4f}   [{v['lo']:.4f}, {v['hi']:.4f}]")
    print("\n  ⚠️ bootstrap 假設樣本獨立,但本資料仍有 patient-level leakage,")
    print("     真實不確定性比這裡更大。這是下限,不是上限。")

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

    print(f"\n已存:{rel_path}、{rel_t_path}" + (f"、{roc_path}" if roc_path else ""))
    print("⚠️ 這些機率仍受 patient-level leakage 影響(無病人 ID),校準亦然。")


if __name__ == "__main__":
    main()
