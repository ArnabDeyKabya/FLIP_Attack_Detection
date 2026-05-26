"""
Detection-metric plots for the Deep k-NN defense thesis sweep.

Generates the full panel of detection diagnostics the thesis needs:

    Per-k figures (one PNG/PDF per k):
        score_dist_k{k}.{png,pdf}                — overlaid score histograms
                                                     for poisoned vs clean
        score_dist_k{k}_log.{png,pdf}            — same on log y
        confusion_matrix_k{k}.{png,pdf}          — at the configured threshold
        threshold_sweep_k{k}.{png,pdf}           — F1 / MCC / precision / recall
                                                     / FPR over threshold grid
        calibration_k{k}.{png,pdf}               — empirical poison rate vs
                                                     mean score per bin

    Aggregate (combined-k) figures:
        roc_all_k.{png,pdf}                      — ROC, one curve per k
        pr_all_k.{png,pdf}                       — PR,  one curve per k
        k_sweep_detection.{png,pdf}              — 6-panel AUROC / AUPRC /
                                                     best-F1 / best-MCC /
                                                     P@k / R@k vs k
        k_sweep_flagged.{png,pdf}                — flagged-precision and
                                                     flagged-recall vs k
        k_sweep_removed.{png,pdf}                — # poisons removed vs
                                                     # clean removed vs k
        score_dist_facet.{png,pdf}               — small-multiples score
                                                     histograms across all k
        per_class_flag_heatmap.{png,pdf}         — per-class flag rate × k

All figures are saved as both PNG (for the digital thesis / PowerPoint) and PDF
(for the print thesis). Style is matplotlib-only, no seaborn, no rcparams from
outside this script — so the output is reproducible.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix


# ----- style ------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "legend.frameon": False,
})


# CIFAR-10 class names (used for per-class plots).
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _save(fig, out_stem: Path):
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"))
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def _load_ground_truth(project_root: Path, sample_row: dict) -> np.ndarray:
    """Reload the ground-truth poison mask used in a given experiment."""
    exp_dir = project_root / "experiments" / sample_row["experiment"]
    labels = np.load(exp_dir / "labels.npy")
    if labels.ndim == 2:
        labels_hard = labels.argmax(axis=1)
    else:
        labels_hard = labels.astype(int)
    true_path = project_root / "precomputed_labels" / sample_row["dataset"] \
        / sample_row["user_model"] / sample_row["poisoner"] / "true.npy"
    true = np.load(true_path)
    if true.ndim == 2:
        true_hard = true.argmax(axis=1)
    else:
        true_hard = true.astype(int)
    return (labels_hard != true_hard).astype(np.int32), labels_hard, true_hard


# ----- per-k figures ----------------------------------------------------------

def plot_score_dist(scores, gt, k, fig_dir: Path, threshold: float):
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.hist(scores[gt == 0], bins=bins, alpha=0.55,
            label=f"clean (n={int((gt == 0).sum())})", color="#1f77b4")
    ax.hist(scores[gt == 1], bins=bins, alpha=0.55,
            label=f"poisoned (n={int((gt == 1).sum())})", color="#d62728")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1,
               label=f"threshold = {threshold:g}")
    ax.set_xlabel("k-NN disagreement score")
    ax.set_ylabel("count")
    ax.set_title(f"Score distribution (k = {k})")
    ax.legend(loc="upper center")
    _save(fig, fig_dir / f"score_dist_k{k}")

    # Log-scale variant — keeps the rare poisoned tail visible
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.hist(scores[gt == 0], bins=bins, alpha=0.55, label="clean", color="#1f77b4")
    ax.hist(scores[gt == 1], bins=bins, alpha=0.55, label="poisoned", color="#d62728")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("k-NN disagreement score")
    ax.set_ylabel("count (log)")
    ax.set_yscale("log")
    ax.set_title(f"Score distribution, log-scale (k = {k})")
    ax.legend(loc="upper center")
    _save(fig, fig_dir / f"score_dist_k{k}_log")


def plot_confusion(scores, gt, k, fig_dir: Path, threshold: float):
    pred = (scores > threshold).astype(int)
    cm_ = confusion_matrix(gt, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(cm_, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_[i, j]:d}",
                    ha="center", va="center",
                    color="white" if cm_[i, j] > cm_.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred clean", "pred poison"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true clean", "true poison"])
    ax.set_title(f"Detection confusion matrix (k = {k}, thr = {threshold:g})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, fig_dir / f"confusion_matrix_k{k}")


def plot_threshold_sweep(sweep_df_k: pd.DataFrame, k, fig_dir: Path,
                          threshold: float):
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for col, label, ls in [
        ("precision", "precision", "-"),
        ("recall",    "recall",    "-"),
        ("f1",        "F1",        "-"),
        ("mcc",       "MCC",       "--"),
        ("fpr",       "FPR",       ":"),
    ]:
        ax.plot(sweep_df_k["threshold"], sweep_df_k[col],
                label=label, linestyle=ls, linewidth=1.6)
    ax.axvline(threshold, color="black", linestyle="--", alpha=0.5,
               label=f"current thr = {threshold:g}")
    ax.set_xlabel("threshold on k-NN disagreement score")
    ax.set_ylabel("metric")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Threshold sweep at k = {k}")
    ax.legend(ncol=2, loc="lower center")
    _save(fig, fig_dir / f"threshold_sweep_k{k}")


def plot_calibration(scores, gt, k, fig_dir: Path, n_bins: int = 20):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(scores, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    mean_score = np.zeros(n_bins)
    poison_rate = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() > 0:
            counts[b] = int(m.sum())
            mean_score[b] = float(scores[m].mean())
            poison_rate[b] = float(gt[m].mean())
        else:
            mean_score[b] = (bin_edges[b] + bin_edges[b + 1]) / 2
            poison_rate[b] = np.nan
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    sizes = 30 + (counts / max(counts.max(), 1)) * 220
    ax.scatter(mean_score, poison_rate, s=sizes, alpha=0.7, color="#d62728",
               edgecolors="black", linewidths=0.5)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1,
            label="y = x (perfect calibration)")
    ax.set_xlabel("mean k-NN disagreement score in bin")
    ax.set_ylabel("empirical poison rate in bin")
    ax.set_title(f"Calibration plot (k = {k}); bubble area  bin count")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend()
    _save(fig, fig_dir / f"calibration_k{k}")


# ----- aggregate (multi-k) figures -------------------------------------------

def plot_roc_all(scores_by_k, gt_by_k, fig_dir: Path):
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    colors = cm.viridis(np.linspace(0, 0.9, len(scores_by_k)))
    for (k, scores), color in zip(sorted(scores_by_k.items()), colors):
        gt = gt_by_k[k]
        if int(gt.sum()) == 0:
            continue
        fpr, tpr, _ = roc_curve(gt, scores)
        from sklearn.metrics import auc as _auc
        ax.plot(fpr, tpr, label=f"k={k}  (AUC={_auc(fpr, tpr):.3f})",
                color=color, linewidth=1.5)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC for k-NN poison detection")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    _save(fig, fig_dir / "roc_all_k")


def plot_pr_all(scores_by_k, gt_by_k, fig_dir: Path):
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    colors = cm.viridis(np.linspace(0, 0.9, len(scores_by_k)))
    for (k, scores), color in zip(sorted(scores_by_k.items()), colors):
        gt = gt_by_k[k]
        if int(gt.sum()) == 0:
            continue
        prec, rec, _ = precision_recall_curve(gt, scores)
        from sklearn.metrics import average_precision_score as _ap
        ax.plot(rec, prec, label=f"k={k}  (AP={_ap(gt, scores):.3f})",
                color=color, linewidth=1.5)
    base_rate = next(iter(gt_by_k.values())).mean()
    ax.axhline(base_rate, color="gray", linestyle="--", linewidth=1,
               label=f"base rate = {base_rate:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall for k-NN poison detection")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    _save(fig, fig_dir / "pr_all_k")


def plot_k_sweep_detection(master: pd.DataFrame, fig_dir: Path):
    """6-panel — AUROC, AUPRC, best F1, best MCC, P@k, R@k — vs k.

    Both `none` and `remove` modes are plotted; their detection metrics are
    identical (detection runs before any removal) but plotting both is the
    sanity check the thesis wants.

    Panels for which the aggregator could not produce the metric (e.g. when
    scores.npy is missing) are silently skipped instead of crashing.
    """
    panels = [
        ("auroc",  "AUROC",       (0.0, 1.02)),
        ("auprc",  "AUPRC",       (0.0, 1.02)),
        ("best_f1", "best F1",    (0.0, 1.02)),
        ("best_mcc", "best MCC",  (0.0, 1.02)),
        ("precision_at_k", "Precision@k (k = num poisoned)", (0.0, 1.02)),
        ("recall_at_k",    "Recall@k (k = num poisoned)",    (0.0, 1.02)),
    ]
    available = [p for p in panels if p[0] in master.columns]
    if not available:
        print("[plot_detection] no detection-metric columns found; "
              "skipping k_sweep_detection.")
        return
    n = len(available)
    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(10.5, 3.0 * rows))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    for (col, title, ylim), ax in zip(available, axes_flat):
        for mode, sub in master.groupby("mode"):
            sub = sub.sort_values("k")
            y = sub[col].astype(float).values
            ax.plot(sub["k"], y, marker="o", linewidth=1.6, label=mode)
        ax.set_xscale("log")
        ax.set_xlabel("k")
        ax.set_ylabel(title)
        ax.set_ylim(*ylim)
        ax.set_title(title + " vs k")
        ax.legend()
    for ax in list(axes_flat)[n:]:
        ax.set_visible(False)
    fig.suptitle("Deep k-NN detection metrics across the k sweep", y=1.00)
    fig.tight_layout()
    _save(fig, fig_dir / "k_sweep_detection")


def plot_k_sweep_flagged(master: pd.DataFrame, fig_dir: Path):
    """Two-panel: flagged-precision and flagged-recall vs k for remove mode."""
    sub = master[master["mode"] == "remove"].sort_values("k")
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].plot(sub["k"], sub["flagged_precision"], marker="o", color="#d62728",
                 linewidth=1.6)
    axes[0].set_xscale("log"); axes[0].set_xlabel("k")
    axes[0].set_ylabel("flagged precision  =  TP / # flagged")
    axes[0].set_title("Flagged precision vs k (auto threshold)")
    axes[0].set_ylim(0, 1.02)
    axes[1].plot(sub["k"], sub["flagged_recall"], marker="o", color="#1f77b4",
                 linewidth=1.6)
    axes[1].set_xscale("log"); axes[1].set_xlabel("k")
    axes[1].set_ylabel("flagged recall  =  TP / # poisoned")
    axes[1].set_title("Flagged recall vs k (auto threshold)")
    axes[1].set_ylim(0, 1.02)
    fig.tight_layout()
    _save(fig, fig_dir / "k_sweep_flagged")


def plot_k_sweep_removed(master: pd.DataFrame, fig_dir: Path):
    """Number-of-poisons removed vs number-of-clean removed, both vs k."""
    sub = master[master["mode"] == "remove"].sort_values("k").copy()
    if sub.empty:
        return
    sub["n_clean_removed"] = sub["n_removed"] - (
        sub["flagged_recall"].fillna(0) * sub["n_true_poisoned"]
    )
    sub["n_poison_removed"] = sub["flagged_recall"].fillna(0) * sub["n_true_poisoned"]

    fig, ax1 = plt.subplots(figsize=(7.0, 4.5))
    ax1.plot(sub["k"], sub["n_poison_removed"], marker="o", color="#d62728",
             label="# poisoned removed")
    ax1.plot(sub["k"], sub["n_clean_removed"], marker="s", color="#1f77b4",
             label="# clean removed")
    ax1.set_xscale("log"); ax1.set_xlabel("k")
    ax1.set_ylabel("# samples")
    ax1.set_title("How the removal cut grows with k")
    ax1.axhline(sub["n_true_poisoned"].max(), color="gray", linestyle="--",
                linewidth=1, label=f"total poisoned = {int(sub['n_true_poisoned'].max())}")
    ax1.legend()
    fig.tight_layout()
    _save(fig, fig_dir / "k_sweep_removed")


def plot_score_dist_facet(scores_by_k, gt_by_k, fig_dir: Path, threshold: float):
    """Small-multiples score histograms across all k. One panel per k."""
    ks = sorted(scores_by_k.keys())
    n = len(ks)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.6 * rows),
                              sharex=True, sharey=False)
    if rows * cols == 1:
        axes = np.array([axes])
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    bins = np.linspace(0, 1, 41)
    for ax, k in zip(axes_flat, ks):
        scores = scores_by_k[k]
        gt = gt_by_k[k]
        ax.hist(scores[gt == 0], bins=bins, alpha=0.55, color="#1f77b4",
                label="clean")
        ax.hist(scores[gt == 1], bins=bins, alpha=0.55, color="#d62728",
                label="poisoned")
        ax.axvline(threshold, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"k = {k}", fontsize=10)
        ax.set_yscale("log")
        ax.tick_params(labelsize=8)
    for ax in list(axes_flat)[len(ks):]:
        ax.set_visible(False)
    fig.supxlabel("k-NN disagreement score")
    fig.supylabel("count (log)")
    fig.legend(["clean", "poisoned"], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    _save(fig, fig_dir / "score_dist_facet")


def plot_per_class_flag_heatmap(per_class_df: pd.DataFrame, fig_dir: Path,
                                  mode: str = "remove"):
    if per_class_df is None or per_class_df.empty:
        return
    sub = per_class_df[per_class_df["mode"] == mode].copy()
    if sub.empty:
        return
    pivot = sub.pivot_table(
        index="assigned_class", columns="k", values="flag_rate", aggfunc="mean",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(0.6 * len(pivot.columns) + 3.0, 5.0))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Reds")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    labels = [CIFAR10_CLASSES[i] if i < len(CIFAR10_CLASSES) else str(i)
              for i in pivot.index]
    ax.set_yticklabels(labels)
    ax.set_xlabel("k")
    ax.set_ylabel("assigned (training) class")
    ax.set_title(f"Per-class flag rate (mode = {mode})")
    fig.colorbar(im, ax=ax, label="flag rate")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=8)
    _save(fig, fig_dir / "per_class_flag_heatmap")


# ----- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name-prefix", required=True)
    ap.add_argument("--k-values", type=int, nargs="+", required=True)
    ap.add_argument("--modes", nargs="+", default=["none", "remove"])
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    project_root = Path(args.project_root) if args.project_root else \
        Path(__file__).resolve().parents[2]
    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = project_root / report_dir

    master = pd.read_csv(report_dir / "master_metrics.csv")
    sweep_df = pd.read_csv(report_dir / "threshold_sweep.csv") \
        if (report_dir / "threshold_sweep.csv").exists() else None
    per_class = pd.read_csv(report_dir / "per_class_detection.csv") \
        if (report_dir / "per_class_detection.csv").exists() else None

    fig_dir = report_dir / "figures" / "detection"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load scores + ground truth for the `remove` mode (detection is identical
    # for the matched `none` run; doesn't matter which we pick).
    pick_mode = "remove" if "remove" in args.modes else args.modes[0]
    scores_by_k, gt_by_k = {}, {}
    for k in args.k_values:
        exp = f"{args.name_prefix}_{pick_mode}_k{k}"
        scores_path = project_root / "experiments" / exp / "scores.npy"
        if not scores_path.exists():
            print(f"[plot_detection] WARN: missing {scores_path}; skipping k={k}")
            continue
        scores = np.load(scores_path)
        sample_row = master[master["experiment"] == exp].iloc[0].to_dict()
        gt, _, _ = _load_ground_truth(project_root, sample_row)
        scores_by_k[k] = scores
        gt_by_k[k]     = gt

    # Per-k figures
    for k in scores_by_k.keys():
        plot_score_dist(scores_by_k[k], gt_by_k[k], k, fig_dir, args.threshold)
        plot_confusion(scores_by_k[k], gt_by_k[k], k, fig_dir, args.threshold)
        plot_calibration(scores_by_k[k], gt_by_k[k], k, fig_dir)
        if sweep_df is not None:
            sub = sweep_df[(sweep_df["k"] == k) & (sweep_df["mode"] == pick_mode)]
            if not sub.empty:
                plot_threshold_sweep(sub.sort_values("threshold"), k, fig_dir,
                                     args.threshold)

    # Aggregate-k figures
    if scores_by_k:
        plot_roc_all(scores_by_k, gt_by_k, fig_dir)
        plot_pr_all(scores_by_k, gt_by_k, fig_dir)
        plot_score_dist_facet(scores_by_k, gt_by_k, fig_dir, args.threshold)
    plot_k_sweep_detection(master, fig_dir)
    plot_k_sweep_flagged(master, fig_dir)
    plot_k_sweep_removed(master, fig_dir)
    if per_class is not None:
        plot_per_class_flag_heatmap(per_class, fig_dir, mode="remove")
        plot_per_class_flag_heatmap(per_class, fig_dir, mode="none")

    print(f"[plot_detection] wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
