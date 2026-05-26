"""
Sample-image grids for the Deep k-NN defense thesis sweep.

For a chosen k value, dump four 8x8 grids of CIFAR images:

    samples_TP_k{k}.{png,pdf}     poisoned    AND flagged  (correct catch)
    samples_FN_k{k}.{png,pdf}     poisoned    AND missed   (missed poison)
    samples_FP_k{k}.{png,pdf}     clean       AND flagged  (false alarm)
    samples_TN_k{k}.{png,pdf}     clean       AND kept     (correct keep)

Plus:

    samples_top_scores_k{k}.{png,pdf}    top 32 most-suspicious samples,
                                          regardless of GT, with score and
                                          poisoned/clean label.
    samples_target_class_overview.{png,pdf} grid of target-class clean
                                                  images for visual context.
    samples_poison_examples.{png,pdf}    grid of (clean source, poisoned)
                                          pairs to show what FLIP does
                                          (it doesn't perturb images, only
                                          flips labels — useful for the
                                          thesis intro).

Each grid title carries (assigned label, true label, score) so the reader can
see what the defense saw vs ground truth.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": False,
})


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _save(fig, out_stem: Path):
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"))
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def _load_cifar_train_images(project_root: Path, dataset: str):
    """Return (images: (N, 32, 32, 3) uint8) for CIFAR-10/100 training set."""
    # The FLIP module layout is `modules/base_utils/datasets.py`, importable as
    # `modules.base_utils.datasets` from the project root.
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from modules.base_utils.datasets import load_dataset
    ds = load_dataset(dataset, train=True)
    images = []
    for i in range(len(ds)):
        x, _ = ds[i]
        # x is a PIL image
        images.append(np.array(x))
    return np.stack(images, axis=0)


def _grid(images: np.ndarray, titles, suptitle: str, out_stem: Path,
           n_cols: int = 8, n_rows: int = 8):
    n_total = n_cols * n_rows
    n_show = min(len(images), n_total)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.7))
    axes = np.array(axes).reshape(n_rows, n_cols)
    for i in range(n_total):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        if i < n_show:
            ax.imshow(images[i])
            if titles is not None:
                ax.set_title(titles[i], fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    _save(fig, out_stem)


def _format_title(assigned: int, true: int, score: float, max_classes: int = 10):
    def lbl(i):
        if i < len(CIFAR10_CLASSES):
            return CIFAR10_CLASSES[i]
        return str(i)
    base = f"y={lbl(assigned)}\ny*={lbl(true)}"
    if score is not None and not np.isnan(score):
        base += f"\ns={score:.2f}"
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name-prefix", required=True)
    ap.add_argument("--k-values", type=int, nargs="+", required=True)
    ap.add_argument("--dataset", default="cifar")
    ap.add_argument("--poisoner", default="1xs")
    ap.add_argument("--target-label", type=int, default=4)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--max-k", type=int, default=4,
                     help="Cap how many k values get full TP/FP/FN/TN grids.")
    args = ap.parse_args()

    project_root = Path(args.project_root) if args.project_root else \
        Path(__file__).resolve().parents[2]
    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = project_root / report_dir

    master = pd.read_csv(report_dir / "master_metrics.csv")
    sample_row = master[master["mode"] == "remove"].iloc[0]
    dataset = sample_row["dataset"]
    user_model = sample_row["user_model"]
    poisoner = sample_row["poisoner"]
    target_label = int(sample_row["target_label"])

    fig_dir = report_dir / "figures" / "samples"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("[plot_samples] loading CIFAR training images ...")
    try:
        images = _load_cifar_train_images(project_root, args.dataset)
    except Exception as e:
        print(f"[plot_samples] could not load CIFAR images: {e}")
        return

    # Load ground truth once (it's the same across k values).
    true_path = project_root / "precomputed_labels" / dataset / user_model \
        / poisoner / "true.npy"
    in_path = project_root / "precomputed_labels" / dataset / user_model \
        / poisoner / "1500.npy"
    if not true_path.exists():
        print(f"[plot_samples] missing {true_path}; cannot build sample grids.")
        return
    if not in_path.exists():
        in_path = None
    true = np.load(true_path)
    true_hard = true.argmax(axis=1) if true.ndim == 2 else true.astype(int)

    # ---- A. Pick representative k values (lowest, middle, highest, k=20) ----
    chosen = []
    for k in args.k_values:
        if k == 20 or k in {args.k_values[0], args.k_values[-1]}:
            chosen.append(k)
    if len(args.k_values) >= 3 and len(chosen) < args.max_k:
        mid = args.k_values[len(args.k_values) // 2]
        if mid not in chosen:
            chosen.append(mid)
    chosen = sorted(set(chosen))[:args.max_k]

    rng = np.random.default_rng(0)

    for k in chosen:
        exp = f"{args.name_prefix}_remove_k{k}"
        labels_p = project_root / "experiments" / exp / "labels.npy"
        scores_p = project_root / "experiments" / exp / "scores.npy"
        if not labels_p.exists() or not scores_p.exists():
            continue
        labels = np.load(labels_p)
        if labels.ndim == 2:
            labels_hard = labels.argmax(axis=1)
        else:
            labels_hard = labels.astype(int)
        scores = np.load(scores_p)
        gt = (labels_hard != true_hard).astype(np.int32)
        flagged = scores > args.threshold

        for kind, mask, suptitle_extra in [
            ("TP", flagged & (gt == 1),  "correctly flagged poison"),
            ("FN", (~flagged) & (gt == 1), "missed poison"),
            ("FP", flagged & (gt == 0),  "false-alarm clean"),
            ("TN", (~flagged) & (gt == 0), "correctly kept clean"),
        ]:
            idx_pool = np.where(mask)[0]
            if len(idx_pool) == 0:
                continue
            n_pick = min(64, len(idx_pool))
            order_by_score = np.argsort(-scores[idx_pool], kind="stable")
            picks = idx_pool[order_by_score[:n_pick]]
            titles = [_format_title(int(labels_hard[i]), int(true_hard[i]),
                                      float(scores[i])) for i in picks]
            _grid(
                images=images[picks],
                titles=titles,
                suptitle=f"{kind}: {suptitle_extra}, k={k}, "
                          f"threshold={args.threshold:g}, "
                          f"n={len(idx_pool)} (showing top {n_pick} by score)",
                out_stem=fig_dir / f"samples_{kind}_k{k}",
            )

        # Top-32 most suspicious (regardless of GT) — useful for thesis
        top = np.argsort(-scores, kind="stable")[:64]
        titles = [_format_title(int(labels_hard[i]), int(true_hard[i]),
                                  float(scores[i])) for i in top]
        _grid(
            images=images[top],
            titles=titles,
            suptitle=f"Top-64 most suspicious samples by k-NN score (k={k})",
            out_stem=fig_dir / f"samples_top_scores_k{k}",
        )

    # ---- B. Target-class overview (clean samples) --------------------------
    tgt_mask = (true_hard == target_label) & (true_hard == true_hard)
    tgt_idx = np.where(tgt_mask)[0]
    if len(tgt_idx) > 0:
        pick = rng.choice(tgt_idx, size=min(64, len(tgt_idx)), replace=False)
        _grid(
            images=images[pick],
            titles=None,
            suptitle=f"Random clean target-class images "
                      f"({CIFAR10_CLASSES[target_label]})",
            out_stem=fig_dir / "samples_target_class_overview",
        )

    # ---- C. Poison examples (label-flipped) --------------------------------
    if in_path is not None and in_path.exists():
        in_labels = np.load(in_path)
        in_hard = in_labels.argmax(axis=1) if in_labels.ndim == 2 \
            else in_labels.astype(int)
        gt_full = (in_hard != true_hard).astype(np.int32)
        pois_idx = np.where(gt_full == 1)[0]
        if len(pois_idx) > 0:
            pick = rng.choice(pois_idx, size=min(64, len(pois_idx)), replace=False)
            titles = [_format_title(int(in_hard[i]), int(true_hard[i]), float("nan"))
                      for i in pick]
            _grid(
                images=images[pick],
                titles=titles,
                suptitle="FLIP poison examples (clean image, flipped label)",
                out_stem=fig_dir / "samples_poison_examples",
            )

    print(f"[plot_samples] wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
