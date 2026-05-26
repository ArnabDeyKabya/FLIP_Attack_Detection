"""
Feature-space + neighborhood plots for the Deep k-NN defense thesis sweep.

Generates:

    features_tsne_class.{png,pdf}              — t-SNE of cached SSL features,
                                                   colored by TRUE class label
    features_tsne_poison.{png,pdf}             — same projection, colored by
                                                   poison vs clean
    features_tsne_score_k{k}.{png,pdf}         — same projection, colored by
                                                   k-NN disagreement score (one
                                                   per k in --score-k-values)
    features_umap_*.{png,pdf}                  — UMAP versions of the above
                                                   (skipped if umap-learn not
                                                   installed)
    features_pair_distance.{png,pdf}           — distribution of cosine
                                                   similarities for
                                                   within-class vs between-class
                                                   vs poison-to-target pairs
    features_neighbor_purity_k{k}.{png,pdf}    — for each sample i, fraction of
                                                   k neighbors sharing its TRUE
                                                   label; hist + ECDF, split
                                                   by poison vs clean

Heavy plots (t-SNE, UMAP) are computed on a subsample (--subsample N) for
runtime; the per-pair-distance and neighbor-purity plots use the full feature
matrix.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm


plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "legend.frameon": False,
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


def _load_features(project_root: Path, dataset: str, encoder: str,
                    cache_dir: str = "data/ssl_features"):
    p = project_root / cache_dir / f"{dataset}_{encoder}_train.npy"
    if not p.exists():
        raise FileNotFoundError(
            f"SSL feature cache not found at {p}. Run an experiment first to "
            f"populate the cache, or pass --skip-features to skip this step."
        )
    return np.load(p)


def _load_ground_truth(project_root: Path, exp_name: str,
                        dataset: str, user_model: str, poisoner: str):
    exp_dir = project_root / "experiments" / exp_name
    if not (exp_dir / "labels.npy").exists():
        return None, None, None
    labels = np.load(exp_dir / "labels.npy")
    if labels.ndim == 2:
        labels_hard = labels.argmax(axis=1)
    else:
        labels_hard = labels.astype(int)
    true_path = project_root / "precomputed_labels" / dataset / user_model \
        / poisoner / "true.npy"
    if not true_path.exists():
        return None, None, None
    true = np.load(true_path)
    if true.ndim == 2:
        true_hard = true.argmax(axis=1)
    else:
        true_hard = true.astype(int)
    gt = (labels_hard != true_hard).astype(np.int32)
    return gt, labels_hard, true_hard


def _tsne(features: np.ndarray, subsample_idx: np.ndarray,
          random_state: int = 0):
    from sklearn.manifold import TSNE
    sub = features[subsample_idx]
    return TSNE(n_components=2, init="pca", learning_rate="auto",
                perplexity=30, random_state=random_state,
                n_iter=1000).fit_transform(sub)


def _umap(features: np.ndarray, subsample_idx: np.ndarray, random_state: int = 0):
    try:
        import umap
    except ImportError:
        return None
    sub = features[subsample_idx]
    return umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                     random_state=random_state).fit_transform(sub)


# ----- scatter helpers --------------------------------------------------------

def plot_embed_by_class(embed, classes_sub, name, fig_dir: Path):
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    cmap = cm.get_cmap("tab10", 10)
    for c in range(10):
        m = classes_sub == c
        if m.sum() == 0:
            continue
        ax.scatter(embed[m, 0], embed[m, 1], s=4, alpha=0.55, color=cmap(c),
                    label=CIFAR10_CLASSES[c] if c < len(CIFAR10_CLASSES) else str(c))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{name}: SSL features colored by true class")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), markerscale=2.5,
              fontsize=9)
    _save(fig, fig_dir / f"features_{name}_class")


def plot_embed_by_poison(embed, poison_sub, name, fig_dir: Path):
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    m_clean = poison_sub == 0
    m_pois  = poison_sub == 1
    ax.scatter(embed[m_clean, 0], embed[m_clean, 1], s=4, alpha=0.4,
                color="#1f77b4", label=f"clean (n={int(m_clean.sum())})")
    ax.scatter(embed[m_pois,  0], embed[m_pois,  1], s=10, alpha=0.9,
                color="#d62728", label=f"poisoned (n={int(m_pois.sum())})",
                edgecolors="black", linewidths=0.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{name}: SSL features colored by poison vs clean")
    ax.legend(loc="lower right", markerscale=2)
    _save(fig, fig_dir / f"features_{name}_poison")


def plot_embed_by_score(embed, scores_sub, name, k, fig_dir: Path):
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    sc = ax.scatter(embed[:, 0], embed[:, 1], s=4, c=scores_sub, cmap="viridis",
                     alpha=0.65)
    fig.colorbar(sc, ax=ax, label="k-NN disagreement score")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{name}: SSL features colored by k-NN score (k = {k})")
    _save(fig, fig_dir / f"features_{name}_score_k{k}")


# ----- pairwise distance ------------------------------------------------------

def plot_pair_distance(features: np.ndarray, true_hard: np.ndarray,
                        gt_poison: np.ndarray, source_label: int,
                        target_label: int, fig_dir: Path,
                        n_pairs: int = 20000, random_state: int = 0):
    rng = np.random.default_rng(random_state)
    N = features.shape[0]

    def _sample_sims(mask_a, mask_b, n):
        idx_a = np.where(mask_a)[0]
        idx_b = np.where(mask_b)[0]
        if len(idx_a) == 0 or len(idx_b) == 0:
            return np.empty(0)
        i = rng.choice(idx_a, size=n, replace=True)
        j = rng.choice(idx_b, size=n, replace=True)
        return (features[i] * features[j]).sum(axis=1)

    in_class_mask = true_hard == target_label
    out_class_mask = true_hard != target_label
    poison_mask = gt_poison.astype(bool)

    sims_within_target = _sample_sims(in_class_mask & ~poison_mask,
                                       in_class_mask & ~poison_mask, n_pairs)
    sims_target_to_other = _sample_sims(in_class_mask & ~poison_mask,
                                         out_class_mask & ~poison_mask, n_pairs)
    sims_poison_to_target = _sample_sims(poison_mask,
                                          in_class_mask & ~poison_mask, n_pairs)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bins = np.linspace(-0.1, 1.0, 60)
    for arr, label, color in [
        (sims_within_target,    f"within target class ({CIFAR10_CLASSES[target_label]})", "#1f77b4"),
        (sims_target_to_other,  "target  any other class", "#7f7f7f"),
        (sims_poison_to_target, f"poison  target class", "#d62728"),
    ]:
        if len(arr) == 0:
            continue
        ax.hist(arr, bins=bins, alpha=0.55, label=label, color=color,
                density=True)
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title("Pairwise SSL feature similarity\n"
                 "(strong poison signal: poison-to-target peak overlaps within-class peak)")
    ax.legend()
    _save(fig, fig_dir / "features_pair_distance")


# ----- neighbor purity --------------------------------------------------------

def plot_neighbor_purity(neighbor_indices: np.ndarray, true_hard: np.ndarray,
                          gt_poison: np.ndarray, k: int, fig_dir: Path):
    """Fraction of k neighbors that share the sample's TRUE label."""
    neighbor_labels = true_hard[neighbor_indices]   # (N, k)
    my_labels = true_hard[:, None]                  # (N, 1)
    purity = (neighbor_labels == my_labels).mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    bins = np.linspace(0, 1, 41)
    axes[0].hist(purity[gt_poison == 0], bins=bins, alpha=0.6, color="#1f77b4",
                  label=f"clean (median = {np.median(purity[gt_poison == 0]):.2f})")
    axes[0].hist(purity[gt_poison == 1], bins=bins, alpha=0.6, color="#d62728",
                  label=f"poisoned (median = {np.median(purity[gt_poison == 1]):.2f})")
    axes[0].set_xlabel("neighbor purity  (fraction matching true label)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Neighbor purity, k = {k}")
    axes[0].legend()

    # ECDF
    for arr, label, color in [
        (purity[gt_poison == 0], "clean",    "#1f77b4"),
        (purity[gt_poison == 1], "poisoned", "#d62728"),
    ]:
        if len(arr) == 0:
            continue
        s = np.sort(arr)
        cdf = np.arange(1, len(s) + 1) / len(s)
        axes[1].plot(s, cdf, label=label, color=color, linewidth=1.6)
    axes[1].set_xlabel("neighbor purity")
    axes[1].set_ylabel("ECDF")
    axes[1].set_title(f"ECDF of neighbor purity, k = {k}")
    axes[1].legend()
    fig.tight_layout()
    _save(fig, fig_dir / f"features_neighbor_purity_k{k}")


# ----- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name-prefix", required=True)
    ap.add_argument("--k-values", type=int, nargs="+", required=True)
    ap.add_argument("--modes", nargs="+", default=["none", "remove"])
    ap.add_argument("--dataset", default="cifar")
    ap.add_argument("--encoder", default="dinov2_vits14")
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--subsample", type=int, default=5000,
                     help="Subsample size for t-SNE / UMAP scatterplots.")
    ap.add_argument("--score-k-values", type=int, nargs="*", default=None,
                     help="Which k values to color the embedding by score for. "
                          "Defaults to a representative subset of --k-values.")
    ap.add_argument("--skip-tsne", action="store_true")
    ap.add_argument("--skip-umap", action="store_true")
    ap.add_argument("--skip-features", action="store_true",
                     help="Skip everything that needs the SSL feature cache.")
    args = ap.parse_args()

    project_root = Path(args.project_root) if args.project_root else \
        Path(__file__).resolve().parents[2]
    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = project_root / report_dir

    fig_dir = report_dir / "figures" / "features"
    fig_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(report_dir / "master_metrics.csv")
    pick_mode = "remove" if "remove" in args.modes else args.modes[0]
    sample_row = master[(master["mode"] == pick_mode)].iloc[0]
    dataset = sample_row["dataset"]
    user_model = sample_row["user_model"]
    poisoner = sample_row["poisoner"]
    source_label = int(sample_row["source_label"])
    target_label = int(sample_row["target_label"])

    if args.skip_features:
        print("[plot_features] --skip-features set; nothing to do.")
        return

    try:
        features = _load_features(project_root, dataset=args.dataset,
                                    encoder=args.encoder)
    except FileNotFoundError as e:
        print(f"[plot_features] {e}; skipping feature-space plots.")
        return

    # Reload ground truth (poison + true labels) from any successful run; all
    # k-mode-experiments share the same FLIP label tensor and true tensor, so
    # the masks are identical regardless of which one we read.
    exp = f"{args.name_prefix}_{pick_mode}_k{args.k_values[0]}"
    gt, labels_hard, true_hard = _load_ground_truth(
        project_root, exp, dataset, user_model, poisoner,
    )
    if gt is None:
        print(f"[plot_features] missing labels.npy / true.npy for {exp}; "
              f"skipping feature-space plots.")
        return

    # ---- pairwise distance + neighbor purity (use full N) -------------------
    plot_pair_distance(features, true_hard, gt, source_label, target_label,
                        fig_dir)

    # Neighbor purity per k (need neighbor_indices)
    for k in args.k_values:
        exp = f"{args.name_prefix}_{pick_mode}_k{k}"
        # The k-NN detector does not currently persist neighbor_indices to disk
        # in run_module — only `scores.npy` is saved. Recompute neighborhood on
        # demand using a quick top-k via numpy (small k cases are cheap).
        n_path = project_root / "experiments" / exp / "neighbor_indices.npy"
        if n_path.exists():
            neigh = np.load(n_path)
        else:
            # Fall back to a fast chunked top-k via torch when available.
            try:
                import torch
                feats_t = torch.from_numpy(features.astype(np.float32))
                if torch.cuda.is_available():
                    feats_t = feats_t.cuda()
                N = feats_t.shape[0]
                topk = []
                chunk = 1024
                for s in range(0, N, chunk):
                    sims = feats_t[s:s + chunk] @ feats_t.T
                    # mask self
                    li = torch.arange(sims.shape[0], device=sims.device)
                    gi = torch.arange(s, s + sims.shape[0], device=sims.device)
                    sims[li, gi] = float("-inf")
                    _, ind = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)
                    topk.append(ind.cpu().numpy())
                neigh = np.concatenate(topk, axis=0)
            except ImportError:
                print(f"[plot_features] torch missing; skipping neighbor_purity for k={k}")
                continue
        plot_neighbor_purity(neigh, true_hard, gt, k, fig_dir)

    # ---- t-SNE / UMAP scatter (subsampled) ----------------------------------
    rng = np.random.default_rng(0)
    if len(features) > args.subsample:
        # Stratify: include all poisons + sample clean
        poison_idx = np.where(gt == 1)[0]
        clean_idx = np.where(gt == 0)[0]
        n_clean = max(args.subsample - len(poison_idx), 0)
        clean_pick = rng.choice(clean_idx, size=min(n_clean, len(clean_idx)),
                                 replace=False)
        subsample_idx = np.concatenate([poison_idx, clean_pick])
        rng.shuffle(subsample_idx)
    else:
        subsample_idx = np.arange(len(features))

    classes_sub = true_hard[subsample_idx]
    poison_sub  = gt[subsample_idx]

    if not args.skip_tsne:
        print(f"[plot_features] running t-SNE on n={len(subsample_idx)} ...")
        try:
            embed = _tsne(features, subsample_idx)
            plot_embed_by_class(embed, classes_sub, "tsne", fig_dir)
            plot_embed_by_poison(embed, poison_sub, "tsne", fig_dir)
            score_ks = args.score_k_values or [args.k_values[0],
                                                args.k_values[len(args.k_values) // 2],
                                                args.k_values[-1]]
            for k in score_ks:
                exp = f"{args.name_prefix}_{pick_mode}_k{k}"
                sp = project_root / "experiments" / exp / "scores.npy"
                if not sp.exists():
                    continue
                scores = np.load(sp)
                plot_embed_by_score(embed, scores[subsample_idx], "tsne", k,
                                     fig_dir)
        except Exception as e:
            print(f"[plot_features] t-SNE failed: {e}")

    if not args.skip_umap:
        print(f"[plot_features] trying UMAP on n={len(subsample_idx)} ...")
        try:
            embed = _umap(features, subsample_idx)
            if embed is None:
                print("[plot_features] umap-learn not installed; skipping UMAP")
            else:
                plot_embed_by_class(embed, classes_sub, "umap", fig_dir)
                plot_embed_by_poison(embed, poison_sub, "umap", fig_dir)
                score_ks = args.score_k_values or [args.k_values[0],
                                                    args.k_values[len(args.k_values) // 2],
                                                    args.k_values[-1]]
                for k in score_ks:
                    exp = f"{args.name_prefix}_{pick_mode}_k{k}"
                    sp = project_root / "experiments" / exp / "scores.npy"
                    if not sp.exists():
                        continue
                    scores = np.load(sp)
                    plot_embed_by_score(embed, scores[subsample_idx], "umap", k,
                                         fig_dir)
        except Exception as e:
            print(f"[plot_features] UMAP failed: {e}")

    print(f"[plot_features] wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
