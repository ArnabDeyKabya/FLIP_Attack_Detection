"""
Predict which (source -> target) pairs a semantic label audit can defend,
without running a single attack or training a single model.

THE QUANTITY
------------
The audit flags a sample when most of its k nearest neighbours disagree with
its label. So for a source-class image relabelled to the target, the score is

    audit_score = 1 - (fraction of its k neighbours that carry the target label)

which depends only on the encoder's neighbourhood structure — not on the attack,
the trigger, the budget, or the victim model. Two numbers follow directly:

    mean audit score   how visible the average source->target flip is
    stealthy pool      how many source-class images could be flipped and stay
                       under the audit threshold

The pool is the one that decides outcomes. The mean is a red herring: even the
closest CIFAR-10 pair has a mean score above 0.9, yet its pool is large enough
to matter, because an attacker needs the tail, not the average.

WHY THIS IS THE CONTRIBUTION
----------------------------
It converts "does this defense work?" from an empirical question requiring a
full attack pipeline into a property of the encoder measurable in seconds. It
also explains the truck->deer result: that pair's stealthy pool is literally
zero, so the constrained attacker there had no viable move and its collapse was
structural, not a sign of a strong defense.

Note the pool counts SOURCE-CLASS images specifically. Flips outside the source
class do not install a source->target backdoor, so they inflate any pool count
that ignores the class.

Scores here use clean neighbour labels, so pools are lower bounds: once many
flips land in one region they shield each other (see the refine_iters loop in
modules/select_flips_adaptive).

Usage:
    python scripts/adaptive/pair_analysis.py
    python scripts/adaptive/pair_analysis.py --dataset cifar_100 --top 30
    python scripts/adaptive/pair_analysis.py --encoder dinov2_vitb14
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.knn_defense.knn_detector import KNNDetector          # noqa: E402
from modules.knn_defense.ssl_features import SSLFeatureExtractor  # noqa: E402
from modules.base_utils.datasets import load_dataset              # noqa: E402


CIFAR10_CLASSES = ["plane", "car", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

# Validated single-hue ordinal ramp (monotone lightness, adjacent dL >= 0.06,
# light end >= 2:1 on the surface). Sequential magnitude takes one hue, never a
# rainbow, so the same steps drive the heatmap colormap.
RAMP = ["#7fb2ec", "#589ae4", "#3a7fd8", "#2867b6", "#1c4c87", "#122f56"]
C_BLUE, C_ORANGE = "#2a78d6", "#eb6834"
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#dcdcda"

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10,
    "axes.labelcolor": INK, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    "legend.frameon": False, "lines.linewidth": 2.0,
})


def class_names(dataset_flag):
    if dataset_flag == "cifar":
        return CIFAR10_CLASSES
    ds = load_dataset(dataset_flag, train=True)
    names = getattr(ds, "classes", None)
    return list(names) if names else None


def compute(features, labels, k, n_classes, taus):
    """Neighbour composition and stealthy pool sizes for every ordered pair."""
    nbr_idx, _ = KNNDetector(k=k).neighbor_graph(features)
    nbr_lbl = labels[nbr_idx]                                   # (N, k)

    composition = np.zeros((n_classes, n_classes))
    pools = {t: np.zeros((n_classes, n_classes), dtype=int) for t in taus}

    for a in range(n_classes):
        rows = nbr_lbl[labels == a]
        if len(rows) == 0:
            continue
        for b in range(n_classes):
            frac = (rows == b).mean(axis=1)                     # per-sample
            composition[a, b] = frac.mean()
            if a == b:
                continue
            score = 1.0 - frac        # audit score if relabelled b
            for t in taus:
                pools[t][a, b] = int((score <= t).sum())

    return composition, pools


def write_csv(path, names, composition, pools, taus, class_counts):
    n = composition.shape[0]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_idx", "target_idx", "source", "target",
                    "source_class_size", "neighbour_frac_target",
                    "mean_audit_score"]
                   + [f"stealthy_pool_tau{t}" for t in taus])
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                w.writerow([a, b, names[a], names[b], class_counts[a],
                            f"{composition[a, b]:.6f}",
                            f"{1 - composition[a, b]:.6f}"]
                           + [pools[t][a, b] for t in taus])


def write_tables(out_dir, names, composition, pools, taus, top, highlight):
    n = composition.shape[0]
    pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    key = taus[-1]
    pairs.sort(key=lambda p: -pools[key][p[0], p[1]])

    lines = [
        "**Stealthy flip pool per class pair.** How many *source-class* images "
        "could be relabelled to the target and still score at or below the "
        "audit threshold. Computed from encoder neighbourhoods alone -- no "
        "attack, no training. A pair with a pool of zero cannot be attacked "
        "stealthily at all, which is why the defense appears total there.",
        "",
        "| source -> target | mean audit score | "
        + " | ".join(f"pool @ tau<={t}" for t in taus) + " |",
        "|---" * (2 + len(taus)) + "|",
    ]
    shown = pairs[:top]
    if highlight and highlight not in shown:
        shown = shown + [highlight]
    for i, (a, b) in enumerate(shown):
        mark = "  **<- yours**" if (a, b) == highlight else ""
        if highlight and i == top:
            lines.append("| ... | | " + " | ".join("" for _ in taus) + " |")
        lines.append(
            f"| {names[a]} -> {names[b]}{mark} | "
            f"{1 - composition[a, b]:.3f} | "
            + " | ".join(str(pools[t][a, b]) for t in taus) + " |"
        )
    (out_dir / "pair_stealth.md").write_text("\n".join(lines) + "\n")

    tex = ["\\begin{table}[t]", "\\centering", "\\small",
           "\\begin{tabular}{l" + "r" * (1 + len(taus)) + "}", "\\toprule",
           "source $\\to$ target & mean audit score & "
           + " & ".join(f"pool $\\tau \\le {t}$" for t in taus) + " \\\\",
           "\\midrule"]
    for a, b in shown:
        tex.append(f"{names[a]} $\\to$ {names[b]} & "
                   f"{1 - composition[a, b]:.3f} & "
                   + " & ".join(str(pools[t][a, b]) for t in taus) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}",
            "\\caption{Stealthy flip pool per class pair, from encoder "
            "neighbourhoods alone. A zero pool means no stealthy attack "
            "exists for that pair.}",
            "\\label{tab:pair_stealth}", "\\end{table}"]
    (out_dir / "pair_stealth.tex").write_text("\n".join(tex) + "\n")
    print(f"  table: pair_stealth.{{md,tex}}")
    return pairs


def fig_heatmap(out_dir, names, composition):
    """Sequential magnitude -> one hue, light to dark, with a scale legend."""
    n = composition.shape[0]
    if n > 20:
        print("  skipped heatmap: too many classes to label legibly")
        return
    cmap = LinearSegmentedColormap.from_list("audit_blue", ["#ffffff"] + RAMP)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.grid(False)
    # Own-class dominates every row; clipping the scale keeps the off-diagonal
    # structure — the part that decides stealth — from washing out.
    shown = composition.copy()
    np.fill_diagonal(shown, np.nan)
    im = ax.imshow(shown, cmap=cmap, vmin=0, vmax=np.nanmax(shown))

    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    ax.set_xlabel("neighbour class (potential target)")
    ax.set_ylabel("true class (potential source)")
    ax.set_title("Where the encoder blurs classes together")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("fraction of neighbours", color=INK_SOFT)
    cbar.outline.set_edgecolor(GRID)

    for a in range(n):
        for b in range(n):
            if a == b or composition[a, b] < 0.02:
                continue
            ax.text(b, a, f"{composition[a, b]*100:.0f}", ha="center",
                    va="center", fontsize=6.5,
                    color="#ffffff" if composition[a, b] > 0.05 else INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"neighbour_composition.{ext}")
    plt.close(fig)
    print("  figure: neighbour_composition.{png,pdf}")


def fig_pool_ranking(out_dir, names, pools, tau, pairs, top, highlight):
    """One measure, so one hue — with the studied pair called out by emphasis."""
    shown = pairs[:top]
    if highlight and highlight not in shown:
        shown = shown + [highlight]
    labels = [f"{names[a]} → {names[b]}" for a, b in shown]
    values = [pools[tau][a, b] for a, b in shown]
    colors = [C_ORANGE if (a, b) == highlight else C_BLUE for a, b in shown]

    fig, ax = plt.subplots(figsize=(5.4, 0.3 * len(shown) + 1.4))
    y = np.arange(len(shown))[::-1]
    ax.barh(y, values, color=colors, height=0.7)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"source-class images flippable at audit score $\\leq$ {tau}")
    ax.set_title("How much room the attacker actually has")
    ax.xaxis.grid(True); ax.yaxis.grid(False)

    for yi, v, (a, b) in zip(y, values, shown):
        note = "  (studied pair)" if (a, b) == highlight else ""
        ax.text(v + max(values) * 0.015, yi, f"{v}{note}", va="center",
                fontsize=7.5, color=INK_SOFT)
    ax.set_xlim(0, max(values) * 1.28)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"stealthy_pool_tau{tau}.{ext}")
    plt.close(fig)
    print(f"  figure: stealthy_pool_tau{tau}.{{png,pdf}}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar", choices=["cifar", "cifar_100"])
    p.add_argument("--encoder", default="dinov2_vits14")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--taus", type=float, nargs="+", default=[0.9, 0.7, 0.5])
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--source-label", type=int, default=9)
    p.add_argument("--target-label", type=int, default=4)
    p.add_argument("--feature-cache", default="data/ssl_features")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir) if args.out_dir else (
        root / "experiments" / f"_report_pairs_{args.dataset}_{args.encoder}")
    out_dir.mkdir(parents=True, exist_ok=True)

    features = SSLFeatureExtractor(args.encoder).extract(
        dataset_flag=args.dataset, cache_dir=str(root / args.feature_cache))

    ds = load_dataset(args.dataset, train=True)
    labels = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int64)
    n_classes = int(labels.max()) + 1
    names = class_names(args.dataset) or [str(i) for i in range(n_classes)]
    counts = np.bincount(labels, minlength=n_classes)

    print(f"Computing pair stealth: {args.dataset}, {args.encoder}, "
          f"k={args.k}, {n_classes} classes, "
          f"{n_classes * (n_classes - 1)} ordered pairs...")
    composition, pools = compute(features, labels, args.k, n_classes, args.taus)

    highlight = (args.source_label, args.target_label)
    if not (0 <= highlight[0] < n_classes and 0 <= highlight[1] < n_classes):
        highlight = None

    write_csv(out_dir / "pair_stealth.csv", names, composition, pools,
              args.taus, counts)
    print("  metrics: pair_stealth.csv")
    pairs = write_tables(out_dir, names, composition, pools, args.taus,
                         args.top, highlight)
    fig_heatmap(out_dir, names, composition)
    fig_pool_ranking(out_dir, names, pools, args.taus[-1], pairs,
                     args.top, highlight)

    tau = args.taus[-1]
    best = pairs[0]
    print(f"\nMost attackable pair at tau<={tau}: "
          f"{names[best[0]]} -> {names[best[1]]} "
          f"({pools[tau][best[0]][best[1]]} stealthy source-class images)")
    if highlight:
        a, b = highlight
        print(f"Studied pair {names[a]} -> {names[b]}: "
              f"{pools[tau][a][b]} stealthy source-class images")
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
