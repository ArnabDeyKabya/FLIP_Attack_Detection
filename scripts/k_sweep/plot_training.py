"""
Training-dynamics + CTA/PTA plots for the Deep k-NN defense thesis sweep.

Generates:

    Per-k figures:
        training_curve_k{k}.{png,pdf}            — CTA and PTA across epochs,
                                                     none vs remove overlaid

    Aggregate (multi-k) figures:
        k_sweep_cta_pta.{png,pdf}                — final CTA and PTA vs k
        cta_pta_tradeoff.{png,pdf}               — scatter of (CTA, PTA),
                                                     one point per (k, mode)
        defense_success_vs_k.{png,pdf}           — defense success defined as
                                                     PTA_none - PTA_remove
                                                     and (PTA_none - PTA_remove)
                                                     / PTA_none vs k
        bar_compare_none_remove.{png,pdf}        — grouped bar chart, CTA &
                                                     PTA × k × mode
        training_curve_facet_cta.{png,pdf}       — small-multiples per-k CTA
        training_curve_facet_pta.{png,pdf}       — small-multiples per-k PTA
"""

import argparse
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
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "legend.frameon": False,
})


def _save(fig, out_stem: Path):
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"))
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


# ----- per-k figures ----------------------------------------------------------

def plot_training_curve(tr_k: pd.DataFrame, k: int, fig_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for mode, sub in tr_k.groupby("mode"):
        sub = sub.sort_values("epoch")
        axes[0].plot(sub["epoch"], sub["cta"], label=mode, linewidth=1.5)
        axes[1].plot(sub["epoch"], sub["pta"], label=mode, linewidth=1.5)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("Clean Test Accuracy (CTA)")
    axes[0].set_title(f"CTA over training (k = {k})"); axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("Poison Test Accuracy (PTA)")
    axes[1].set_title(f"PTA over training (k = {k})"); axes[1].set_ylim(0, 1)
    axes[1].legend()
    fig.tight_layout()
    _save(fig, fig_dir / f"training_curve_k{k}")


# ----- aggregate figures ------------------------------------------------------

def plot_k_sweep_cta_pta(master: pd.DataFrame, fig_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for mode, sub in master.groupby("mode"):
        sub = sub.sort_values("k")
        axes[0].plot(sub["k"], sub["cta"], marker="o", label=mode, linewidth=1.6)
        axes[1].plot(sub["k"], sub["pta"], marker="o", label=mode, linewidth=1.6)
    for ax, ylabel, title in [
        (axes[0], "Clean Test Accuracy (CTA)", "CTA vs k"),
        (axes[1], "Poison Test Accuracy (PTA)", "PTA vs k"),
    ]:
        ax.set_xscale("log"); ax.set_xlabel("k")
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_ylim(0, 1.02); ax.legend()
    fig.tight_layout()
    _save(fig, fig_dir / "k_sweep_cta_pta")


def plot_cta_pta_tradeoff(master: pd.DataFrame, fig_dir: Path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors = {"none": "#1f77b4", "remove": "#d62728"}
    for mode, sub in master.groupby("mode"):
        sub = sub.sort_values("k")
        ax.plot(sub["cta"], sub["pta"], marker="o", color=colors.get(mode, None),
                label=mode, linewidth=1.4)
        for _, r in sub.iterrows():
            ax.annotate(f"k={int(r['k'])}", (r["cta"], r["pta"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7,
                        color=colors.get(mode, "black"))
    ax.set_xlabel("Clean Test Accuracy (CTA)")
    ax.set_ylabel("Poison Test Accuracy (PTA)")
    ax.set_title("CTA / PTA trade-off across k\n"
                 "(bottom-right = strong defense + preserved clean accuracy)")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    # Annotate quadrants
    ax.axhline(0.5, linestyle=":", color="gray", linewidth=0.8)
    ax.axvline(0.5, linestyle=":", color="gray", linewidth=0.8)
    ax.legend()
    _save(fig, fig_dir / "cta_pta_tradeoff")


def plot_defense_success_vs_k(master: pd.DataFrame, fig_dir: Path):
    """PTA_none - PTA_remove (absolute) and relative drop, vs k."""
    if not {"none", "remove"}.issubset(set(master["mode"].unique())):
        return  # need both modes
    pivot = master.pivot_table(index="k", columns="mode",
                                values=["cta", "pta"]).sort_index()
    pivot.columns = [f"{m}_{c}" for c, m in pivot.columns]
    pivot["pta_drop"]  = pivot["none_pta"] - pivot["remove_pta"]
    pivot["pta_drop_rel"] = pivot["pta_drop"] / pivot["none_pta"].clip(lower=1e-6)
    pivot["cta_drop"]  = pivot["none_cta"] - pivot["remove_cta"]
    pivot = pivot.reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    axes[0].plot(pivot["k"], pivot["pta_drop"], marker="o", color="#d62728",
                 label="absolute PTA drop")
    axes[0].plot(pivot["k"], pivot["cta_drop"], marker="s", color="#1f77b4",
                 label="absolute CTA drop")
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].set_xscale("log"); axes[0].set_xlabel("k")
    axes[0].set_ylabel("accuracy drop (none − remove)")
    axes[0].set_title("Effect of defense on test accuracies")
    axes[0].legend()
    axes[1].plot(pivot["k"], pivot["pta_drop_rel"], marker="o", color="#d62728")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_xscale("log"); axes[1].set_xlabel("k")
    axes[1].set_ylabel("relative PTA drop")
    axes[1].set_title("Defense success rate (relative PTA collapse)")
    axes[1].set_ylim(-0.05, 1.05)
    fig.tight_layout()
    _save(fig, fig_dir / "defense_success_vs_k")


def plot_bar_compare(master: pd.DataFrame, fig_dir: Path):
    if not {"none", "remove"}.issubset(set(master["mode"].unique())):
        return
    ks = sorted(master["k"].unique())
    width = 0.38
    x = np.arange(len(ks))
    cta_none   = master[master["mode"] == "none"].set_index("k").reindex(ks)["cta"].values
    cta_remove = master[master["mode"] == "remove"].set_index("k").reindex(ks)["cta"].values
    pta_none   = master[master["mode"] == "none"].set_index("k").reindex(ks)["pta"].values
    pta_remove = master[master["mode"] == "remove"].set_index("k").reindex(ks)["pta"].values

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    axes[0].bar(x - width / 2, cta_none, width, label="none", color="#1f77b4")
    axes[0].bar(x + width / 2, cta_remove, width, label="remove", color="#d62728")
    axes[0].set_xticks(x); axes[0].set_xticklabels(ks)
    axes[0].set_xlabel("k"); axes[0].set_ylabel("CTA")
    axes[0].set_title("Clean Test Accuracy: none vs remove")
    axes[0].set_ylim(0, 1.02); axes[0].legend()

    axes[1].bar(x - width / 2, pta_none, width, label="none", color="#1f77b4")
    axes[1].bar(x + width / 2, pta_remove, width, label="remove", color="#d62728")
    axes[1].set_xticks(x); axes[1].set_xticklabels(ks)
    axes[1].set_xlabel("k"); axes[1].set_ylabel("PTA")
    axes[1].set_title("Poison Test Accuracy: none vs remove\n(lower = stronger defense)")
    axes[1].set_ylim(0, 1.02); axes[1].legend()
    fig.tight_layout()
    _save(fig, fig_dir / "bar_compare_none_remove")


def plot_training_facet(tr: pd.DataFrame, fig_dir: Path, target_col: str,
                         stem: str, ylabel: str):
    ks = sorted(tr["k"].unique())
    n = len(ks)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.6 * rows),
                              sharex=True, sharey=True)
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    for ax, k in zip(axes_flat, ks):
        sub = tr[tr["k"] == k]
        for mode, s in sub.groupby("mode"):
            s = s.sort_values("epoch")
            ax.plot(s["epoch"], s[target_col], label=mode, linewidth=1.4)
        ax.set_title(f"k = {k}", fontsize=10)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=8)
    for ax in list(axes_flat)[len(ks):]:
        ax.set_visible(False)
    fig.supxlabel("epoch")
    fig.supylabel(ylabel)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    _save(fig, fig_dir / stem)


# ----- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name-prefix", required=True)
    ap.add_argument("--k-values", type=int, nargs="+", required=True)
    ap.add_argument("--modes", nargs="+", default=["none", "remove"])
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    project_root = Path(args.project_root) if args.project_root else \
        Path(__file__).resolve().parents[2]
    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = project_root / report_dir

    master = pd.read_csv(report_dir / "master_metrics.csv")
    tr = pd.read_csv(report_dir / "training_curves.csv") \
        if (report_dir / "training_curves.csv").exists() else None

    fig_dir = report_dir / "figures" / "training"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if tr is not None and not tr.empty:
        for k in args.k_values:
            tr_k = tr[tr["k"] == k]
            if not tr_k.empty:
                plot_training_curve(tr_k, k, fig_dir)
        plot_training_facet(tr, fig_dir, "cta",
                             "training_curve_facet_cta", "CTA")
        plot_training_facet(tr, fig_dir, "pta",
                             "training_curve_facet_pta", "PTA")

    plot_k_sweep_cta_pta(master, fig_dir)
    plot_cta_pta_tradeoff(master, fig_dir)
    plot_defense_success_vs_k(master, fig_dir)
    plot_bar_compare(master, fig_dir)

    print(f"[plot_training] wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
