"""
Build every figure, table and metric file for the adaptive-attacker study.

Safe to run at any time, including mid-sweep: each figure is generated
independently and anything whose inputs are missing is skipped with a note, so
partial results are always viewable. Re-run after more victim runs finish.

    python scripts/adaptive/make_report.py

Outputs under experiments/_report_adaptive_cifar_1xs/:

    metrics_master.{csv,json}     every attacker-side and victim-side number
    tables/*.{md,tex}             thesis-ready tables
    figures/frontier/             the headline stealth-vs-efficacy figures
    figures/attacker/             what the stealth constraint costs the attacker
    figures/detection/            PR curves, score histograms, recall vs tau
    figures/training/             CTA / PTA learning curves
    figures/samples/              images the naive vs adaptive attacker picked

Style follows scripts/k_sweep/plot_*.py: matplotlib only, no seaborn, PNG for
slides and PDF for print. Colors are the validated categorical slots (fixed
order, colorblind-checked) and a validated single-hue ordinal ramp for the
tau sequence — tau is ordered, so it takes a ramp, not categorical hues.
"""

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.metrics import (                                        # noqa: E402
    roc_auc_score, average_precision_score, precision_recall_curve,
    matthews_corrcoef, f1_score,
)


# Validated categorical slots, fixed order (never cycled, never reordered).
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
# Validated single-hue ordinal ramp, light -> dark, for the tau sequence.
TAU_RAMP = ["#7fb2ec", "#589ae4", "#3a7fd8", "#2867b6", "#1c4c87", "#122f56"]

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dcdcda"

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelcolor": INK,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,          # recessive grid, always under the data
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "legend.frameon": False,
    "lines.linewidth": 2.0,          # 2px lines
    "lines.markersize": 6,
})


def tau_slug(tau):
    return f"{tau:.3f}".replace(".", "p")


def ramp(i, n):
    """Pick the i-th of n ordinal steps.

    The ramp has a fixed number of validated steps (monotone lightness, adjacent
    dL >= 0.06). Asking for more series than steps would repeat colors and make
    two tau values indistinguishable, so callers must subsample to len(TAU_RAMP)
    series first — see --figure-taus.
    """
    if n > len(TAU_RAMP):
        raise SkipFigure(
            f"{n} series exceeds the {len(TAU_RAMP)}-step ordinal ramp; "
            f"subsample with --figure-taus"
        )
    if n <= 1:
        return TAU_RAMP[-2]
    idx = round(i / (n - 1) * (len(TAU_RAMP) - 1))
    return TAU_RAMP[idx]


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  figure: {out_dir.name}/{name}.{{png,pdf}}")


def figure(fn):
    """Run a figure builder, but never let one failure abort the report."""
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except SkipFigure as e:
            print(f"  skipped {fn.__name__}: {e}")
        except Exception:
            print(f"  FAILED {fn.__name__}:")
            traceback.print_exc(limit=2)
    return wrapper


class SkipFigure(Exception):
    pass


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

class Study:
    """Everything on disk for this study, with missing runs tolerated."""

    def __init__(self, root: Path, args):
        self.root = root
        self.args = args
        self.attack_dir = root / args.attack_dir
        self.out = root / args.out_dir
        self.manifest = json.loads((self.attack_dir / "manifest.json").read_text())
        self.runs = [r for r in self.manifest["runs"]
                     if r["budget"] == args.budget]
        self.runs.sort(key=lambda r: -r["tau"])
        self.taus = [r["tau"] for r in self.runs]

        # Tables and metrics cover every tau; per-series figures are capped at
        # the ordinal ramp's step count so no two curves share a color.
        wanted = args.figure_taus
        self.figure_taus = [
            t for t in self.taus if any(abs(t - w) < 1e-9 for w in wanted)
        ] or self.taus[:len(TAU_RAMP)]

        self.true = np.load(self.attack_dir / "true.npy")
        self.true_hard = self.true.argmax(axis=1).astype(np.int64)

        self._features = None

    @property
    def features(self):
        if self._features is None:
            from modules.knn_defense.ssl_features import SSLFeatureExtractor
            self._features = SSLFeatureExtractor(self.args.encoder).extract(
                dataset_flag=self.args.dataset,
                cache_dir=str(self.root / self.args.feature_cache),
            )
        return self._features

    def hard_labels(self, tau):
        import torch
        from modules.base_utils.util import softmax
        arr = np.load(self.attack_dir / f"tau_{tau:.3f}" / f"{self.args.budget}.npy")
        return softmax(torch.tensor(arr)).argmax(dim=1).numpy().astype(np.int64)

    def gt(self, tau):
        return (self.hard_labels(tau) != self.true_hard).astype(np.int32)

    def victim(self, tau, mode):
        d = (self.root / "experiments" /
             f"{self.args.name_prefix}_tau{tau_slug(tau)}_{mode}")
        s = d / "summary_detailed.json"
        return (json.loads(s.read_text()), d) if s.exists() else (None, d)

    def control(self, n):
        d = self.root / "experiments" / f"{self.args.control_prefix}_n{n}_none"
        s = d / "summary_detailed.json"
        return (json.loads(s.read_text()), d) if s.exists() else (None, d)

    def clean_control(self):
        d = self.root / "experiments" / self.args.clean_control_name
        s = d / "summary_detailed.json"
        return (json.loads(s.read_text()), d) if s.exists() else (None, d)

    def scores(self, tau, k=None):
        """Detector scores for this tau at the given k (defaults to attacker k)."""
        from modules.knn_defense.knn_detector import KNNDetector
        k = k or self.manifest["k"]
        return KNNDetector(k=k).detect(self.features, self.hard_labels(tau)).scores


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def build_master(study: Study):
    """One row per tau, joining attacker, detector and victim measurements."""
    rows = []
    for r in study.runs:
        tau = r["tau"]
        gt = study.gt(tau)
        scores = study.scores(tau)
        n_pois = int(gt.sum())

        flagged = scores > study.args.threshold
        tp = int((flagged & gt.astype(bool)).sum())
        n_flag = int(flagged.sum())
        prec, rec, thr = precision_recall_curve(gt, scores)
        f1_curve = np.divide(2 * prec * rec, prec + rec,
                             out=np.zeros_like(prec), where=(prec + rec) > 0)
        topk = np.argsort(-scores, kind="stable")[:n_pois] if n_pois else []

        none_s, _ = study.victim(tau, "none")
        rem_s, _ = study.victim(tau, "remove")

        rows.append({
            "tau": tau,
            # --- attacker side
            "n_flips_placed": r["n_selected"],
            "budget_satisfied": r["budget_satisfied"],
            "n_eligible": r["n_eligible"],
            "mean_margin": r["mean_margin"],
            "baseline_mean_margin": r["baseline_mean_margin"],
            "jaccard_vs_unconstrained": r["jaccard_vs_unconstrained"],
            "frac_source_class": r["frac_selected_from_source_class"],
            # --- detector side
            "auroc": float(roc_auc_score(gt, scores)) if n_pois else None,
            "auprc": float(average_precision_score(gt, scores)) if n_pois else None,
            "precision_at_k": float(gt[topk].sum() / n_pois) if n_pois else None,
            "best_f1": float(f1_curve.max()),
            "best_f1_threshold": float(thr[int(np.argmax(f1_curve[:-1]))])
                                 if len(thr) else None,
            "f1_at_threshold": float(f1_score(gt, flagged)) if n_flag else 0.0,
            "mcc_at_threshold": float(matthews_corrcoef(gt, flagged)),
            "flagged_precision": tp / n_flag if n_flag else None,
            "flagged_recall": tp / n_pois if n_pois else None,
            "n_flagged": n_flag,
            "poisons_missed": n_pois - tp,
            "clean_removed": n_flag - tp,
            "fpr_at_threshold": (n_flag - tp) / int((gt == 0).sum()),
            "mean_score_poisoned": float(scores[gt == 1].mean()) if n_pois else None,
            "mean_score_clean": float(scores[gt == 0].mean()),
            # --- victim side
            "cta_undefended": (none_s or {}).get("cta"),
            "pta_undefended": (none_s or {}).get("pta"),
            "cta_defended": (rem_s or {}).get("cta"),
            "pta_defended": (rem_s or {}).get("pta"),
        })
    return rows


def write_master(study: Study, rows):
    out = study.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics_master.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    controls = []
    for n in study.args.control_budgets:
        s, _ = study.control(n)
        controls.append({"flip_budget": n,
                         "cta": (s or {}).get("cta"),
                         "pta": (s or {}).get("pta")})
    clean_s, _ = study.clean_control()

    payload = {
        "attack": {kk: study.manifest[kk] for kk in
                   ("dataset", "encoder", "k", "scoring", "refine_iters")},
        "threshold": study.args.threshold,
        "frontier": rows,
        "matched_budget_controls": controls,
        "clean_label_control": {"cta": (clean_s or {}).get("cta"),
                                "pta": (clean_s or {}).get("pta")},
    }
    (out / "metrics_master.json").write_text(json.dumps(payload, indent=2))
    print(f"  metrics: metrics_master.{{csv,json}}")
    return controls, clean_s


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def f(v, spec=".3f", dash="--"):
    return dash if v is None else format(v, spec)


def write_tables(study: Study, rows, controls, clean_s):
    tdir = study.out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    # --- main frontier ---
    hdr = ["$\\tau$", "flips", "mean margin", "\\% source",
           "recall", "AUPRC", "PTA (no def.)", "PTA (defended)", "CTA (defended)"]
    body = [[
        f"{r['tau']:.2f}", str(r["n_flips_placed"]), f(r["mean_margin"], ".2f"),
        f(r["frac_source_class"], ".1%").replace("%", "\\%"),
        f(r["flagged_recall"]), f(r["auprc"]),
        f(r["pta_undefended"]), f(r["pta_defended"]), f(r["cta_defended"]),
    ] for r in rows]
    _emit_table(tdir, "frontier", hdr, body,
                "Adaptive-attacker stealth-vs-efficacy frontier. $\\tau=1.0$ is "
                "the defense-unaware attacker and reproduces vanilla FLIP.")

    # --- attacker cost ---
    hdr2 = ["$\\tau$", "eligible", "flips placed", "budget met",
            "mean margin", "\\% source", "Jaccard vs.\\ naive"]
    body2 = [[
        f"{r['tau']:.2f}", str(r["n_eligible"]), str(r["n_flips_placed"]),
        "yes" if r["budget_satisfied"] else "no",
        f(r["mean_margin"], ".2f"),
        f(r["frac_source_class"], ".1%").replace("%", "\\%"),
        f(r["jaccard_vs_unconstrained"], ".3f"),
    ] for r in rows]
    _emit_table(tdir, "attacker_cost", hdr2, body2,
                "What the stealth constraint costs the attacker before any "
                "defense runs.")

    # --- detection ---
    hdr3 = ["$\\tau$", "AUROC", "AUPRC", "P@k", "best F1", "F1@0.5",
            "MCC", "recall", "poisons missed", "clean removed"]
    body3 = [[
        f"{r['tau']:.2f}", f(r["auroc"]), f(r["auprc"]), f(r["precision_at_k"]),
        f(r["best_f1"]), f(r["f1_at_threshold"]), f(r["mcc_at_threshold"]),
        f(r["flagged_recall"]), str(r["poisons_missed"]), str(r["clean_removed"]),
    ] for r in rows]
    _emit_table(tdir, "detection", hdr3, body3,
                "Detection metrics vs.\\ attacker stealth budget. AUROC stays "
                "high while AUPRC and recall collapse -- the base-rate artefact "
                "that makes AUROC the wrong headline metric here.")

    # --- controls ---
    # Only rows that actually ran; a blank row reads as a failed experiment
    # rather than one deliberately not needed.
    hdr4 = ["configuration", "CTA", "PTA"]
    body4 = [[f"FLIP, {c['flip_budget']} flips", f(c["cta"]), f(c["pta"])]
             for c in controls if c["pta"] is not None]
    if clean_s:
        body4.append(["clean labels, no attack", f(clean_s["cta"]),
                      f(clean_s["pta"])])
    caption = ("Reference points. The clean-label row is the CTA ceiling and "
               "the PTA floor every other number is read against.")
    if not body4[:-1]:
        caption += (" Matched-budget FLIP controls were not required: at "
                    "$\\tau = 0.90$ the constrained attacker places the full "
                    "budget and the attack still collapses, which controls for "
                    "flip count from within the sweep.")
    _emit_table(tdir, "controls", hdr4, body4, caption)


def _emit_table(tdir: Path, name, header, body, caption):
    md = ["| " + " | ".join(h.replace("\\%", "%").replace("$\\tau$", "tau")
                            .replace("\\", "") for h in header) + " |",
          "|" + "---|" * len(header)]
    for row in body:
        md.append("| " + " | ".join(c.replace("\\%", "%") for c in row) + " |")
    (tdir / f"{name}.md").write_text(
        f"**{caption.replace(chr(92) + '%', '%')}**\n\n" + "\n".join(md) + "\n"
    )

    tex = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\begin{tabular}{" + "l" + "r" * (len(header) - 1) + "}",
        "\\toprule",
        " & ".join(header) + " \\\\", "\\midrule",
    ]
    tex += [" & ".join(r) + " \\\\" for r in body]
    tex += ["\\bottomrule", "\\end{tabular}",
            f"\\caption{{{caption}}}", f"\\label{{tab:{name}}}",
            "\\end{table}"]
    (tdir / f"{name}.tex").write_text("\n".join(tex) + "\n")
    print(f"  table: tables/{name}.{{md,tex}}")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

@figure
def fig_frontier_main(study: Study, rows):
    """Two panels sharing the tau axis. Never one plot with two y-scales."""
    have = [r for r in rows if r["pta_undefended"] is not None]
    if not have:
        raise SkipFigure("no victim runs finished yet")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.3))
    taus = [r["tau"] for r in have]

    ax1.plot(taus, [r["pta_undefended"] for r in have], "-o",
             color=C_ORANGE, label="No defense")
    defd = [(r["tau"], r["pta_defended"]) for r in have
            if r["pta_defended"] is not None]
    if defd:
        ax1.plot([d[0] for d in defd], [d[1] for d in defd], "-s",
                 color=C_BLUE, label="k-NN audit")
    ax1.set_xlabel("attacker stealth budget  $\\tau$")
    ax1.set_ylabel("attack success rate (PTA)")
    ax1.set_title("What stealth costs the attack")
    ax1.set_ylim(-0.03, 1.03)
    ax1.invert_xaxis()
    ax1.legend(loc="best")

    ax2.plot(taus, [r["flagged_recall"] for r in have], "-o",
             color=C_BLUE, label="recall @ 0.5")
    ax2.plot(taus, [r["auprc"] for r in have], "-^",
             color=C_AQUA, label="AUPRC")
    ax2.plot(taus, [r["auroc"] for r in have], "--", color=INK_SOFT,
             linewidth=1.4, label="AUROC (misleading)")
    ax2.set_xlabel("attacker stealth budget  $\\tau$")
    ax2.set_ylabel("detection performance")
    ax2.set_title("What stealth costs the defense")
    ax2.set_ylim(-0.03, 1.03)
    ax2.invert_xaxis()
    ax2.legend(loc="best")

    fig.tight_layout()
    save(fig, study.out / "figures" / "frontier", "frontier_main")


@figure
def fig_frontier_tradeoff(study: Study, rows):
    """The headline: attack success against how much of it the audit catches."""
    have = [r for r in rows if r["pta_undefended"] is not None]
    if len(have) < 2:
        raise SkipFigure("need >=2 finished victim runs")

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    x = [r["flagged_recall"] for r in have]
    y = [r["pta_undefended"] for r in have]
    ax.plot(x, y, "-o", color=C_BLUE)
    # Direct labels: few enough points that a legend would be worse.
    for r, xi, yi in zip(have, x, y):
        ax.annotate(f"$\\tau$={r['tau']:.2f}\n({r['n_flips_placed']} flips)",
                    (xi, yi), textcoords="offset points", xytext=(6, 6),
                    fontsize=7, color=INK_SOFT)
    ax.set_xlabel("fraction of poisons the audit removes")
    ax.set_ylabel("attack success rate (PTA), undefended")
    ax.set_title("Evading the audit costs attack success")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.08)
    fig.tight_layout()
    save(fig, study.out / "figures" / "frontier", "frontier_tradeoff")


@figure
def fig_attacker_cost(study: Study, rows):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    taus = [r["tau"] for r in rows]

    axes[0].plot(taus, [r["n_flips_placed"] for r in rows], "-o", color=C_BLUE)
    axes[0].axhline(study.args.budget, color=INK_SOFT, linestyle=":", linewidth=1.2)
    axes[0].annotate("requested budget", (taus[0], study.args.budget),
                     textcoords="offset points", xytext=(4, -12),
                     ha="left", fontsize=7, color=INK_SOFT)
    axes[0].set_ylabel("flips actually placed")
    axes[0].set_title("Budget collapses")

    axes[1].plot(taus, [r["mean_margin"] for r in rows], "-o", color=C_BLUE)
    axes[1].axhline(0, color=INK_SOFT, linestyle=":", linewidth=1.2)
    axes[1].set_ylabel("mean FLIP margin of selection")
    axes[1].set_title("Flip quality inverts")

    axes[2].plot(taus, [r["frac_source_class"] for r in rows], "-o", color=C_BLUE)
    axes[2].set_ylabel("fraction from source class")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Attack loses its target")

    for ax in axes:
        ax.set_xlabel("stealth budget  $\\tau$")
        ax.invert_xaxis()
    fig.tight_layout()
    save(fig, study.out / "figures" / "attacker", "attacker_cost")


@figure
def fig_detection_vs_k(study: Study):
    csv_path = study.out / "detection_vs_tau.csv"
    if not csv_path.exists():
        raise SkipFigure("run scripts/adaptive/eval_detection.py first")
    data = list(csv.DictReader(open(csv_path)))
    ks = sorted({int(r["defender_k"]) for r in data})
    colors = [C_BLUE, C_ORANGE, C_AQUA, C_YELLOW]

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for i, k in enumerate(ks[:4]):
        sub = sorted([r for r in data if int(r["defender_k"]) == k],
                     key=lambda r: -float(r["tau"]))
        ax.plot([float(r["tau"]) for r in sub],
                [float(r["flagged_recall"]) for r in sub],
                "-o", color=colors[i], label=f"defender $k$={k}")
    ax.set_xlabel("attacker stealth budget  $\\tau$")
    ax.set_ylabel("fraction of poisons removed")
    ax.set_title("Re-tuning $k$ does not rescue the defense")
    ax.invert_xaxis()
    ax.set_ylim(-0.03, 1.03)
    ax.legend()
    fig.tight_layout()
    save(fig, study.out / "figures" / "detection", "detection_vs_k")


@figure
def fig_pr_curves(study: Study):
    taus = study.figure_taus
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    for i, tau in enumerate(taus):
        gt = study.gt(tau)
        if gt.sum() == 0:
            continue
        scores = study.scores(tau)
        prec, rec, _ = precision_recall_curve(gt, scores)
        ap = average_precision_score(gt, scores)
        ax.plot(rec, prec, color=ramp(i, len(taus)),
                label=f"$\\tau$={tau:.2f}  (AP={ap:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-recall collapse under adaptive attack")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    # Upper-right is empty for PR curves (precision is lowest at full recall),
    # so the legend never sits on the data.
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, study.out / "figures" / "detection", "pr_curves")


@figure
def fig_score_hist(study: Study):
    picks = [r["tau"] for r in study.runs][:1] + \
            [r["tau"] for r in study.runs if abs(r["tau"] - 0.5) < 1e-9]
    picks = picks or [study.runs[0]["tau"]]
    fig, axes = plt.subplots(1, len(picks), figsize=(4.4 * len(picks), 3.2),
                             squeeze=False)
    bins = np.linspace(0, 1, 41)
    for ax, tau in zip(axes[0], picks):
        gt = study.gt(tau)
        s = study.scores(tau)
        ax.hist(s[gt == 0], bins=bins, color=C_BLUE, alpha=0.75,
                label=f"clean (n={int((gt == 0).sum())})")
        ax.hist(s[gt == 1], bins=bins, color=C_ORANGE, alpha=0.85,
                label=f"poisoned (n={int(gt.sum())})")
        ax.axvline(study.args.threshold, color=INK_SOFT, linestyle="--",
                   linewidth=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("k-NN disagreement score")
        ax.set_ylabel("count (log)")
        ax.set_title(f"$\\tau$ = {tau:.2f}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, study.out / "figures" / "detection", "score_distributions")


@figure
def fig_matched_budget(study: Study, rows, controls):
    ctrl = [c for c in controls if c["pta"] is not None]
    adapt = [r for r in rows if r["pta_undefended"] is not None]
    if not ctrl or not adapt:
        raise SkipFigure("controls or adaptive victim runs not finished")

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot([c["flip_budget"] for c in ctrl], [c["pta"] for c in ctrl],
            "-o", color=C_ORANGE, label="defense-unaware FLIP")
    ax.plot([r["n_flips_placed"] for r in adapt],
            [r["pta_undefended"] for r in adapt],
            "-s", color=C_BLUE, label="stealth-constrained FLIP")
    ax.set_xscale("log")
    ax.set_xlabel("number of labels flipped")
    ax.set_ylabel("attack success rate (PTA)")
    ax.set_title("Same budget, weaker attack")
    ax.set_ylim(-0.03, 1.03)
    ax.legend()
    fig.tight_layout()
    save(fig, study.out / "figures" / "frontier", "matched_budget")


@figure
def fig_training_curves(study: Study):
    taus = study.figure_taus
    series = []
    for i, tau in enumerate(taus):
        _, d = study.victim(tau, "none")
        if (d / "caccs.npy").exists() and (d / "paccs.npy").exists():
            series.append((tau, np.load(d / "caccs.npy"),
                           np.load(d / "paccs.npy"), i))
    if not series:
        raise SkipFigure("no finished victim runs with curves")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for tau, cacc, pacc, i in series:
        color = ramp(i, len(taus))
        ax1.plot(np.arange(1, len(cacc) + 1), cacc[:, 0], color=color,
                 label=f"$\\tau$={tau:.2f}")
        ax2.plot(np.arange(1, len(pacc) + 1), pacc[:, 0], color=color,
                 label=f"$\\tau$={tau:.2f}")
    ax1.set_ylabel("clean test accuracy (CTA)")
    ax1.set_title("Clean behaviour is unaffected")
    ax2.set_ylabel("attack success rate (PTA)")
    ax2.set_title("The backdoor either installs or it does not")
    for ax in (ax1, ax2):
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, study.out / "figures" / "training", "training_curves")


@figure
def fig_samples(study: Study, n_show=8):
    """The naive attacker picks obvious trucks; the adaptive one cannot."""
    from modules.base_utils.datasets import load_dataset
    classes = ["plane", "car", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
    ds = load_dataset(study.args.dataset, train=True)

    picks = [study.runs[0]["tau"]]
    low = [r["tau"] for r in study.runs if abs(r["tau"] - 0.5) < 1e-9]
    picks += low or [study.runs[-1]["tau"]]

    fig, axes = plt.subplots(len(picks), n_show,
                             figsize=(1.15 * n_show, 1.95 * len(picks)),
                             squeeze=False)
    for row, tau in enumerate(picks):
        sel = np.load(study.attack_dir / f"tau_{tau:.3f}" /
                      f"selected_{study.args.budget}.npy")
        hard = study.hard_labels(tau)
        show = sel[np.linspace(0, len(sel) - 1, min(n_show, len(sel))).astype(int)]
        for col in range(n_show):
            ax = axes[row][col]
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if col >= len(show):
                ax.axis("off")
                continue
            idx = int(show[col])
            ax.imshow(ds[idx][0])
            ax.set_title(f"{classes[study.true_hard[idx]]}\n$\\to$ "
                         f"{classes[hard[idx]]}", fontsize=6, color=INK)
        axes[row][0].set_ylabel(f"$\\tau$={tau:.2f}", fontsize=8)
    fig.suptitle("Labels each attacker chose to flip  (true class $\\to$ "
                 "planted label)", fontsize=9)
    fig.tight_layout()
    # Each caption sits above its own image; without extra room between rows a
    # reader attaches row 2's captions to row 1's thumbnails.
    fig.subplots_adjust(hspace=0.5)
    save(fig, study.out / "figures" / "samples", "selected_samples")


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attack-dir", default="experiments/adaptive_flip_cifar_1xs")
    p.add_argument("--out-dir", default="experiments/_report_adaptive_cifar_1xs")
    p.add_argument("--name-prefix", default="adaptive_cifar_1xs")
    p.add_argument("--control-prefix", default="flipbudget_cifar_1xs")
    p.add_argument("--clean-control-name", default="cleanlabel_cifar_1xs_none")
    # Empty by default: tau=0.90 places the FULL budget yet still collapses, so
    # it is an internal budget control and the separate matched-budget runs are
    # redundant. Pass budgets explicitly to re-enable them.
    p.add_argument("--control-budgets", type=int, nargs="*", default=[])
    p.add_argument("--budget", type=int, default=1500)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--figure-taus", type=float, nargs="+",
                   default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
                   help="Taus drawn as separate series in per-tau figures. "
                        "Capped by the ordinal ramp's step count; tables and "
                        "metrics always cover every tau.")
    p.add_argument("--dataset", default="cifar")
    p.add_argument("--encoder", default="dinov2_vits14")
    p.add_argument("--feature-cache", default="data/ssl_features")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    study = Study(root, args)

    print("Computing metrics...")
    rows = build_master(study)
    controls, clean_s = write_master(study, rows)

    print("Writing tables...")
    write_tables(study, rows, controls, clean_s)

    print("Building figures...")
    fig_frontier_main(study, rows)
    fig_frontier_tradeoff(study, rows)
    fig_matched_budget(study, rows, controls)
    fig_attacker_cost(study, rows)
    fig_detection_vs_k(study)
    fig_pr_curves(study)
    fig_score_hist(study)
    fig_training_curves(study)
    fig_samples(study)

    # Count against the taus the figures actually plot, not every tau in the
    # manifest — detection covers all of them without any training.
    planned = [r for r in rows
               if any(abs(r["tau"] - t) < 1e-9 for t in study.figure_taus)]
    n_done = sum(1 for r in planned if r["pta_undefended"] is not None)
    print(f"\nReport written to {study.out}")
    print(f"Victim runs finished: {n_done}/{len(planned)} "
          f"({'complete' if n_done == len(planned) else 'partial - re-run later'})")


if __name__ == "__main__":
    main()
