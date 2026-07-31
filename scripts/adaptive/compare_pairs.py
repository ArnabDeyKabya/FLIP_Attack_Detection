"""
The headline comparison: the same defense on a defensible pair and a vulnerable one.

Both studies run the identical attack, defense, encoder, model and budget. The
only difference is which (source -> target) pair is attacked, and the pair
analysis predicted the outcome before either victim was trained:

    truck -> deer   0 stealthy source-class images  -> attacker has no move
    dog   -> cat  265 stealthy source-class images  -> attacker has room

Putting the two frontiers on one axis is what turns a single result into a
claim about the defense: its strength is not a property of the method but of
the encoder's neighbourhood structure around the attacked pair, and that
structure is measurable in advance.

Each pair carries its own clean-label floor, drawn as a reference line. The
floors differ by 60x (0.001 vs 0.061) because a clean model already confuses
dogs with cats and never confuses trucks with deer, so raw PTA is not
comparable across pairs -- only lift over the floor is.

Usage:
    python scripts/adaptive/compare_pairs.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# The tables carry arrows. The Windows console defaults to cp1252, which cannot
# encode them, so echoing a table would abort a run whose files already wrote
# fine. Degrade unencodable glyphs instead of failing.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Two series -> the first two categorical slots, in fixed order.
C_BLUE, C_ORANGE = "#2a78d6", "#eb6834"
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#dcdcda"

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10,
    "axes.labelcolor": INK, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 6,
})


def write_text(path: Path, text: str):
    """Always UTF-8: these tables contain arrows, and Windows would otherwise
    write them in cp1252 and fail."""
    path.write_text(text, encoding="utf-8")


def load(root: Path, report_dir: str):
    path = root / report_dir / "metrics_master.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run make_report.py for that study")
    return json.loads(path.read_text())


def series(payload, taus, key):
    by_tau = {round(r["tau"], 3): r for r in payload["frontier"]}
    out = []
    for t in taus:
        r = by_tau.get(round(t, 3))
        out.append(None if r is None else r.get(key))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report-a", default="experiments/_report_adaptive_cifar_1xs")
    p.add_argument("--label-a", default="truck → deer  (pool 0)")
    p.add_argument("--report-b", default="experiments/_report_adaptive_cifar_dogcat")
    p.add_argument("--label-b", default="dog → cat  (pool 265)")
    p.add_argument("--taus", type=float, nargs="+",
                   default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    p.add_argument("--out-dir", default="experiments/_report_pair_comparison")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    A, B = load(root, args.report_a), load(root, args.report_b)
    taus = args.taus
    floor_a = A["clean_label_control"]["pta"]
    floor_b = B["clean_label_control"]["pta"]

    und_a, und_b = series(A, taus, "pta_undefended"), series(B, taus, "pta_undefended")
    def_a, def_b = series(A, taus, "pta_defended"), series(B, taus, "pta_defended")
    rec_a, rec_b = series(A, taus, "flagged_recall"), series(B, taus, "flagged_recall")

    # ---------- figure ----------
    # One panel per pair rather than all four PTA curves overlaid: with the
    # floors drawn too that would be six lines on one axes. Juxtaposing panels
    # on a shared y-scale compares the pairs just as directly and keeps each
    # panel to two series. Colour encodes the condition (undefended vs
    # defended), consistent across panels, so the reader learns it once.
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4))

    for ax, und, defd, floor, title in (
        (axes[0], und_a, def_a, floor_a, args.label_a),
        (axes[1], und_b, def_b, floor_b, args.label_b),
    ):
        ax.plot(taus, und, "-o", color=C_ORANGE, label="no defense")
        ax.plot(taus, defd, "-s", color=C_BLUE, label="k-NN audit")
        ax.axhline(floor, color=INK_SOFT, linestyle=":", linewidth=1.2)
        ax.annotate(f"clean floor {floor:.3f}", (taus[0], floor),
                    textcoords="offset points", xytext=(4, 5), ha="left",
                    fontsize=7, color=INK_SOFT)
        ax.set_xlabel("attacker stealth budget  $\\tau$")
        ax.set_ylabel("attack success rate (PTA)")
        ax.set_title(title)
        ax.set_ylim(-0.04, 1.04)
        ax.invert_xaxis()
        ax.legend(loc="upper right", fontsize=8)

    axes[2].plot(taus, rec_a, "-o", color=C_BLUE, label=args.label_a)
    axes[2].plot(taus, rec_b, "-s", color=C_ORANGE, label=args.label_b)
    axes[2].set_xlabel("attacker stealth budget  $\\tau$")
    axes[2].set_ylabel("fraction of poisons removed")
    axes[2].set_title("What the audit detects")
    axes[2].set_ylim(-0.04, 1.04)
    axes[2].invert_xaxis()
    axes[2].legend(loc="best", fontsize=8)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"pair_comparison.{ext}")
    plt.close(fig)
    print(f"  figure: pair_comparison.{{png,pdf}}")

    # ---------- table ----------
    def sub(x, y):
        return None if x is None or y is None else x - y

    rows = []
    for i, t in enumerate(taus):
        rows.append({
            "tau": t,
            "pta_undef_truck_deer": und_a[i],
            "pta_defended_truck_deer": def_a[i],
            # What the defense actually buys, and what survives it.
            "benefit_truck_deer": sub(und_a[i], def_a[i]),
            "lift_defended_truck_deer": sub(def_a[i], floor_a),
            "recall_truck_deer": rec_a[i],
            "pta_undef_dog_cat": und_b[i],
            "pta_defended_dog_cat": def_b[i],
            "benefit_dog_cat": sub(und_b[i], def_b[i]),
            "lift_defended_dog_cat": sub(def_b[i], floor_b),
            "recall_dog_cat": rec_b[i],
        })
    with open(out / "pair_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def f(v, s=".3f"):
        return "--" if v is None else format(v, s)

    md = [
        "**Same defense, two pairs.** Identical attack, encoder, model, budget "
        "and defense; only the attacked class pair differs.",
        "",
        "- **benefit** = PTA without the defense minus PTA with it: what the "
        "audit actually buys.",
        "- **lift** = defended PTA minus that pair's clean-label floor "
        f"({floor_a:.3f} for truck→deer, {floor_b:.3f} for dog→cat): what "
        "survives the audit. Raw PTA is not comparable across pairs, because a "
        "clean model already confuses dogs with cats and never confuses trucks "
        "with deer.",
        "",
        "Blank truck→deer defended cells below $\\tau = 1.0$ are deliberate, not "
        "missing data: the undefended attack there is already at the clean-label "
        "floor (PTA $\\leq$ 0.017), and applying a defense to an attack that does "
        "not work measures nothing.",
        "",
        "| tau | t→d undef | t→d def | benefit | lift | recall | "
        "d→c undef | d→c def | benefit | lift | recall |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['tau']:.2f} | {f(r['pta_undef_truck_deer'])} | "
            f"{f(r['pta_defended_truck_deer'])} | {f(r['benefit_truck_deer'])} | "
            f"{f(r['lift_defended_truck_deer'])} | {f(r['recall_truck_deer'])} | "
            f"{f(r['pta_undef_dog_cat'])} | {f(r['pta_defended_dog_cat'])} | "
            f"{f(r['benefit_dog_cat'])} | {f(r['lift_defended_dog_cat'])} | "
            f"{f(r['recall_dog_cat'])} |"
        )
    write_text(out / "pair_comparison.md", "\n".join(md) + "\n")

    tex = ["\\begin{table}[t]", "\\centering", "\\small",
           "\\begin{tabular}{lrrrrrrrrrr}", "\\toprule",
           "& \\multicolumn{5}{c}{truck $\\to$ deer} "
           "& \\multicolumn{5}{c}{dog $\\to$ cat} \\\\",
           "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}",
           "$\\tau$ & undef. & def. & benefit & lift & recall "
           "& undef. & def. & benefit & lift & recall \\\\",
           "\\midrule"]
    for r in rows:
        tex.append(
            f"{r['tau']:.2f} & {f(r['pta_undef_truck_deer'])} & "
            f"{f(r['pta_defended_truck_deer'])} & {f(r['benefit_truck_deer'])} & "
            f"{f(r['lift_defended_truck_deer'])} & {f(r['recall_truck_deer'])} & "
            f"{f(r['pta_undef_dog_cat'])} & {f(r['pta_defended_dog_cat'])} & "
            f"{f(r['benefit_dog_cat'])} & {f(r['lift_defended_dog_cat'])} & "
            f"{f(r['recall_dog_cat'])} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}",
            "\\caption{The defense's strength is a property of the attacked "
            "class pair, not of the method. \\emph{benefit} is the PTA the audit "
            "removes; \\emph{lift} is the PTA that survives it, above that "
            f"pair's clean-label floor ({floor_a:.3f} and {floor_b:.3f}).}}",
            "\\label{tab:pair_comparison}", "\\end{table}"]
    write_text(out / "pair_comparison.tex", "\n".join(tex) + "\n")
    print("  table: pair_comparison.{md,tex}")

    print("\n" + "\n".join(md[2:]))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
