"""
Join attacker-side and victim-side results into the stealth-vs-efficacy frontier.

Reads:
    experiments/adaptive_flip_cifar_1xs/manifest.json     attacker diagnostics
    experiments/adaptive_cifar_1xs_tau*_{none,remove}/summary_detailed.json
    experiments/flipbudget_cifar_1xs_n*_none/summary_detailed.json

Writes (under experiments/_report_adaptive_cifar_1xs/):
    frontier.csv        one row per tau, attacker + undefended + defended
    controls.csv        matched-budget vanilla-FLIP controls
    frontier.md         thesis-ready table

The headline reading: PTA_undefended is what the stealth constraint costs the
attacker before any defense runs at all; PTA_defended is what survives the
audit. An attacker only wins where PTA_defended stays high.

Usage:
    python scripts/adaptive/aggregate_frontier.py
"""

import argparse
import csv
import json
from pathlib import Path


def load_summary(root: Path, name: str):
    path = root / "experiments" / name / "summary_detailed.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def tau_slug(tau: float) -> str:
    return f"{tau:.3f}".replace(".", "p")


def fmt(v, spec=".4f", missing="--"):
    return missing if v is None else format(v, spec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attack-dir", default="experiments/adaptive_flip_cifar_1xs")
    p.add_argument("--name-prefix", default="adaptive_cifar_1xs")
    p.add_argument("--control-prefix", default="flipbudget_cifar_1xs")
    p.add_argument("--control-budgets", type=int, nargs="+",
                   default=[150, 300, 500, 1000])
    p.add_argument("--budget", type=int, default=1500)
    p.add_argument("--out-dir", default="experiments/_report_adaptive_cifar_1xs")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / args.attack_dir / "manifest.json").read_text())
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in manifest["runs"]:
        if r["budget"] != args.budget:
            continue
        tau = r["tau"]
        none_s = load_summary(root, f"{args.name_prefix}_tau{tau_slug(tau)}_none")
        rem_s = load_summary(root, f"{args.name_prefix}_tau{tau_slug(tau)}_remove")
        rem_det = (rem_s or {}).get("detection_metrics", {})

        rows.append({
            "tau": tau,
            "n_flips_placed": r["n_selected"],
            "budget_satisfied": r["budget_satisfied"],
            "n_eligible": r["n_eligible"],
            "mean_margin": r["mean_margin"],
            "baseline_mean_margin": r["baseline_mean_margin"],
            "jaccard_vs_unconstrained": r["jaccard_vs_unconstrained"],
            "frac_selected_from_source_class":
                r["frac_selected_from_source_class"],
            "mean_defender_score_on_poisons":
                r["mean_defender_score_on_poisons"],
            "frac_poisons_above_half": r["frac_poisons_above_half"],
            "cta_undefended": (none_s or {}).get("cta"),
            "pta_undefended": (none_s or {}).get("pta"),
            "cta_defended": (rem_s or {}).get("cta"),
            "pta_defended": (rem_s or {}).get("pta"),
            "auroc": rem_det.get("auroc"),
            "auprc": rem_det.get("auprc"),
            "flagged_precision": rem_det.get("flagged_precision"),
            "flagged_recall": rem_det.get("flagged_recall"),
            "n_removed": (rem_s or {}).get("n_removed"),
        })

    rows.sort(key=lambda x: -x["tau"])
    frontier_csv = out_dir / "frontier.csv"
    with open(frontier_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    control_rows = []
    for n in args.control_budgets:
        s = load_summary(root, f"{args.control_prefix}_n{n}_none")
        control_rows.append({
            "flip_budget": n,
            "cta_undefended": (s or {}).get("cta"),
            "pta_undefended": (s or {}).get("pta"),
        })
    controls_csv = out_dir / "controls.csv"
    with open(controls_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(control_rows[0].keys()))
        w.writeheader()
        w.writerows(control_rows)

    lines = [
        "# Adaptive-attacker stealth-vs-efficacy frontier",
        "",
        f"Attack: FLIP on {manifest['dataset']}, encoder {manifest['encoder']}, "
        f"attacker assumes k={manifest['k']}, budget {args.budget}.",
        "",
        "`tau` is the attacker's stealth budget: a flip is admissible only if "
        "the defender's k-NN disagreement score for it is at most tau. "
        "tau = 1.0 is the defense-unaware attacker and reproduces vanilla FLIP.",
        "",
        "## Frontier",
        "",
        "| tau | flips placed | mean margin | % from source class | mean score "
        "on poisons | PTA undefended | PTA defended | CTA defended | AUPRC |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['tau']:.2f} | {r['n_flips_placed']} | "
            f"{fmt(r['mean_margin'], '.2f')} | "
            f"{fmt(r['frac_selected_from_source_class'], '.1%')} | "
            f"{fmt(r['mean_defender_score_on_poisons'], '.3f')} | "
            f"{fmt(r['pta_undefended'], '.3f')} | "
            f"{fmt(r['pta_defended'], '.3f')} | "
            f"{fmt(r['cta_defended'], '.3f')} | "
            f"{fmt(r['auprc'], '.3f')} |"
        )

    lines += [
        "",
        "## Matched-budget controls (defense-unaware FLIP, no defense)",
        "",
        "Isolates flip COUNT from flip QUALITY: compare each frontier row "
        "against the control at the nearest budget.",
        "",
        "| flip budget | CTA | PTA |",
        "|---|---|---|",
    ]
    for r in control_rows:
        lines.append(
            f"| {r['flip_budget']} | {fmt(r['cta_undefended'], '.3f')} | "
            f"{fmt(r['pta_undefended'], '.3f')} |"
        )

    (out_dir / "frontier.md").write_text("\n".join(lines) + "\n")

    print(f"wrote {frontier_csv}")
    print(f"wrote {controls_csv}")
    print(f"wrote {out_dir / 'frontier.md'}")
    print()
    print("\n".join(lines[10:]))


if __name__ == "__main__":
    main()
