r"""
Generate thesis-ready tables from the aggregated metrics.

Reads <report_dir>/master_metrics.csv and emits:

    <report_dir>/tables/table_headline.md
    <report_dir>/tables/table_headline.tex
    <report_dir>/tables/table_detection_sweep.md
    <report_dir>/tables/table_detection_sweep.tex
    <report_dir>/tables/table_training_sweep.md
    <report_dir>/tables/table_training_sweep.tex
    <report_dir>/tables/table_best_thresholds.md
    <report_dir>/tables/table_best_thresholds.tex
    <report_dir>/tables/table_per_class.md          (if per_class_detection.csv exists)

The LaTeX outputs use the booktabs style and are ready to \input{} into the
thesis. The Markdown variants are convenient for the project README / for
PowerPoint reports.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _fmt(x, digits=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):d}"
    return f"{x:.{digits}f}"


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"[make_tables] wrote {path}")


def write_headline(master: pd.DataFrame, out_dir: Path):
    """Compact headline: k × (mode pair) × {CTA, PTA, AUROC, best F1, flagged
    P, flagged R}.
    """
    cols_md = [
        "k", "AUROC", "AUPRC",
        "CTA none", "CTA remove",
        "PTA none", "PTA remove",
        "flag P", "flag R",
        "best F1", "best MCC",
    ]
    rows_md = []
    for k in sorted(master["k"].unique()):
        sub = master[master["k"] == k]
        row_n = sub[sub["mode"] == "none"]
        row_r = sub[sub["mode"] == "remove"]
        if row_n.empty or row_r.empty:
            continue
        rn = row_n.iloc[0]
        rr = row_r.iloc[0]
        rows_md.append([
            int(k),
            _fmt(rr.get("auroc")),
            _fmt(rr.get("auprc")),
            _fmt(rn.get("cta")), _fmt(rr.get("cta")),
            _fmt(rn.get("pta")), _fmt(rr.get("pta")),
            _fmt(rr.get("flagged_precision")),
            _fmt(rr.get("flagged_recall")),
            _fmt(rr.get("best_f1")),
            _fmt(rr.get("best_mcc")),
        ])

    # Markdown
    md = ["| " + " | ".join(cols_md) + " |",
          "|" + "|".join(["---"] * len(cols_md)) + "|"]
    for r in rows_md:
        md.append("| " + " | ".join(str(v) for v in r) + " |")
    _write(out_dir / "table_headline.md",
            "## Headline: Deep k-NN defense across k\n\n"
            "AUROC / AUPRC measured on the full training set scores; flag P / R\n"
            "evaluated at the configured auto threshold (default 0.5).\n\n"
            + "\n".join(md) + "\n")

    # LaTeX
    headers = ["$k$", "AUROC", "AUPRC",
                "CTA$_\\text{none}$", "CTA$_\\text{remove}$",
                "PTA$_\\text{none}$", "PTA$_\\text{remove}$",
                "flag P", "flag R", "best F1", "best MCC"]
    lines = [
        r"\begin{tabular}{r" + "c" * (len(headers) - 1) + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for r in rows_md:
        lines.append(" & ".join(str(v) for v in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out_dir / "table_headline.tex", "\n".join(lines) + "\n")


def write_detection_sweep(master: pd.DataFrame, out_dir: Path):
    """Detection-only — one row per (k, mode), all detection metrics."""
    cols = ["mode", "k", "n_true_poisoned", "n_flagged", "n_removed",
             "auroc", "auprc", "precision_at_k", "recall_at_k",
             "flagged_precision", "flagged_recall",
             "best_f1", "best_f1_threshold",
             "best_mcc", "best_mcc_threshold",
             "f1_at_thr", "mcc_at_thr",
             "balanced_acc_at_thr",
             "score_mean_poisoned", "score_mean_clean"]
    cols = [c for c in cols if c in master.columns]
    sub = master[cols].sort_values(["mode", "k"]).reset_index(drop=True)

    # Markdown
    md_header = "| " + " | ".join(cols) + " |"
    md_sep    = "|" + "|".join(["---"] * len(cols)) + "|"
    md_lines = [md_header, md_sep]
    for _, r in sub.iterrows():
        md_lines.append(
            "| " + " | ".join(_fmt(r[c]) if isinstance(r[c], float)
                                else str(r[c]) for c in cols) + " |"
        )
    _write(out_dir / "table_detection_sweep.md",
            "## Detection metrics — full k sweep\n\n" + "\n".join(md_lines) + "\n")

    # LaTeX
    headers = [c.replace("_", " ") for c in cols]
    lines = [
        r"\begin{tabular}{l" + "c" * (len(headers) - 1) + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for _, r in sub.iterrows():
        cells = [_fmt(r[c]) if isinstance(r[c], (float, np.floating))
                  else str(r[c]) for c in cols]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out_dir / "table_detection_sweep.tex", "\n".join(lines) + "\n")


def write_training_sweep(master: pd.DataFrame, out_dir: Path):
    cols = ["mode", "k", "n_train_samples", "n_removed",
             "cta", "pta"]
    cols = [c for c in cols if c in master.columns]
    sub = master[cols].sort_values(["mode", "k"]).reset_index(drop=True)

    md_lines = ["| " + " | ".join(cols) + " |",
                 "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in sub.iterrows():
        md_lines.append("| " + " | ".join(
            _fmt(r[c]) if isinstance(r[c], (float, np.floating)) else str(r[c])
            for c in cols) + " |")
    _write(out_dir / "table_training_sweep.md",
            "## Training-time metrics — full k sweep\n\n" + "\n".join(md_lines) + "\n")

    lines = [
        r"\begin{tabular}{l" + "c" * (len(cols) - 1) + "}",
        r"\toprule",
        " & ".join([c.replace("_", " ") for c in cols]) + r" \\",
        r"\midrule",
    ]
    for _, r in sub.iterrows():
        cells = [_fmt(r[c]) if isinstance(r[c], (float, np.floating))
                  else str(r[c]) for c in cols]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out_dir / "table_training_sweep.tex", "\n".join(lines) + "\n")


def write_best_thresholds(master: pd.DataFrame, out_dir: Path):
    cols = ["k", "best_f1", "best_f1_threshold", "best_mcc", "best_mcc_threshold"]
    cols = [c for c in cols if c in master.columns]
    sub = master[master["mode"] == "remove"][cols] \
            .sort_values("k").reset_index(drop=True)

    md_lines = ["| " + " | ".join(cols) + " |",
                 "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in sub.iterrows():
        md_lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    _write(out_dir / "table_best_thresholds.md",
            "## Best operating thresholds per k\n\n"
            "F1 and MCC peak at different thresholds; this table lets the user\n"
            "tune the runtime threshold for their accuracy / coverage preference.\n\n"
            + "\n".join(md_lines) + "\n")

    lines = [
        r"\begin{tabular}{r" + "c" * (len(cols) - 1) + "}",
        r"\toprule",
        " & ".join([c.replace("_", " ") for c in cols]) + r" \\",
        r"\midrule",
    ]
    for _, r in sub.iterrows():
        lines.append(" & ".join(_fmt(r[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out_dir / "table_best_thresholds.tex", "\n".join(lines) + "\n")


def write_per_class(per_class: pd.DataFrame, out_dir: Path, mode: str = "remove"):
    sub = per_class[per_class["mode"] == mode].copy()
    if sub.empty:
        return
    pivot = sub.pivot_table(index="assigned_class", columns="k",
                              values=["flag_rate", "n_true_poisoned", "n_flagged"],
                              aggfunc="first").sort_index()
    pivot = pivot.swaplevel(0, 1, axis=1).sort_index(axis=1)

    md_lines = ["## Per-class detection (mode = " + mode + ")\n",
                 "Columns are repeated triples (n_true_poisoned, n_flagged, flag_rate) per k.\n"]
    head = ["class"]
    for k in sorted(set(c[0] for c in pivot.columns)):
        head += [f"k={k}: P", f"k={k}: F", f"k={k}: rate"]
    md_lines.append("| " + " | ".join(head) + " |")
    md_lines.append("|" + "|".join(["---"] * len(head)) + "|")
    for cls_idx in pivot.index:
        row = [CIFAR10_CLASSES[cls_idx] if cls_idx < len(CIFAR10_CLASSES)
                else str(cls_idx)]
        for k in sorted(set(c[0] for c in pivot.columns)):
            for metric in ["n_true_poisoned", "n_flagged", "flag_rate"]:
                val = pivot.get((k, metric), pd.Series([np.nan])).iloc[
                    list(pivot.index).index(cls_idx)
                ] if (k, metric) in pivot.columns else None
                row.append(_fmt(val))
        md_lines.append("| " + " | ".join(row) + " |")
    _write(out_dir / f"table_per_class_{mode}.md", "\n".join(md_lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name-prefix", required=True)
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    project_root = Path(args.project_root) if args.project_root else \
        Path(__file__).resolve().parents[2]
    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = project_root / report_dir
    out_dir = report_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(report_dir / "master_metrics.csv")

    write_headline(master, out_dir)
    write_detection_sweep(master, out_dir)
    write_training_sweep(master, out_dir)
    write_best_thresholds(master, out_dir)

    per_class_p = report_dir / "per_class_detection.csv"
    if per_class_p.exists():
        per_class = pd.read_csv(per_class_p)
        write_per_class(per_class, out_dir, mode="remove")
        write_per_class(per_class, out_dir, mode="none")

    print(f"[make_tables] tables in {out_dir}")


if __name__ == "__main__":
    main()
