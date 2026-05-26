"""
Aggregate per-(k, mode) experiment outputs into a single master CSV + JSON.

For each experiment <prefix>_<mode>_k<k>/ under experiments/, this reads:
    summary.json
    summary_detailed.json
    detection_metrics.json
    scores.npy
    paccs.npy, caccs.npy
    kept_indices.npy, removed_indices.npy
    labels.npy

and re-computes a broader metric panel (F1, MCC, balanced accuracy, TPR / FPR
sweep stats over a threshold grid, max-F1, max-MCC, per-class detection rate)
that the existing run_module already covers only partially.

Writes:
    <report_dir>/master_metrics.csv
    <report_dir>/master_metrics.json
    <report_dir>/threshold_sweep.csv      (per-k, fine threshold grid)
    <report_dir>/per_class_detection.csv  (per-(k, mode, true_class))
    <report_dir>/training_curves.csv      (per-(k, mode, epoch) CTA + PTA)

The downstream plot_* scripts only read these aggregated artifacts, so plotting
runs are fast even when the underlying experiments contain hundreds of MB.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, balanced_accuracy_score, f1_score, matthews_corrcoef,
)


# ----- helpers ----------------------------------------------------------------

def _safe_load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _safe_load_npy(path: Path):
    if not path.exists():
        return None
    return np.load(path)


def _mcc(tp, fp, fn, tn):
    """Matthews correlation coefficient from a 2x2 confusion matrix."""
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom <= 0:
        return 0.0
    return float((tp * tn - fp * fn) / np.sqrt(denom))


def _f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float(2 * tp / denom)


def _ground_truth_poisoned(labels_assigned: np.ndarray,
                            true_labels: np.ndarray) -> np.ndarray:
    """Reconstruct the binary poison mask used in training."""
    if labels_assigned.ndim == 2:
        labels_assigned = labels_assigned.argmax(axis=1)
    if true_labels.ndim == 2:
        true_labels = true_labels.argmax(axis=1)
    return (labels_assigned != true_labels).astype(np.int32)


def _threshold_sweep(scores: np.ndarray, gt: np.ndarray, n_thresholds: int = 51):
    """Sweep a uniform threshold grid in [0, 1]; return per-threshold metrics."""
    rows = []
    grid = np.linspace(0.0, 1.0, n_thresholds)
    N = len(scores)
    P = int(gt.sum())
    Ngt0 = N - P
    for t in grid:
        flagged = scores > t
        n_flag = int(flagged.sum())
        tp = int((flagged & gt.astype(bool)).sum())
        fp = n_flag - tp
        fn = P - tp
        tn = Ngt0 - fp
        precision = tp / n_flag if n_flag > 0 else None
        recall = tp / P if P > 0 else None
        f1 = _f1(tp, fp, fn) if (tp + fp + fn) > 0 else None
        mcc = _mcc(tp, fp, fn, tn)
        fpr = fp / Ngt0 if Ngt0 > 0 else None
        rows.append({
            "threshold": float(t),
            "n_flagged": n_flag,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "mcc": mcc, "fpr": fpr,
        })
    return rows


def _max_metric_threshold(sweep_rows, metric: str):
    """Return (best_threshold, best_value) for the named metric, ignoring None."""
    best_t, best_v = None, -np.inf
    for r in sweep_rows:
        v = r.get(metric)
        if v is None:
            continue
        if v > best_v:
            best_v, best_t = v, r["threshold"]
    return best_t, (best_v if best_v > -np.inf else None)


# ----- per-experiment extraction ---------------------------------------------

def process_experiment(
    project_root: Path,
    exp_name: str,
    *,
    threshold: float,
):
    exp_dir = project_root / "experiments" / exp_name
    summary = _safe_load_json(exp_dir / "summary.json")
    detailed = _safe_load_json(exp_dir / "summary_detailed.json")
    det = _safe_load_json(exp_dir / "detection_metrics.json")
    if summary is None or detailed is None or det is None:
        print(f"[aggregate] WARN: missing JSONs in {exp_dir}; skipping")
        return None

    scores       = _safe_load_npy(exp_dir / "scores.npy")
    labels_assn  = _safe_load_npy(exp_dir / "labels.npy")
    kept_idx     = _safe_load_npy(exp_dir / "kept_indices.npy")
    removed_idx  = _safe_load_npy(exp_dir / "removed_indices.npy")
    paccs        = _safe_load_npy(exp_dir / "paccs.npy")
    caccs        = _safe_load_npy(exp_dir / "caccs.npy")

    # Reconstruct the ground-truth poison mask
    true_lbl_rel = detailed.get("true_labels") or None
    # detailed doesn't include true_labels path; try the standard FLIP layout.
    poisoner = detailed["poisoner"]
    dataset  = detailed["dataset"]
    user_model = detailed["user_model"]
    true_lbl_path = project_root / "precomputed_labels" / dataset / user_model \
        / poisoner / "true.npy"
    if not true_lbl_path.exists():
        print(f"[aggregate] WARN: cannot find true labels at {true_lbl_path}; "
              f"falling back to summary detection metrics only.")
        gt = None
    else:
        true_labels = np.load(true_lbl_path)
        gt = _ground_truth_poisoned(labels_assn, true_labels)

    # ---- master row -----------------------------------------------------
    row = {
        "experiment": exp_name,
        "mode": detailed["mode"],
        "k": int(detailed["k"]),
        "scoring": detailed["scoring"],
        "encoder": detailed["encoder"],
        "dataset": dataset,
        "user_model": user_model,
        "poisoner": poisoner,
        "source_label": detailed["source_label"],
        "target_label": detailed["target_label"],
        "seed": detailed["seed"],
        "n_total": detailed["n_total"],
        "n_true_poisoned": detailed["n_true_poisoned"],
        "n_removed": detailed["n_removed"],
        "n_train_samples": detailed["n_train_samples"],
        "cta": detailed["cta"],
        "pta": detailed["pta"],
        # Existing detection metrics (from run_module)
        "auroc": det["auroc"],
        "auprc": det["auprc"],
        "precision_at_k": det["precision_at_k"],
        "recall_at_k": det["recall_at_k"],
        "flagged_precision": det["flagged_precision"],
        "flagged_recall": det["flagged_recall"],
    }

    sweep_rows = None
    if scores is not None and gt is not None:
        # ---- Re-compute extended metrics from scores ---------------------
        if int(gt.sum()) > 0:
            row["auroc_recheck"] = float(roc_auc_score(gt, scores))
            row["auprc_recheck"] = float(average_precision_score(gt, scores))

        # Threshold-grid sweep
        sweep_rows = _threshold_sweep(scores, gt)
        # Headline metrics at the configured threshold
        flagged = scores > threshold
        tp = int((flagged & gt.astype(bool)).sum())
        fp = int(flagged.sum()) - tp
        fn = int(gt.sum()) - tp
        tn = len(gt) - tp - fp - fn
        row["tp_at_thr"]  = tp
        row["fp_at_thr"]  = fp
        row["fn_at_thr"]  = fn
        row["tn_at_thr"]  = tn
        row["f1_at_thr"]  = _f1(tp, fp, fn)
        row["mcc_at_thr"] = _mcc(tp, fp, fn, tn)
        row["balanced_acc_at_thr"] = float(balanced_accuracy_score(
            gt, flagged.astype(int)
        )) if gt.sum() > 0 and (gt == 0).sum() > 0 else None

        # Best F1 / MCC achievable across the threshold grid
        t_best_f1, best_f1 = _max_metric_threshold(sweep_rows, "f1")
        t_best_mcc, best_mcc = _max_metric_threshold(sweep_rows, "mcc")
        row["best_f1"] = best_f1
        row["best_f1_threshold"] = t_best_f1
        row["best_mcc"] = best_mcc
        row["best_mcc_threshold"] = t_best_mcc

        # Score distribution stats
        if (gt == 1).any():
            row["score_mean_poisoned"] = float(scores[gt == 1].mean())
            row["score_std_poisoned"]  = float(scores[gt == 1].std())
        if (gt == 0).any():
            row["score_mean_clean"]    = float(scores[gt == 0].mean())
            row["score_std_clean"]     = float(scores[gt == 0].std())

    # ---- Per-class detection rates ---------------------------------------
    per_class_rows = []
    if scores is not None and gt is not None and labels_assn is not None:
        if labels_assn.ndim == 2:
            assigned_hard = labels_assn.argmax(axis=1)
        else:
            assigned_hard = labels_assn.astype(int)
        flagged = (scores > threshold).astype(int)
        for cls in sorted(np.unique(assigned_hard).tolist()):
            mask = assigned_hard == cls
            if not mask.any():
                continue
            cls_gt = gt[mask]
            cls_fl = flagged[mask]
            n = int(mask.sum())
            n_p = int(cls_gt.sum())
            n_fl = int(cls_fl.sum())
            tp = int((cls_fl == 1) & (cls_gt == 1)).sum() if False \
                 else int(((cls_fl == 1) & (cls_gt == 1)).sum())
            fp = n_fl - tp
            fn = n_p - tp
            per_class_rows.append({
                "experiment": exp_name,
                "mode": detailed["mode"],
                "k": int(detailed["k"]),
                "assigned_class": int(cls),
                "n_samples": n,
                "n_true_poisoned": n_p,
                "n_flagged": n_fl,
                "tp": tp, "fp": fp, "fn": fn,
                "precision": tp / n_fl if n_fl > 0 else None,
                "recall": tp / n_p if n_p > 0 else None,
                "flag_rate": n_fl / n if n > 0 else None,
            })

    # ---- Per-epoch training curves ---------------------------------------
    training_rows = []
    if paccs is not None and caccs is not None:
        E = min(len(paccs), len(caccs))
        for e in range(E):
            training_rows.append({
                "experiment": exp_name,
                "mode": detailed["mode"],
                "k": int(detailed["k"]),
                "epoch": e + 1,
                "cta": float(caccs[e, 0]),
                "clean_loss": float(caccs[e, 1]) if caccs.shape[1] > 1 else None,
                "pta": float(paccs[e, 0]),
                "poison_loss": float(paccs[e, 1]) if paccs.shape[1] > 1 else None,
            })

    return {
        "row": row,
        "sweep": [{**r, "experiment": exp_name, "mode": detailed["mode"],
                   "k": int(detailed["k"])} for r in sweep_rows] if sweep_rows else [],
        "per_class": per_class_rows,
        "training": training_rows,
    }


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
    report_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_sweeps = []
    all_perclass = []
    all_training = []

    for k in args.k_values:
        for mode in args.modes:
            exp = f"{args.name_prefix}_{mode}_k{k}"
            out = process_experiment(project_root, exp, threshold=args.threshold)
            if out is None:
                continue
            all_rows.append(out["row"])
            all_sweeps.extend(out["sweep"])
            all_perclass.extend(out["per_class"])
            all_training.extend(out["training"])

    if not all_rows:
        raise SystemExit("[aggregate] no experiments found; nothing to write.")

    master = pd.DataFrame(all_rows).sort_values(["mode", "k"]).reset_index(drop=True)
    master_csv = report_dir / "master_metrics.csv"
    master_json = report_dir / "master_metrics.json"
    master.to_csv(master_csv, index=False)
    master.to_json(master_json, orient="records", indent=2)
    print(f"[aggregate] wrote {master_csv} ({len(master)} rows)")

    if all_sweeps:
        sweep = pd.DataFrame(all_sweeps)
        sweep_csv = report_dir / "threshold_sweep.csv"
        sweep.to_csv(sweep_csv, index=False)
        print(f"[aggregate] wrote {sweep_csv} ({len(sweep)} rows)")

    if all_perclass:
        pc = pd.DataFrame(all_perclass)
        pc_csv = report_dir / "per_class_detection.csv"
        pc.to_csv(pc_csv, index=False)
        print(f"[aggregate] wrote {pc_csv} ({len(pc)} rows)")

    if all_training:
        tr = pd.DataFrame(all_training)
        tr_csv = report_dir / "training_curves.csv"
        tr.to_csv(tr_csv, index=False)
        print(f"[aggregate] wrote {tr_csv} ({len(tr)} rows)")


if __name__ == "__main__":
    main()
