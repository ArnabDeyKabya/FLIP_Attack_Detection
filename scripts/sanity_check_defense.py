#!/usr/bin/env python
"""
End-to-end sanity check for the KNN defense modules on CIFAR-10 / CIFAR-100.

Two modes:

  --demo:
      Random label flipping baseline. Picks `--demo-budget` images of class
      `--demo-source`, relabels them as `--demo-target`. No FLIP labels needed.
      The defense should achieve AUROC > 0.9 here — anything less means the
      pipeline is broken. This is the cheapest way to validate the stack.

  --labels-path / --true-path:
      Use real FLIP precomputed labels (download from
      https://github.com/SewoongLab/FLIP/releases/). Numbers here are the
      actual research result for that (model, poisoner, budget) cell.

Outputs:
  - Printed summary (feature validation + detection metrics for modes
    none / auto / budget).
  - Saved per-sample scores, indices, and a JSON summary under
    out/knn_defense_sanity/<tag>/.

Usage (from project root):

  # Quickest validation, no FLIP labels required:
  python scripts/sanity_check_defense.py --demo

  # Real FLIP run (after downloading precomputed labels):
  python scripts/sanity_check_defense.py \
      --labels-path precomputed_labels/cifar/r32p/1xs/1500.npy \
      --true-path  precomputed_labels/cifar/r32p/1xs/true.npy
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import TensorDataset

from modules.base_utils.datasets import load_dataset
from modules.knn_defense.ssl_features import SSLFeatureExtractor
from modules.knn_defense.defense_modes import apply_defense


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def load_flip_label_pair(labels_path: Path, true_path: Path):
    """Load FLIP-format label tensors and convert to hard labels + GT mask.

    FLIP saves soft labels as (N, n_classes) numpy arrays. The argmax of each
    row is the (possibly flipped) hard label. is_poisoned = argmax(syn) !=
    argmax(true).
    """
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}\n"
            f"Download FLIP precomputed labels from "
            f"https://github.com/SewoongLab/FLIP/releases/"
        )
    if not true_path.exists():
        raise FileNotFoundError(f"True labels file not found: {true_path}")

    syn = np.load(labels_path)
    true = np.load(true_path)
    if syn.shape != true.shape:
        raise ValueError(
            f"shape mismatch: labels {syn.shape} vs true {true.shape}"
        )

    hard = syn.argmax(axis=1).astype(np.int64)
    true_hard = true.argmax(axis=1).astype(np.int64)
    is_poisoned = (hard != true_hard).astype(np.int32)
    return hard, true_hard, is_poisoned


def build_demo_labels(dataset_flag: str, source_label: int, target_label: int,
                      budget: int, seed: int = 0):
    """Random label flipping baseline: flip `budget` images from source_label
    to target_label. No FLIP optimisation involved.
    """
    base = load_dataset(dataset_flag, train=True)
    true_hard = np.array([y for _, y in base], dtype=np.int64)
    N = len(true_hard)

    source_idx = np.where(true_hard == source_label)[0]
    if budget > len(source_idx):
        raise ValueError(
            f"budget {budget} > source class {source_label} size "
            f"{len(source_idx)}"
        )
    rng = np.random.default_rng(seed)
    flip_idx = rng.choice(source_idx, size=budget, replace=False)

    hard = true_hard.copy()
    hard[flip_idx] = target_label
    is_poisoned = np.zeros(N, dtype=np.int32)
    is_poisoned[flip_idx] = 1
    return hard, true_hard, is_poisoned


# ---------------------------------------------------------------------------
# Feature validation — sanity gates #5, #6 from the plan
# ---------------------------------------------------------------------------

def validate_features(features: np.ndarray, true_hard: np.ndarray,
                      n_samples: int = 5000, seed: int = 0) -> dict:
    """Within / between class cosine sim + cosine k-NN class accuracy.

    Catches preprocessing bugs (wrong normalisation, wrong image size, etc.)
    cheaply. DINOv2 ViT-S/14 on CIFAR-10 should give gap >= 0.15 and k-NN
    accuracy >= 0.85.
    """
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(features))
    idx = rng.choice(len(features), n, replace=False)
    feats = features[idx]
    labs = true_hard[idx]

    sims = feats @ feats.T
    same = labs[:, None] == labs[None, :]
    diag = np.eye(n, dtype=bool)
    within_mask = same & ~diag
    between_mask = ~same & ~diag

    within = float(sims[within_mask].mean()) if within_mask.any() else float("nan")
    between = float(sims[between_mask].mean()) if between_mask.any() else float("nan")

    from sklearn.neighbors import KNeighborsClassifier
    split = n // 2
    knn = KNeighborsClassifier(n_neighbors=10, metric="cosine")
    knn.fit(feats[:split], labs[:split])
    acc = float(knn.score(feats[split:], labs[split:]))

    return {
        "within_class_sim": within,
        "between_class_sim": between,
        "gap": within - between,
        "knn_class_acc_k10": acc,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _fmt(v):
    if v is None:
        return "  n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _print_metrics(m: dict, indent: str = "  "):
    keys = ["auroc", "auprc", "precision_at_k", "recall_at_k",
            "flagged_precision", "flagged_recall",
            "n_samples", "n_true_poisoned", "n_flagged"]
    for k in keys:
        if k in m:
            print(f"{indent}{k:22s}: {_fmt(m[k])}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KNN defense sanity check on CIFAR-10 / CIFAR-100",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="cifar",
                        choices=["cifar", "cifar_100"])
    parser.add_argument("--encoder", default="dinov2_vits14",
                        choices=["dinov2_vits14", "dinov2_vitb14"])
    parser.add_argument("--k", type=int, default=20,
                        help="k for k-NN detection")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="threshold for 'auto' removal policy")
    parser.add_argument("--cache-dir", default="data/ssl_features")
    parser.add_argument("--output-dir", default="out/knn_defense_sanity")

    parser.add_argument("--labels-path", type=Path, default=None,
                        help="Path to FLIP poisoned soft labels .npy")
    parser.add_argument("--true-path", type=Path, default=None,
                        help="Path to FLIP true soft labels .npy")
    parser.add_argument("--demo", action="store_true",
                        help="Use random label flipping (no FLIP labels needed)")
    parser.add_argument("--demo-budget", type=int, default=1500)
    parser.add_argument("--demo-source", type=int, default=9,
                        help="Source class for demo flips (default: truck)")
    parser.add_argument("--demo-target", type=int, default=4,
                        help="Target class for demo flips (default: deer)")
    parser.add_argument("--demo-seed", type=int, default=0)

    args = parser.parse_args()

    if not args.demo and (args.labels_path is None or args.true_path is None):
        parser.error(
            "Either --demo OR both --labels-path and --true-path are required."
        )
    if args.demo and (args.labels_path is not None or args.true_path is not None):
        parser.error("--demo is mutually exclusive with --labels-path/--true-path.")

    print("=" * 70)
    print("KNN Defense — Sanity Check")
    print("=" * 70)

    # 1. Build labels
    if args.demo:
        print(f"\n[1/4] labels  : DEMO (random flips "
              f"{args.demo_source} -> {args.demo_target}, "
              f"budget={args.demo_budget}, seed={args.demo_seed})")
        hard_labels, true_hard, is_poisoned = build_demo_labels(
            args.dataset, args.demo_source, args.demo_target,
            args.demo_budget, args.demo_seed,
        )
        run_tag = f"demo_{args.dataset}_{args.demo_source}to{args.demo_target}_b{args.demo_budget}"
    else:
        print(f"\n[1/4] labels  : {args.labels_path}")
        hard_labels, true_hard, is_poisoned = load_flip_label_pair(
            args.labels_path, args.true_path,
        )
        run_tag = args.labels_path.parent.name + "_" + args.labels_path.stem

    N = len(hard_labels)
    n_poisoned = int(is_poisoned.sum())
    print(f"        N total     : {N}")
    print(f"        N poisoned  : {n_poisoned} ({100*n_poisoned/N:.2f}%)")

    # 2. Extract features (cached after first run)
    print(f"\n[2/4] features: {args.encoder} on {args.dataset}")
    extractor = SSLFeatureExtractor(args.encoder)
    features = extractor.extract(
        dataset_flag=args.dataset,
        cache_dir=args.cache_dir,
        batch_size=256,
    )
    if features.shape[0] != N:
        raise RuntimeError(
            f"Feature/label count mismatch: features={features.shape[0]}, "
            f"labels={N}. Check that --dataset matches the label file."
        )

    # 3. Validate feature quality
    print("\n[3/4] feature validation:")
    val = validate_features(features, true_hard)
    for k, v in val.items():
        print(f"        {k:22s}: {_fmt(v)}")
    # Soft gate — warn but don't fail. Detection numbers will tell us if
    # something is actually broken.
    if val["gap"] < 0.15 or val["knn_class_acc_k10"] < 0.60:
        print("        WARNING: feature quality below recommended thresholds "
              "(gap >= 0.15, k-NN acc >= 0.60). Detection may be unreliable.")

    # 4. Detection + defense modes
    dummy = TensorDataset(torch.zeros(N, 1), torch.from_numpy(hard_labels))

    print(f"\n[4/4] detection (k={args.k}, scoring=disagreement):")

    print("\n  mode='none' (baseline detection — no removal):")
    _, info_none = apply_defense(
        user_dataset=dummy, features=features,
        hard_labels=hard_labels, is_poisoned_gt=is_poisoned,
        knn_cfg={"k": args.k, "scoring": "disagreement"},
        mode="none",
    )
    _print_metrics(info_none["detection_metrics"], indent="    ")

    print(f"\n  mode='remove', policy='auto', threshold={args.threshold}:")
    _, info_auto = apply_defense(
        user_dataset=dummy, features=features,
        hard_labels=hard_labels, is_poisoned_gt=is_poisoned,
        knn_cfg={"k": args.k, "scoring": "disagreement",
                 "removal_count": "auto", "threshold": args.threshold},
        mode="remove",
    )
    _print_metrics(info_auto["detection_metrics"], indent="    ")

    print(f"\n  mode='remove', policy='budget' (oracle k={n_poisoned}):")
    _, info_budget = apply_defense(
        user_dataset=dummy, features=features,
        hard_labels=hard_labels, is_poisoned_gt=is_poisoned,
        knn_cfg={"k": args.k, "scoring": "disagreement",
                 "removal_count": "budget"},
        mode="remove",
    )
    _print_metrics(info_budget["detection_metrics"], indent="    ")

    # Save outputs
    out_dir = Path(args.output_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "scores.npy", info_none["scores"])
    np.save(out_dir / "is_poisoned_gt.npy", is_poisoned)
    np.save(out_dir / "removed_auto.npy", info_auto["removed_indices"])
    np.save(out_dir / "removed_budget.npy", info_budget["removed_indices"])

    summary = {
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "run_tag": run_tag,
        "n_total": N,
        "n_poisoned": n_poisoned,
        "feature_validation": val,
        "metrics_none": info_none["detection_metrics"],
        "metrics_auto": info_auto["detection_metrics"],
        "metrics_budget": info_budget["detection_metrics"],
        "policy_resolved_auto": info_auto["policy_resolved"],
        "policy_resolved_budget": info_budget["policy_resolved"],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[done] outputs saved under {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
