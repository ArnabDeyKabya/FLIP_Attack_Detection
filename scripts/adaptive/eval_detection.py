"""
Detection-side evaluation of the adaptive attacker — no victim training needed.

Whether the audit still *finds* the poisons is a property of the labels and the
feature space alone, so it can be answered for every tau in seconds. Only the
question "does the surviving attack still install a backdoor?" needs training
(scripts/adaptive/run_adaptive_sweep.py).

Sweeps the defender's k as well as the attacker's tau, which answers the
obvious rebuttal: the attacker optimised against k=20, so can the defender
simply audit at a different k and recover? A defender who can escape by
changing k has a cheap fix; one who cannot has a real problem.

Writes experiments/_report_adaptive_cifar_1xs/detection_vs_tau.csv and .md.

Usage:
    python scripts/adaptive/eval_detection.py
    python scripts/adaptive/eval_detection.py --defender-ks 20 --thresholds 0.5
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.base_utils.util import softmax                          # noqa: E402
from modules.knn_defense.knn_detector import KNNDetector             # noqa: E402
from modules.knn_defense.ssl_features import SSLFeatureExtractor     # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score   # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attack-dir", default="experiments/adaptive_flip_cifar_1xs")
    p.add_argument("--budget", type=int, default=1500)
    p.add_argument("--defender-ks", type=int, nargs="+",
                   default=[5, 20, 100, 500])
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.5])
    p.add_argument("--dataset", default="cifar")
    p.add_argument("--encoder", default="dinov2_vits14")
    p.add_argument("--feature-cache", default="data/ssl_features")
    p.add_argument("--out-dir", default="experiments/_report_adaptive_cifar_1xs")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / args.attack_dir / "manifest.json").read_text())
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    features = SSLFeatureExtractor(args.encoder).extract(
        dataset_flag=args.dataset,
        cache_dir=str(root / args.feature_cache),
    )
    true = np.load(root / args.attack_dir / "true.npy")
    true_hard = true.argmax(axis=1).astype(np.int64)

    graphs = {}
    for k in args.defender_ks:
        det = KNNDetector(k=k)
        graphs[k] = (det, *det.neighbor_graph(features))

    rows = []
    for r in sorted(manifest["runs"], key=lambda x: -x["tau"]):
        if r["budget"] != args.budget:
            continue
        tau = r["tau"]
        labels = np.load(root / args.attack_dir / f"tau_{tau:.3f}" /
                         f"{args.budget}.npy")
        hard = softmax(torch.tensor(labels)).argmax(dim=1).numpy().astype(np.int64)
        gt = (hard != true_hard).astype(np.int32)
        n_pois = int(gt.sum())

        for k, (det, nbr_idx, nbr_sims) in graphs.items():
            scores = det.detect(features, hard).scores
            auroc = float(roc_auc_score(gt, scores)) if 0 < n_pois else None
            auprc = float(average_precision_score(gt, scores)) if 0 < n_pois else None
            topk = np.argsort(-scores, kind="stable")[:n_pois]
            prec_at_k = float(gt[topk].sum() / n_pois) if n_pois else None

            for thr in args.thresholds:
                flagged = scores > thr
                n_flag = int(flagged.sum())
                tp = int((flagged & gt.astype(bool)).sum())
                rows.append({
                    "tau": tau,
                    "attacker_k": manifest["k"],
                    "defender_k": k,
                    "threshold": thr,
                    "n_poisons": n_pois,
                    "auroc": auroc,
                    "auprc": auprc,
                    "precision_at_k": prec_at_k,
                    "n_flagged": n_flag,
                    "flagged_precision": tp / n_flag if n_flag else None,
                    "flagged_recall": tp / n_pois if n_pois else None,
                    "poisons_missed": n_pois - tp,
                    "clean_removed": n_flag - tp,
                })

    csv_path = out_dir / "detection_vs_tau.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# Does the audit still detect a defense-aware attacker?",
        "",
        f"Attacker optimises against k={manifest['k']}, threshold 0.5. "
        f"Defender columns vary k to test whether simply auditing at a "
        f"different k recovers detection.",
        "",
        "`recall` is the fraction of actually-placed poisons the defender "
        "removes at its operating threshold — the number that decides whether "
        "the backdoor survives.",
        "",
    ]
    for thr in args.thresholds:
        lines += [
            f"## Threshold {thr}", "",
            "| tau | poisons placed | " + " | ".join(
                f"recall @ k={k}" for k in args.defender_ks
            ) + " | AUPRC @ attacker k | clean removed @ attacker k |",
            "|---" * (3 + len(args.defender_ks)) + "|",
        ]
        for tau in sorted({r["tau"] for r in rows}, reverse=True):
            sel = [r for r in rows if r["tau"] == tau and r["threshold"] == thr]
            by_k = {r["defender_k"]: r for r in sel}
            atk = by_k[manifest["k"]]
            cells = " | ".join(
                f"{by_k[k]['flagged_recall']:.3f}" if by_k[k]["flagged_recall"]
                is not None else "--" for k in args.defender_ks
            )
            lines.append(
                f"| {tau:.2f} | {atk['n_poisons']} | {cells} | "
                f"{atk['auprc']:.3f} | {atk['clean_removed']} |"
            )
        lines.append("")

    md_path = out_dir / "detection_vs_tau.md"
    md_path.write_text("\n".join(lines) + "\n")

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
