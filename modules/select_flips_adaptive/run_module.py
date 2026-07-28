"""
Adaptive (defense-aware) FLIP: selects label flips under a stealth constraint.

THREAT MODEL
------------
The baseline `select_flips` module implements a *defense-unaware* attacker: it
ranks every training sample by how easily the expert trajectory can be pushed
to misclassify it, and flips the top-N. That attacker has no idea a semantic
audit is coming, so it happily flips trucks to "deer" — samples whose k-NN
neighbourhood in DINOv2 space screams disagreement.

This module implements the *defense-aware* attacker assumed by Kerckhoffs's
principle: it knows the encoder, k, the scoring rule and the threshold, and it
can evaluate the defender's score for any candidate flip before committing to
it. It therefore solves a constrained version of the same problem:

    maximise   sum of FLIP margins over the selected set S
    subject to |S| = N
               defender_score(i, flipped_label(i)) <= tau   for all i in S

`tau` is the attacker's stealth budget. tau = 1.0 imposes no constraint and
must reproduce `select_flips` exactly (asserted in the manifest as
`reproduces_baseline`). Lowering tau forces the attacker to abandon its
highest-margin candidates in favour of flips that survive the audit — and the
resulting margin loss is precisely the price of evading detection.

Sweeping tau traces the attacker's stealth-vs-efficacy frontier. Training a
victim model on each output measures what that price buys in attack success.

WHAT THE ATTACKER MAY AND MAY NOT DO
------------------------------------
May:     choose *which* samples to flip, with full knowledge of the defense.
May not: choose a *different* target label than FLIP's optimisation assigned.
         Re-optimising the label assignment jointly with the stealth constraint
         is a strictly stronger attack and is left as future work; the frontier
         measured here is therefore a lower bound on attacker capability.

POISON-POISON REINFORCEMENT
---------------------------
The defender scores a sample against the labels its neighbours *carry*, which
after the attack includes other poisons. A cluster of co-located flips mutually
lowers its own disagreement scores — a real effect the attacker can exploit for
free. `refine_iters` runs the selection to a fixed point over the poisoned
assignment rather than the clean one, so the attacker gets that benefit. Set
`refine_iters = 1` to disable and score against clean labels only.

Outputs (under output_dir/):
    tau_{tau}/{budget}.npy      poisoned label tensors, same format and
                                directory layout as select_flips, so any
                                train_user / train_user_defense config can
                                consume them by path alone
    tau_{tau}/true.npy          clean labels (copied, for convenience)
    tau_{tau}/selected_{budget}.npy   int64 indices of the flipped samples
    manifest.json               per-(tau, budget) attacker-side diagnostics
"""

from pathlib import Path
import json
import glob
import sys

import numpy as np

from modules.base_utils.util import extract_toml, slurmify_path
from modules.knn_defense.ssl_features import SSLFeatureExtractor
from modules.knn_defense.knn_detector import KNNDetector, scores_from_neighbors


# select_flips writes the flipped row as (expert soft label - PENALTY * onehot),
# which drives the true class below every competitor so argmax lands on the
# expert's preferred wrong class. Mirrored here so both modules agree.
TRUE_CLASS_PENALTY = 50000


def _compute_margins(input_label_glob: str, true: np.ndarray):
    """Rank samples by how readily the expert ensemble abandons the true class.

    Mirrors the margin computation in modules/select_flips/run_module.py.

    For a sample the experts already misclassify, the margin is
    (top logit - true-class logit) > 0: the attack is essentially free.
    For a correctly classified sample it is (runner-up - top) < 0: how close
    the model already is to flipping. Higher is easier to attack.

    Returns:
        (margins, mean_labels): (N,) float64 margins minimised over expert
        checkpoints, and the (N, C) mean expert soft labels.
    """
    distances = []
    all_labels = []

    matches = sorted(glob.glob(input_label_glob))
    if not matches:
        raise FileNotFoundError(
            f"input_label_glob matched no files: {input_label_glob}"
        )

    for f in matches:
        labels = np.load(f)

        dists = np.zeros(len(labels))
        inds = labels.argmax(axis=1) != true.argmax(axis=1)
        dists[inds] = labels[inds].max(axis=1) - \
            labels[inds][np.arange(inds.sum()), true[inds].argmax(axis=1)]

        ordered = np.sort(labels[~inds])
        dists[~inds] = ordered[:, -2] - ordered[:, -1]
        distances.append(dists)
        all_labels.append(labels)

    margins = np.stack(distances).min(axis=0)
    mean_labels = np.stack(all_labels).mean(axis=0)
    return margins, mean_labels


def _flipped_rows(mean_labels: np.ndarray, true: np.ndarray) -> np.ndarray:
    """The label row select_flips would write for every sample, if flipped."""
    return mean_labels - TRUE_CLASS_PENALTY * true


def _select(margins: np.ndarray, eligible: np.ndarray, n: int) -> np.ndarray:
    """Top-n eligible samples by margin, ties broken as in select_flips.

    select_flips uses `np.argsort(margins)[-n:]`, so replicate that ordering
    exactly on the eligible subset: mask ineligible margins to -inf and take
    the same tail slice.
    """
    masked = np.where(eligible, margins, -np.inf)
    n_eligible = int(eligible.sum())
    take = min(n, n_eligible)
    if take == 0:
        return np.empty(0, dtype=np.int64)
    return np.argsort(masked)[-take:].astype(np.int64)


def run(experiment_name, module_name, **kwargs):
    slurm_id = kwargs.get("slurm_id", None)
    args = extract_toml(experiment_name, module_name)

    budgets = args.get("budgets", [1500])
    taus = args.get("taus", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1])
    refine_iters = int(args.get("refine_iters", 3))
    dataset_flag = args["dataset"]

    input_label_glob = slurmify_path(args["input_label_glob"], slurm_id)
    true_labels = slurmify_path(args["true_labels"], slurm_id)
    output_dir = Path(slurmify_path(args["output_dir"], slurm_id))
    output_dir.mkdir(parents=True, exist_ok=True)

    knn_cfg = dict(args["knn"])
    k = int(knn_cfg["k"])
    scoring = knn_cfg.get("scoring", "disagreement")

    if refine_iters < 1:
        raise ValueError(f"refine_iters must be >= 1, got {refine_iters}")

    true = np.load(true_labels)
    true_hard = true.argmax(axis=1).astype(np.int64)
    n_total = len(true)

    print("Calculating margins...")
    margins, mean_labels = _compute_margins(input_label_glob, true)
    flip_rows = _flipped_rows(mean_labels, true)
    flip_hard = flip_rows.argmax(axis=1).astype(np.int64)

    n_unchanged = int((flip_hard == true_hard).sum())
    if n_unchanged:
        # Would mean the penalty failed to dislodge the true class — the
        # sample would be "flipped" to its own label and silently do nothing.
        raise RuntimeError(
            f"{n_unchanged} candidate flips resolve to the true label; "
            f"TRUE_CLASS_PENALTY={TRUE_CLASS_PENALTY} is too small for these "
            f"expert logits."
        )

    print(f"Building k-NN graph (k={k}, encoder={knn_cfg['encoder']})...")
    extractor = SSLFeatureExtractor(knn_cfg["encoder"])
    features = extractor.extract(
        dataset_flag=dataset_flag,
        cache_dir=knn_cfg["feature_cache"],
        batch_size=int(knn_cfg.get("batch_size", 256)),
    )
    if features.shape[0] != n_total:
        raise RuntimeError(
            f"Feature/label count mismatch: features={features.shape[0]}, "
            f"labels={n_total}"
        )

    detector = KNNDetector(k=k, scoring=scoring)
    neighbor_indices, neighbor_sims = detector.neighbor_graph(features)

    manifest = {
        "dataset": dataset_flag,
        "encoder": knn_cfg["encoder"],
        "k": k,
        "scoring": scoring,
        "refine_iters": refine_iters,
        "n_total": n_total,
        "input_label_glob": input_label_glob,
        "runs": [],
    }

    np.save(output_dir / "true.npy", true)

    for n in budgets:
        # Reference point: what the defense-unaware attacker would pick.
        baseline_idx = _select(margins, np.ones(n_total, dtype=bool), n)
        baseline_set = set(baseline_idx.tolist())
        baseline_margin = float(margins[baseline_idx].mean())

        for tau in taus:
            tau = float(tau)
            tau_dir = output_dir / f"tau_{tau:.3f}"
            tau_dir.mkdir(parents=True, exist_ok=True)

            # Iterate to a fixed point: score candidates against the labels
            # the dataset carries *after* the currently-selected flips, so
            # co-located poisons get credit for shielding each other.
            assignment = true_hard.copy()
            selected = np.empty(0, dtype=np.int64)
            iter_stats = []

            for it in range(refine_iters):
                cand_scores = scores_from_neighbors(
                    neighbor_indices=neighbor_indices,
                    neighbor_sims=neighbor_sims,
                    label_assignment=assignment,
                    query_labels=flip_hard,
                    scoring=scoring,
                )
                eligible = cand_scores <= tau
                new_selected = _select(margins, eligible, n)

                prev, cur = set(selected.tolist()), set(new_selected.tolist())
                union = len(prev | cur)
                iter_stats.append({
                    "iter": it,
                    "n_eligible": int(eligible.sum()),
                    "n_selected": int(len(new_selected)),
                    "jaccard_vs_prev": (
                        len(prev & cur) / union if union else 1.0
                    ),
                })

                selected = new_selected
                assignment = true_hard.copy()
                assignment[selected] = flip_hard[selected]

                if it > 0 and iter_stats[-1]["jaccard_vs_prev"] == 1.0:
                    break

            # Final scores are what the defender actually sees at train time.
            final_scores = scores_from_neighbors(
                neighbor_indices=neighbor_indices,
                neighbor_sims=neighbor_sims,
                label_assignment=assignment,
                query_labels=assignment,
                scoring=scoring,
            )

            poisoned = np.zeros(n_total, dtype=bool)
            poisoned[selected] = True

            labels_out = true.copy()
            labels_out[selected] = flip_rows[selected]
            np.save(tau_dir / f"{n}.npy", labels_out)
            np.save(tau_dir / f"selected_{n}.npy", selected)

            sel_set = set(selected.tolist())
            overlap_union = len(baseline_set | sel_set)
            run_stats = {
                "budget": int(n),
                "tau": tau,
                "label_path": str(tau_dir / f"{n}.npy"),
                "n_requested": int(n),
                "n_selected": int(len(selected)),
                "budget_satisfied": bool(len(selected) == n),
                "n_eligible": iter_stats[-1]["n_eligible"],
                "mean_margin": (
                    float(margins[selected].mean()) if len(selected) else None
                ),
                "baseline_mean_margin": baseline_margin,
                "margin_retention": (
                    float(margins[selected].mean() / baseline_margin)
                    if len(selected) and baseline_margin != 0 else None
                ),
                "jaccard_vs_unconstrained": (
                    len(baseline_set & sel_set) / overlap_union
                    if overlap_union else 1.0
                ),
                "mean_defender_score_on_poisons": (
                    float(final_scores[poisoned].mean())
                    if poisoned.any() else None
                ),
                "mean_defender_score_on_clean": float(
                    final_scores[~poisoned].mean()
                ),
                "frac_poisons_above_half": (
                    float((final_scores[poisoned] > 0.5).mean())
                    if poisoned.any() else None
                ),
                "frac_selected_from_source_class": (
                    float((true_hard[selected] == args["source_label"]).mean())
                    if len(selected) else None
                ),
                "iterations": iter_stats,
            }

            if tau >= 1.0:
                # tau=1.0 cannot exclude anything (scores live in [0, 1]), so
                # this must be the unconstrained attack. Asserting it here
                # makes the whole sweep self-validating.
                run_stats["reproduces_baseline"] = bool(
                    sel_set == baseline_set
                )

            manifest["runs"].append(run_stats)

            print(
                f"  budget={n:5d} tau={tau:.3f}  "
                f"selected={len(selected):5d}/{n}  "
                f"eligible={run_stats['n_eligible']:6d}  "
                f"margin_retention="
                f"{run_stats['margin_retention'] or float('nan'):.3f}  "
                f"mean_score_on_poisons="
                f"{run_stats['mean_defender_score_on_poisons'] or float('nan'):.3f}"
            )

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(manifest['runs'])} label sets to {output_dir}")


if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
