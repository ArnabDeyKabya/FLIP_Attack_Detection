"""
Defense-mode wrapper for the KNN defense on the FLIP codebase.

Given the user_dataset built by train_user, SSL features, and the assigned
hard labels, this module:
    1. Always runs KNNDetector to produce per-sample suspicion scores.
    2. Always computes detection metrics (AUROC / AUPRC / precision@k /
       recall@k) so the mode='none' baseline emits comparable numbers for
       free.
    3. If mode='remove', applies the configured removal policy and returns
       a Subset of the user_dataset with the flagged rows dropped.
    4. If mode='none', returns the user_dataset unchanged so the run is
       byte-for-byte equivalent to vanilla train_user (regression check).

Removal policies (knn_cfg["removal_count"]):
    'auto'   : remove samples with score > threshold. Defender-side decision,
               no oracle knowledge — used for the headline mitigation result.
    'budget' : remove the top-N samples by score, with N = num_true_poisoned.
               Oracle-k cut. Used for apples-to-apples comparison with the
               Deep k-NN literature (Peri et al. 2020) where precision@k is
               the standard reported metric.
    int N    : remove the top-N samples by score. Used for the
               "2 * budget" over-removal ablation.

The ground-truth poisoned mask (is_poisoned_gt) is consumed for METRIC
LOGGING ONLY. The 'auto' policy does not read it; the 'budget' policy uses
its sum to set N (this is the oracle assumption, by design).
"""

from typing import Tuple, Union

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import Dataset, Subset

from modules.knn_defense.knn_detector import KNNDetector


SUPPORTED_MODES = {"none", "remove"}

RemovalPolicy = Union[str, int]


def _compute_detection_metrics(
    scores: np.ndarray,
    is_poisoned_gt: np.ndarray,
    flagged_indices: np.ndarray,
) -> dict:
    """Compute threshold-free metrics plus point metrics at the chosen cut.

    All values are Python scalars (or None) so the dict is JSON-serialisable.
    Returns None for any metric that is undefined for the given inputs
    (e.g. AUROC when there are zero true positives).
    """
    N = int(len(scores))
    n_true_pos = int(is_poisoned_gt.sum())
    n_flagged = int(len(flagged_indices))

    metrics = {
        "n_samples": N,
        "n_true_poisoned": n_true_pos,
        "n_flagged": n_flagged,
    }

    if 0 < n_true_pos < N:
        metrics["auroc"] = float(roc_auc_score(is_poisoned_gt, scores))
        metrics["auprc"] = float(
            average_precision_score(is_poisoned_gt, scores)
        )
    else:
        metrics["auroc"] = None
        metrics["auprc"] = None

    # Precision@k / Recall@k at k = num_true_poisoned (literature standard).
    if n_true_pos > 0:
        topk_idx = np.argsort(-scores, kind="stable")[:n_true_pos]
        tp_at_k = int(is_poisoned_gt[topk_idx].sum())
        metrics["precision_at_k"] = tp_at_k / n_true_pos
        metrics["recall_at_k"] = tp_at_k / n_true_pos
    else:
        metrics["precision_at_k"] = None
        metrics["recall_at_k"] = None

    # Point metrics at the actual flagging cut.
    if n_flagged > 0 and n_true_pos > 0:
        flag_mask = np.zeros(N, dtype=bool)
        flag_mask[flagged_indices] = True
        gt_mask = is_poisoned_gt.astype(bool)
        tp = int((flag_mask & gt_mask).sum())
        metrics["flagged_precision"] = tp / n_flagged
        metrics["flagged_recall"] = tp / n_true_pos
    else:
        metrics["flagged_precision"] = None
        metrics["flagged_recall"] = None

    return metrics


def _choose_removal_indices(
    scores: np.ndarray,
    policy: RemovalPolicy,
    threshold: float,
    n_true_poisoned: int,
) -> np.ndarray:
    """Resolve the removal policy to a concrete index array.

    Returns an int64 ndarray of indices to remove (possibly empty). Ties
    are broken in a stable, dataset-order-preserving way via argsort with
    kind='stable'.
    """
    N = int(len(scores))

    if isinstance(policy, str):
        if policy == "auto":
            return np.where(scores > threshold)[0].astype(np.int64)
        if policy == "budget":
            if n_true_poisoned <= 0:
                return np.empty(0, dtype=np.int64)
            n = min(n_true_poisoned, N)
            return np.argsort(-scores, kind="stable")[:n].astype(np.int64)
        raise ValueError(
            f"unknown removal_count policy '{policy}'. "
            f"Expected 'auto', 'budget', or an int."
        )

    if isinstance(policy, (int, np.integer)):
        n = int(policy)
        if n < 0:
            raise ValueError(f"removal_count int must be >= 0, got {n}")
        if n == 0:
            return np.empty(0, dtype=np.int64)
        n = min(n, N)
        return np.argsort(-scores, kind="stable")[:n].astype(np.int64)

    raise TypeError(
        f"removal_count must be str or int, got {type(policy).__name__}"
    )


def apply_defense(
    user_dataset: Dataset,
    features: np.ndarray,
    hard_labels: np.ndarray,
    is_poisoned_gt: np.ndarray,
    knn_cfg: dict,
    mode: str,
) -> Tuple[Dataset, dict]:
    """Run detection and apply the configured defense mode.

    Args:
        user_dataset:   the LabelWrappedDataset built by train_user. The
                        defense may wrap it in a Subset(kept_indices) or
                        return it unchanged.
        features:       (N, D) float32 L2-normalised SSL features.
        hard_labels:    (N,) int, argmax of the (possibly poisoned) label
                        tensor — i.e. the label the user model would
                        actually train on.
        is_poisoned_gt: (N,) {0, 1}, ground truth poisoned mask. Used for
                        METRIC LOGGING ONLY (and to set N for the 'budget'
                        oracle policy).
        knn_cfg:        the [train_user_defense.knn] config block. Required
                        keys: 'k'. Optional keys: 'scoring' (default
                        'disagreement'), 'removal_count' (default 'auto'),
                        'threshold' (default 0.5).
        mode:           'none' or 'remove'.

    Returns:
        (filtered_dataset, info)
            filtered_dataset:
                user_dataset if mode='none' (regression-equivalent to vanilla
                train_user), or Subset(user_dataset, kept_indices) otherwise.
            info: dict with keys
                  scores, neighbor_indices, neighbor_sims,
                  kept_indices, removed_indices,
                  detection_metrics, policy_resolved, k, scoring.
    """
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"unknown mode '{mode}'. Supported: {sorted(SUPPORTED_MODES)}"
        )

    N = int(features.shape[0])
    if hard_labels.shape != (N,):
        raise ValueError(
            f"hard_labels shape {hard_labels.shape} does not match "
            f"features rows ({N},)"
        )
    if is_poisoned_gt.shape != (N,):
        raise ValueError(
            f"is_poisoned_gt shape {is_poisoned_gt.shape} does not match "
            f"features rows ({N},)"
        )
    if len(user_dataset) != N:
        raise ValueError(
            f"user_dataset length {len(user_dataset)} does not match "
            f"features rows ({N})"
        )

    detector = KNNDetector(
        k=int(knn_cfg["k"]),
        scoring=knn_cfg.get("scoring", "disagreement"),
    )
    result = detector.detect(features, hard_labels)

    if mode == "none":
        removed = np.empty(0, dtype=np.int64)
        policy_resolved = {"mode": "none"}
    else:
        policy = knn_cfg.get("removal_count", "auto")
        threshold = float(knn_cfg.get("threshold", 0.5))
        n_true_pos = int(is_poisoned_gt.sum())
        removed = _choose_removal_indices(
            result.scores, policy, threshold, n_true_pos,
        )
        policy_resolved = {
            "mode": "remove",
            "removal_count": policy,
            "threshold": threshold,
            "n_true_poisoned": n_true_pos,
        }

    kept_mask = np.ones(N, dtype=bool)
    kept_mask[removed] = False
    kept = np.where(kept_mask)[0].astype(np.int64)

    if mode == "none":
        filtered_dataset = user_dataset
    else:
        filtered_dataset = Subset(user_dataset, kept.tolist())

    detection_metrics = _compute_detection_metrics(
        scores=result.scores,
        is_poisoned_gt=is_poisoned_gt.astype(np.int32, copy=False),
        flagged_indices=removed,
    )

    info = {
        "scores": result.scores,
        "neighbor_indices": result.neighbor_indices,
        "neighbor_sims": result.neighbor_sims,
        "kept_indices": kept,
        "removed_indices": removed,
        "detection_metrics": detection_metrics,
        "policy_resolved": policy_resolved,
        "k": result.k,
        "scoring": result.scoring,
    }
    return filtered_dataset, info
