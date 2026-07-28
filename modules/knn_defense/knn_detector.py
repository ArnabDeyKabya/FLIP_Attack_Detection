"""
k-NN neighborhood-consistency detector for the KNN defense wrapper on FLIP.

For each sample i with assigned hard label y_i, find its k nearest neighbors
in SSL feature space (cosine similarity = inner product on L2-normalised
features) and score how many of those neighbors disagree with y_i. A high
disagreement score means "this sample's label is inconsistent with its
semantic neighborhood" — the signature of a label-flip poisoning.

Two scoring modes:
    disagreement          : score = mean_j 1[y_j != y_i]                 in [0, 1]
    weighted_disagreement : score = sum_j sim(i,j) * 1[y_j != y_i]       in [0, 1]
                                    / sum_j sim(i,j)

A score of 0.5 means "majority of k semantic neighbors disagree with the
assigned label" — the natural threshold for the 'auto' removal policy.

Backend: chunked torch matmul + torch.topk. Uses CUDA when available,
falls back to CPU. Peak extra memory is O(chunk_size * N) — well under 1 GB
for CIFAR-scale (N=50k, chunk_size=1024).

The neighbor graph depends only on features, never on labels. `neighbor_graph`
exposes it separately so the same graph can be rescored under many hypothetical
label assignments without recomputing the O(N^2 D) similarity scan. The adaptive
attacker (modules/select_flips_adaptive) uses this to evaluate, for every
candidate flip, the score the defender would assign — guaranteeing attacker and
defender share one implementation rather than two that can silently diverge.
"""

from dataclasses import dataclass

import numpy as np
import torch


SUPPORTED_SCORINGS = {"disagreement", "weighted_disagreement"}


def scores_from_neighbors(
    neighbor_indices: np.ndarray,
    neighbor_sims: np.ndarray,
    label_assignment: np.ndarray,
    query_labels: np.ndarray,
    scoring: str = "disagreement",
    chunk_size: int = 4096,
) -> np.ndarray:
    """Score a precomputed neighbor graph under an arbitrary label assignment.

    This is the scoring half of KNNDetector.detect, factored out so it can be
    replayed cheaply on hypothetical labelings.

    Args:
        neighbor_indices: (N, k) int64 neighbor ids, as returned by
                          KNNDetector.neighbor_graph.
        neighbor_sims:    (N, k) float32 similarities, descending. Only read
                          when scoring='weighted_disagreement'; may be None
                          otherwise.
        label_assignment: (N,) int, the label each dataset row *carries*.
                          Neighbor labels are read from this array.
        query_labels:     (N,) int, the label being tested for each row. Pass
                          the same array as label_assignment to reproduce
                          KNNDetector.detect; pass a hypothetical flipped
                          labeling to ask "what would the defender score this
                          sample if I flipped it to L?".
        scoring:          'disagreement' or 'weighted_disagreement'.
        chunk_size:       rows per block, bounding peak memory at O(chunk*k).

    Returns:
        (N,) float32 scores in [0, 1], higher = more suspicious.
    """
    if scoring not in SUPPORTED_SCORINGS:
        raise ValueError(
            f"unknown scoring '{scoring}'. Supported: {sorted(SUPPORTED_SCORINGS)}"
        )
    N, k = neighbor_indices.shape
    if label_assignment.shape != (N,):
        raise ValueError(
            f"label_assignment shape {label_assignment.shape} does not match "
            f"neighbor rows ({N},)"
        )
    if query_labels.shape != (N,):
        raise ValueError(
            f"query_labels shape {query_labels.shape} does not match "
            f"neighbor rows ({N},)"
        )
    if scoring == "weighted_disagreement" and neighbor_sims is None:
        raise ValueError("weighted_disagreement requires neighbor_sims")

    labels = label_assignment.astype(np.int64, copy=False)
    queries = query_labels.astype(np.int64, copy=False)
    scores = np.empty(N, dtype=np.float32)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        nbr_labels = labels[neighbor_indices[start:end]]        # (B, k)
        disagree = (nbr_labels != queries[start:end, None])     # (B, k) bool

        if scoring == "disagreement":
            scores[start:end] = disagree.mean(axis=1, dtype=np.float64)
        else:
            sims = np.clip(
                neighbor_sims[start:end].astype(np.float64), 0.0, None,
            )
            total = sims.sum(axis=1)
            weighted = (sims * disagree).sum(axis=1)
            unweighted = disagree.mean(axis=1, dtype=np.float64)
            scores[start:end] = np.where(
                total > 0, weighted / np.maximum(total, 1e-12), unweighted,
            )

    return scores


@dataclass
class KNNDetection:
    """Container for k-NN detection results.

    Attributes:
        scores:           (N,) float32, higher = more suspicious, in [0, 1].
        neighbor_indices: (N, k) int64, dataset indices of the k nearest
                          neighbors of each sample (excluding self), sorted
                          by descending similarity.
        neighbor_sims:    (N, k) float32, cosine similarities to those
                          neighbors, descending.
        hard_labels:      (N,) int64, copy of the labels used for scoring.
        k:                int, the k used.
        scoring:          str, the scoring mode used.
    """
    scores: np.ndarray
    neighbor_indices: np.ndarray
    neighbor_sims: np.ndarray
    hard_labels: np.ndarray
    k: int
    scoring: str


class KNNDetector:
    """k-NN neighborhood-consistency detector."""

    def __init__(
        self,
        k: int = 20,
        scoring: str = "disagreement",
        chunk_size: int = 1024,
        device: str = None,
    ):
        if scoring not in SUPPORTED_SCORINGS:
            raise ValueError(
                f"unknown scoring '{scoring}'. "
                f"Supported: {sorted(SUPPORTED_SCORINGS)}"
            )
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

        self.k = k
        self.scoring = scoring
        self.chunk_size = chunk_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def neighbor_graph(self, features: np.ndarray):
        """Compute the label-independent k-NN graph.

        Separated from detect() because the graph depends only on features:
        it can be computed once and rescored under many label assignments
        (see scores_from_neighbors).

        Args:
            features: (N, D) float32 L2-normalised feature matrix.

        Returns:
            (neighbor_indices, neighbor_sims): (N, k) int64 and (N, k) float32,
            self excluded, sorted by descending similarity.
        """
        N, D = features.shape
        if N == 0:
            raise ValueError("features must be non-empty")
        if self.k >= N:
            raise ValueError(f"k={self.k} must be strictly less than N={N}")
        if features.dtype != np.float32:
            features = features.astype(np.float32, copy=False)

        norms = np.linalg.norm(features, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(
                f"features are not L2-normalised "
                f"(norm range {norms.min():.6f}..{norms.max():.6f}). "
                f"Run SSLFeatureExtractor.extract or normalise upstream."
            )

        feats_t = torch.from_numpy(features).to(self.device)

        top_sims = torch.empty(
            (N, self.k), dtype=torch.float32, device=self.device,
        )
        top_idx = torch.empty(
            (N, self.k), dtype=torch.long, device=self.device,
        )

        for start in range(0, N, self.chunk_size):
            end = min(start + self.chunk_size, N)
            chunk = feats_t[start:end]                       # (B, D)
            sims = chunk @ feats_t.T                         # (B, N)

            # Mask out self-similarity: sims[i, start+i] = -inf for i in [0, B).
            local_idx = torch.arange(end - start, device=self.device)
            global_idx = torch.arange(start, end, device=self.device)
            sims[local_idx, global_idx] = float("-inf")

            vals, inds = torch.topk(
                sims, self.k, dim=1, largest=True, sorted=True,
            )
            top_sims[start:end] = vals
            top_idx[start:end] = inds

        return (
            top_idx.detach().cpu().numpy().astype(np.int64),
            top_sims.detach().cpu().numpy().astype(np.float32),
        )

    def detect(
        self,
        features: np.ndarray,
        hard_labels: np.ndarray,
    ) -> KNNDetection:
        """Run detection on the full training set.

        Args:
            features:    (N, D) float32 L2-normalised feature matrix, as
                         produced by SSLFeatureExtractor.extract.
            hard_labels: (N,) integer labels — the labels the user model
                         would actually train on (i.e. argmax of the soft
                         label tensor, or the hard label tensor itself).

        Returns:
            KNNDetection with per-sample suspicion scores and neighborhoods.
        """
        N = features.shape[0]
        if hard_labels.shape != (N,):
            raise ValueError(
                f"hard_labels shape {hard_labels.shape} does not match "
                f"features rows ({N},)"
            )

        neighbor_indices, neighbor_sims = self.neighbor_graph(features)
        labels = hard_labels.astype(np.int64, copy=False)

        scores = scores_from_neighbors(
            neighbor_indices=neighbor_indices,
            neighbor_sims=neighbor_sims,
            label_assignment=labels,
            query_labels=labels,
            scoring=self.scoring,
        )

        return KNNDetection(
            scores=scores,
            neighbor_indices=neighbor_indices,
            neighbor_sims=neighbor_sims,
            hard_labels=hard_labels.astype(np.int64, copy=True),
            k=self.k,
            scoring=self.scoring,
        )
