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
"""

from dataclasses import dataclass

import numpy as np
import torch


SUPPORTED_SCORINGS = {"disagreement", "weighted_disagreement"}


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
        N, D = features.shape
        if N == 0:
            raise ValueError("features must be non-empty")
        if hard_labels.shape != (N,):
            raise ValueError(
                f"hard_labels shape {hard_labels.shape} does not match "
                f"features rows ({N},)"
            )
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
        labels_t = torch.from_numpy(
            hard_labels.astype(np.int64, copy=False)
        ).to(self.device)

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

        neighbor_labels = labels_t[top_idx]                  # (N, k)
        query_labels = labels_t.unsqueeze(1)                 # (N, 1)
        disagree = (neighbor_labels != query_labels).float() # (N, k)

        if self.scoring == "disagreement":
            scores_t = disagree.mean(dim=1)
        else:  # weighted_disagreement
            sims_clipped = top_sims.clamp_min(0.0)
            total = sims_clipped.sum(dim=1)
            weighted = (sims_clipped * disagree).sum(dim=1)
            # When all k similarities are <= 0 (rare on L2-normed features
            # with semantic neighborhoods), fall back to unweighted score
            # rather than emitting a misleading 0.
            unweighted = disagree.mean(dim=1)
            scores_t = torch.where(
                total > 0,
                weighted / total.clamp_min(1e-12),
                unweighted,
            )

        return KNNDetection(
            scores=scores_t.detach().cpu().numpy().astype(np.float32),
            neighbor_indices=top_idx.detach().cpu().numpy().astype(np.int64),
            neighbor_sims=top_sims.detach().cpu().numpy().astype(np.float32),
            hard_labels=hard_labels.astype(np.int64, copy=True),
            k=self.k,
            scoring=self.scoring,
        )
