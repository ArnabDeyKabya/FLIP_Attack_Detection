"""
SSL feature extraction for the KNN defense wrapper on the FLIP codebase.

Loads a frozen DINOv2 encoder via torch.hub, runs every CIFAR-10 / CIFAR-100
training image through it once with a deterministic eval-mode transform
(no augmentation), L2-normalises the resulting CLS embeddings, and caches
them to disk.

Cache key is (dataset_flag, encoder_name) only. Features are independent of
labels / poisoners / budgets / seeds, so the same cache file is reused across
every defense experiment on the same dataset.

Output ordering matches the original torchvision CIFAR train ordering — i.e.
features[i] corresponds to load_dataset(dataset_flag, train=True)[i], which is
the same indexing FLIP's label tensors use (see modules/base_utils/datasets.py
construction of mtt_distill_dataset and the precomputed labels release).

Supported:
    dataset_flag : 'cifar' (CIFAR-10), 'cifar_100' (CIFAR-100)
    encoder_name : 'dinov2_vits14' (D=384), 'dinov2_vitb14' (D=768)
"""

from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from modules.base_utils.datasets import load_dataset


# DINOv2 was pretrained with ImageNet normalisation. Using CIFAR stats here
# silently degrades features and is a common, hard-to-notice bug.
SSL_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
SSL_NORMALIZE_STD = (0.229, 0.224, 0.225)
SSL_IMAGE_SIZE = 224

SSL_TRANSFORM = transforms.Compose([
    transforms.Resize(SSL_IMAGE_SIZE,
                      interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(SSL_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(SSL_NORMALIZE_MEAN, SSL_NORMALIZE_STD),
])


FEATURE_DIMS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
}

SUPPORTED_DATASETS = {'cifar', 'cifar_100'}


class _SSLImageDataset(Dataset):
    """Wraps a torchvision dataset to apply only the SSL eval transform.

    The base dataset is expected to yield (PIL.Image, int_label); the label
    is discarded here so this layer is provably label-independent.
    """

    def __init__(self, base_dataset: Dataset):
        self.base = base_dataset

    def __getitem__(self, i: int) -> torch.Tensor:
        x, _ = self.base[i]
        return SSL_TRANSFORM(x)

    def __len__(self) -> int:
        return len(self.base)


class SSLFeatureExtractor:
    """Frozen DINOv2 feature extractor with on-disk caching."""

    def __init__(self, encoder_name: str = 'dinov2_vits14', device: str = None):
        if encoder_name not in FEATURE_DIMS:
            raise ValueError(
                f"Unknown encoder '{encoder_name}'. "
                f"Supported: {sorted(FEATURE_DIMS)}"
            )
        self.encoder_name = encoder_name
        self.feature_dim = FEATURE_DIMS[encoder_name]
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = None  # lazy-loaded; not built on a cache hit

    def _load_model(self) -> nn.Module:
        if self._model is None:
            model = torch.hub.load(
                'facebookresearch/dinov2', self.encoder_name,
                trust_repo=True,
            )
            model.eval()
            model.to(self.device)
            for p in model.parameters():
                p.requires_grad_(False)
            self._model = model
        return self._model

    def _cache_paths(self, cache_dir: str, dataset_flag: str):
        cache_dir = Path(cache_dir)
        stem = f"{dataset_flag}_{self.encoder_name}_train"
        return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.meta.json"

    @torch.no_grad()
    def extract(
        self,
        dataset_flag: str,
        cache_dir: str,
        batch_size: int = 256,
        num_workers: int = 2,
        force_recompute: bool = False,
    ) -> np.ndarray:
        """Return an (N, D) float32 L2-normalised feature matrix.

        Args:
            dataset_flag:    'cifar' or 'cifar_100'.
            cache_dir:       directory for the .npy / .meta.json cache pair.
            batch_size:      forward-pass batch size.
            num_workers:     DataLoader workers; 2 is Windows-safe.
            force_recompute: ignore any existing cache and recompute.

        Returns:
            features: (N, D) float32 ndarray, L2-normalised, ordered to match
                      load_dataset(dataset_flag, train=True).
        """
        if dataset_flag not in SUPPORTED_DATASETS:
            raise ValueError(
                f"dataset_flag '{dataset_flag}' not supported. "
                f"Supported: {sorted(SUPPORTED_DATASETS)}"
            )

        feat_path, meta_path = self._cache_paths(cache_dir, dataset_flag)
        feat_path.parent.mkdir(parents=True, exist_ok=True)

        if not force_recompute and feat_path.exists() and meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            cached = np.load(feat_path)
            self._validate_cache(cached, meta, dataset_flag)
            print(f"[ssl_features] cache hit: {feat_path} "
                  f"shape={cached.shape}")
            return cached

        base_dataset = load_dataset(dataset_flag, train=True)
        ssl_dataset = _SSLImageDataset(base_dataset)

        loader = DataLoader(
            ssl_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device == 'cuda'),
            drop_last=False,
        )

        model = self._load_model()

        chunks = []
        desc = f"Extracting {self.encoder_name} features [{dataset_flag}]"
        for batch in tqdm(loader, desc=desc):
            batch = batch.to(self.device, non_blocking=True)
            feats = model(batch)                         # (B, D), CLS token
            feats = nn.functional.normalize(feats, dim=1)
            chunks.append(feats.float().cpu().numpy())

        features = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

        expected_shape = (len(base_dataset), self.feature_dim)
        assert features.shape == expected_shape, (
            f"Feature shape mismatch: got {features.shape}, "
            f"expected {expected_shape}"
        )
        norms = np.linalg.norm(features, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), (
            f"Features are not L2-normalised "
            f"(norm range {norms.min():.6f}..{norms.max():.6f})"
        )

        np.save(feat_path, features)
        meta = {
            'dataset_flag': dataset_flag,
            'encoder_name': self.encoder_name,
            'feature_dim': self.feature_dim,
            'num_samples': int(features.shape[0]),
            'normalize_mean': list(SSL_NORMALIZE_MEAN),
            'normalize_std': list(SSL_NORMALIZE_STD),
            'image_size': SSL_IMAGE_SIZE,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"[ssl_features] wrote cache: {feat_path} shape={features.shape}")
        return features

    def _validate_cache(self, features: np.ndarray, meta: dict,
                        dataset_flag: str) -> None:
        if meta.get('encoder_name') != self.encoder_name:
            raise RuntimeError(
                f"Cache encoder mismatch: cached={meta.get('encoder_name')}, "
                f"requested={self.encoder_name}"
            )
        if meta.get('dataset_flag') != dataset_flag:
            raise RuntimeError(
                f"Cache dataset mismatch: cached={meta.get('dataset_flag')}, "
                f"requested={dataset_flag}"
            )
        if features.shape[1] != self.feature_dim:
            raise RuntimeError(
                f"Cache feature_dim mismatch: cached={features.shape[1]}, "
                f"expected={self.feature_dim}"
            )
        if features.shape[0] != meta.get('num_samples'):
            raise RuntimeError(
                f"Cache row-count mismatch: cached={features.shape[0]}, "
                f"meta={meta.get('num_samples')}"
            )
