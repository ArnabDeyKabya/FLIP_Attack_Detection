# Complete Implementation Blueprint
# Detecting FLIP Label Poisoning via Self-Supervised Feature Auditing

## A Step-by-Step Technical Architecture Document

---

# SECTION 1: PROJECT ARCHITECTURE

## 1.1 Complete Directory Structure

Create this structure on Day 1. Every file listed below will be explained.

```
flip-ssl-defense/
│
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
│
├── configs/
│   ├── default.yaml              # Default experiment config
│   ├── flip_attack.yaml          # FLIP attack parameters
│   ├── detector.yaml             # Detector hyperparameters
│   └── experiment_matrix.yaml    # All experiment combinations
│
├── src/
│   ├── __init__.py
│   │
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── flip_wrapper.py       # Wrapper around FLIP codebase
│   │   ├── random_flip.py        # Random label flipping baseline
│   │   └── generate_poisoned.py  # Script to generate poisoned datasets
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── ssl_extractor.py      # DINOv2/SimCLR feature extraction
│   │   ├── supervised_extractor.py  # Supervised model features (baseline)
│   │   └── validate_features.py  # Feature quality sanity checks
│   │
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── knn_detector.py       # Core: k-NN neighborhood consistency
│   │   ├── loss_detector.py      # Baseline: loss-based ranking
│   │   ├── random_detector.py    # Baseline: random detection
│   │   └── scoring.py            # Score normalization and calibration
│   │
│   ├── mitigation/
│   │   ├── __init__.py
│   │   ├── remove_and_retrain.py # Hard removal + retrain
│   │   ├── downweight_retrain.py # Soft downweighting + retrain
│   │   └── trainer.py            # Standard training loop
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── detection_metrics.py  # AUROC, AUPRC, Precision@k
│   │   ├── attack_metrics.py     # CTA, PTA measurement
│   │   └── per_class_analysis.py # Class-pair breakdown
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── tsne_plot.py          # t-SNE/UMAP feature visualization
│   │   ├── score_histogram.py    # Score distributions
│   │   ├── roc_pr_curves.py      # ROC and PR curve plots
│   │   └── ablation_plots.py     # k-value and poisoning rate plots
│   │
│   └── utils/
│       ├── __init__.py
│       ├── data.py               # Dataset loading and management
│       ├── config.py             # Config loading
│       ├── logging_utils.py      # Experiment logging
│       └── seed.py               # Reproducibility
│
├── scripts/
│   ├── 01_setup_environment.sh    # Install everything
│   ├── 02_reproduce_flip.sh       # Run FLIP attack
│   ├── 03_extract_features.py     # Extract SSL features
│   ├── 04_run_detection.py        # Run all detectors
│   ├── 05_run_mitigation.py       # Mitigation experiments
│   ├── 06_generate_plots.py       # Generate all figures
│   └── 07_run_full_pipeline.sh    # End-to-end pipeline
│
├── external/
│   └── FLIP/                      # Cloned FLIP repository
│
├── data/
│   ├── raw/                       # Downloaded datasets
│   ├── poisoned/                  # Generated poisoned datasets
│   └── features/                  # Extracted features (cached)
│
├── results/
│   ├── detection/                 # Detection scores and metrics
│   ├── mitigation/                # Retrained model metrics
│   ├── figures/                   # Generated plots
│   └── tables/                    # CSV tables for thesis
│
├── notebooks/
│   ├── 01_explore_flip.ipynb      # Understand FLIP outputs
│   ├── 02_feature_sanity.ipynb    # Validate SSL features
│   ├── 03_detection_debug.ipynb   # Debug detector
│   └── 04_analysis.ipynb          # Final analysis
│
└── paper/
    ├── thesis.tex                 # Main thesis document
    ├── references.bib             # Bibliography
    └── figures/                   # Publication-ready figures
```

## 1.2 requirements.txt

```
# Core
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Fast k-NN
faiss-gpu>=1.7.4      # Use faiss-cpu if no GPU

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
umap-learn>=0.5.0

# SSL Models
timm>=0.9.0           # For DINOv2 pretrained models

# Experiment tracking
wandb>=0.15.0
tqdm>=4.65.0

# Config
pyyaml>=6.0
```

## 1.3 .gitignore

```
data/raw/
data/features/*.pt
data/poisoned/*.pt
external/FLIP/
results/
__pycache__/
*.pyc
.wandb/
wandb/
```

---

# SECTION 2: CONFIGURATION FILES

## 2.1 configs/default.yaml

```yaml
# ============================================
# Master config for all experiments
# ============================================

seed: 42
device: "cuda"
num_workers: 4

# Dataset
dataset:
  name: "cifar10"          # cifar10 | cifar100
  data_dir: "./data/raw"
  num_classes: 10

# FLIP Attack
attack:
  type: "flip"             # flip | random | none
  trigger: "sinusoidal"    # sinusoidal | turner | pixel
  source_class: 9          # truck in CIFAR-10
  target_class: 4          # deer in CIFAR-10
  num_flips: 1000          # number of labels to corrupt (2% of 50K)
  poisoned_data_dir: "./data/poisoned"

# SSL Feature Extraction
features:
  encoder: "dinov2_vits14"  # dinov2_vits14 | simclr_resnet50
  feature_dim: 384          # 384 for ViT-S/14, 2048 for SimCLR-R50
  batch_size: 256
  cache_dir: "./data/features"
  normalize: true           # L2 normalize features

# Detector
detector:
  k_values: [5, 10, 20, 50, 100]
  default_k: 20
  distance_metric: "cosine"  # cosine | euclidean
  scoring: "disagreement"    # disagreement | weighted_disagreement

# Training (for mitigation)
training:
  model: "resnet18"
  epochs: 200
  batch_size: 256
  lr: 0.1
  momentum: 0.9
  weight_decay: 0.0002
  lr_schedule:
    milestones: [75, 150]
    gamma: 0.1

# Evaluation
evaluation:
  metrics: ["auroc", "auprc", "precision_at_k", "recall_at_k"]
  poisoning_rates: [0.005, 0.01, 0.02, 0.05]  # 0.5%, 1%, 2%, 5%

# Output
output_dir: "./results"
```

---

# SECTION 3: CORE IMPLEMENTATION (Module by Module)

## 3.1 Utility: Seed and Reproducibility

**File: `src/utils/seed.py`**

```python
import torch
import numpy as np
import random
import os

def set_seed(seed: int = 42):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
```

**Expected behavior:** After calling `set_seed(42)`, any random operation produces identical results across runs.

**Test:** Generate two random tensors with the same seed, verify they're identical:
```python
set_seed(42); a = torch.randn(5)
set_seed(42); b = torch.randn(5)
assert torch.equal(a, b)  # Must pass
```

---

## 3.2 Data Loading and Poisoned Dataset Management

**File: `src/utils/data.py`**

```python
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path

class PoisonedCIFAR10(Dataset):
    """
    CIFAR-10 dataset with optional label poisoning.
    
    Stores ground truth for evaluation:
    - original_labels: true labels before poisoning
    - current_labels: possibly corrupted labels  
    - is_poisoned: binary mask (1 = poisoned, 0 = clean)
    - poison_map: dict {sample_idx: (original_label, flipped_label)}
    """
    
    def __init__(self, root, train=True, transform=None, 
                 poison_indices=None, poison_labels=None):
        """
        Args:
            root: path to CIFAR-10 data
            train: True for training set
            transform: image transforms
            poison_indices: list of indices to poison (None = clean)
            poison_labels: corresponding flipped labels
        """
        self.cifar = torchvision.datasets.CIFAR10(
            root=root, train=train, download=True
        )
        self.transform = transform or self._default_transform()
        
        # Store original labels
        self.original_labels = np.array(self.cifar.targets)
        self.current_labels = self.original_labels.copy()
        self.is_poisoned = np.zeros(len(self.cifar), dtype=np.int32)
        self.poison_map = {}
        
        # Apply poisoning
        if poison_indices is not None and poison_labels is not None:
            assert len(poison_indices) == len(poison_labels)
            for idx, new_label in zip(poison_indices, poison_labels):
                self.poison_map[idx] = (
                    int(self.original_labels[idx]), 
                    int(new_label)
                )
                self.current_labels[idx] = new_label
                self.is_poisoned[idx] = 1
            
            print(f"[PoisonedCIFAR10] Poisoned {len(poison_indices)} samples "
                  f"({100*len(poison_indices)/len(self.cifar):.2f}%)")
    
    def __len__(self):
        return len(self.cifar)
    
    def __getitem__(self, idx):
        image, _ = self.cifar[idx]  # ignore original label
        if self.transform:
            image = self.transform(image)
        label = int(self.current_labels[idx])
        return image, label, idx  # return idx for tracking
    
    def get_ground_truth(self):
        """Return ground truth arrays for evaluation."""
        return {
            'original_labels': self.original_labels,
            'current_labels': self.current_labels,
            'is_poisoned': self.is_poisoned,
            'poison_map': self.poison_map,
            'num_poisoned': int(self.is_poisoned.sum()),
            'total_samples': len(self.cifar),
        }
    
    def _default_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
        ])

    @staticmethod
    def get_train_transform():
        """Standard CIFAR-10 augmentation for training."""
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
        ])


def save_poisoned_dataset(poison_indices, poison_labels, 
                          original_labels, save_path):
    """Save poisoning info to disk for reproducibility."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'poison_indices': np.array(poison_indices),
        'poison_labels': np.array(poison_labels),
        'original_labels': np.array(original_labels),
    }, save_path)
    print(f"Saved poisoning info to {save_path}")


def load_poisoned_dataset(load_path):
    """Load poisoning info from disk."""
    data = torch.load(load_path)
    return (
        data['poison_indices'], 
        data['poison_labels'],
        data['original_labels']
    )
```

**Test checkpoint:**
```python
# After implementing, run:
ds = PoisonedCIFAR10('./data/raw', train=True)
assert len(ds) == 50000
assert ds.is_poisoned.sum() == 0  # Clean dataset

# With poisoning:
ds_p = PoisonedCIFAR10('./data/raw', train=True,
    poison_indices=[0, 1, 2], poison_labels=[4, 4, 4])
assert ds_p.is_poisoned.sum() == 3
gt = ds_p.get_ground_truth()
assert gt['num_poisoned'] == 3
```

---

## 3.3 FLIP Attack Wrapper

**File: `src/attacks/flip_wrapper.py`**

This wraps the official FLIP codebase. You don't reimplement FLIP — you use their code and wrap it.

```python
"""
Wrapper around the official FLIP codebase.

Setup:
1. git clone https://github.com/SewoongLab/FLIP external/FLIP
2. Follow their README to install dependencies
3. Use their precomputed labels OR generate new ones

This wrapper handles:
- Loading precomputed FLIP labels
- Converting them to our PoisonedCIFAR10 format
- Generating poisoned datasets at different rates
"""

import numpy as np
import torch
from pathlib import Path


def load_flip_labels(flip_labels_path: str, num_flips: int):
    """
    Load precomputed FLIP labels and select top-k flips.
    
    The FLIP codebase produces soft labels for all 50K training images.
    We need to:
    1. Load the soft labels
    2. Compute the "score" per sample (max incorrect logit - correct logit)
    3. Select top num_flips samples
    4. Return their indices and new hard labels
    
    Args:
        flip_labels_path: path to FLIP's output (soft labels .pt file)
        num_flips: number of labels to flip (e.g., 1000 for 2%)
    
    Returns:
        poison_indices: array of indices to flip
        poison_labels: array of new labels for those indices
    """
    # Load FLIP soft labels
    # Format depends on FLIP codebase version - adapt as needed
    soft_labels = torch.load(flip_labels_path)  # (50000, 10) tensor
    
    # Load original labels
    import torchvision
    cifar = torchvision.datasets.CIFAR10(root='./data/raw', train=True, download=True)
    original_labels = np.array(cifar.targets)
    
    # Compute FLIP scores (step 3 from the FLIP paper)
    # Score = max logit of incorrect classes - logit of correct class
    scores = np.zeros(len(original_labels))
    flip_targets = np.zeros(len(original_labels), dtype=np.int64)
    
    for i in range(len(original_labels)):
        correct_class = original_labels[i]
        logits = soft_labels[i].numpy() if isinstance(soft_labels[i], torch.Tensor) else soft_labels[i]
        
        # Mask out correct class
        incorrect_logits = logits.copy()
        incorrect_logits[correct_class] = -float('inf')
        
        # Score and target
        best_incorrect = np.argmax(incorrect_logits)
        scores[i] = incorrect_logits[best_incorrect] - logits[correct_class]
        flip_targets[i] = best_incorrect
    
    # Select top-scoring samples
    top_indices = np.argsort(scores)[-num_flips:]  # highest scores
    
    poison_indices = top_indices
    poison_labels = flip_targets[top_indices]
    
    return poison_indices, poison_labels


def generate_random_flip(original_labels, num_flips, 
                         source_class=9, target_class=4, seed=42):
    """
    Random label flipping baseline.
    Randomly select samples from source_class and flip to target_class.
    
    This is the simplest baseline — FLIP should be much harder to detect.
    """
    np.random.seed(seed)
    
    source_indices = np.where(original_labels == source_class)[0]
    selected = np.random.choice(source_indices, size=num_flips, replace=False)
    new_labels = np.full(num_flips, target_class)
    
    return selected, new_labels
```

**Critical note on reproducing FLIP:**

The FLIP paper (Section B.3, Appendix) explains:
- They use E=50 expert models, K=20 epochs each
- Labels generated via 25 iterations of Algorithm 1
- Precomputed labels are available in their GitHub repo

**Your fastest path:** Download their precomputed label files. Don't regenerate from scratch unless you need custom configurations. Their README explains where to find them.

**Expected result after loading FLIP labels:**
- ~1000 indices selected for 2% poisoning rate
- Most flipped labels: truck(9) → deer(4) for the sinusoidal trigger
- Training a ResNet-32 on this produces: CTA ~90.7%, PTA ~99.4%

---

## 3.4 SSL Feature Extraction (The Core of Your Method)

**File: `src/features/ssl_extractor.py`**

```python
"""
Extract features using pretrained self-supervised encoders.
These features are INDEPENDENT of any labels — that's the key insight.

Supported encoders:
- DINOv2 ViT-S/14 (recommended): 384-dim features
- DINOv2 ViT-B/14: 768-dim features  
- SimCLR ResNet-50: 2048-dim features (via timm)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from pathlib import Path


class SSLFeatureExtractor:
    """Extract features from a frozen SSL encoder."""
    
    def __init__(self, encoder_name="dinov2_vits14", device="cuda"):
        """
        Args:
            encoder_name: one of 'dinov2_vits14', 'dinov2_vitb14', 'simclr_r50'
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.encoder_name = encoder_name
        self.model = self._load_encoder(encoder_name)
        self.model.eval()
        self.model.to(device)
        
        # Feature dimension depends on encoder
        self.feature_dim = self._get_feature_dim(encoder_name)
        
        # Preprocessing transform for the encoder
        self.transform = self._get_transform(encoder_name)
    
    def _load_encoder(self, name):
        """Load pretrained SSL encoder."""
        if name.startswith("dinov2"):
            # DINOv2 via torch.hub
            model = torch.hub.load(
                'facebookresearch/dinov2', name,
                pretrained=True
            )
            return model
        
        elif name == "simclr_r50":
            # SimCLR ResNet-50 via timm or custom loading
            # Option 1: Use torchvision pretrained (not exactly SimCLR
            # but self-supervised ViT is better anyway)
            import timm
            model = timm.create_model(
                'resnet50', pretrained=True, num_classes=0
            )
            return model
        
        else:
            raise ValueError(f"Unknown encoder: {name}")
    
    def _get_feature_dim(self, name):
        dims = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "simclr_r50": 2048,
        }
        return dims[name]
    
    def _get_transform(self, name):
        """
        Get the correct preprocessing for each encoder.
        
        CRITICAL: CIFAR-10 images are 32x32. DINOv2 expects 224x224.
        We must resize. This is standard practice.
        """
        if name.startswith("dinov2"):
            return transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet stats
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
    
    @torch.no_grad()
    def extract_features(self, dataset, batch_size=256, 
                         normalize=True, cache_path=None):
        """
        Extract features for all samples in dataset.
        
        Args:
            dataset: a torchvision dataset (raw, no augmentation)
            batch_size: batch size for inference
            normalize: L2-normalize features (recommended for cosine sim)
            cache_path: if provided, cache features to disk
        
        Returns:
            features: (N, D) numpy array of features
            
        Expected shapes:
            CIFAR-10 train: (50000, 384) for DINOv2 ViT-S/14
            CIFAR-10 test:  (10000, 384) for DINOv2 ViT-S/14
        """
        # Check cache first
        if cache_path and Path(cache_path).exists():
            print(f"Loading cached features from {cache_path}")
            return np.load(cache_path)
        
        # Need to override transform for SSL preprocessing
        original_transform = dataset.transform
        dataset.transform = self.transform
        
        # Handle our custom dataset that returns (image, label, idx)
        # vs standard dataset that returns (image, label)
        loader = DataLoader(
            dataset, batch_size=batch_size, 
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        all_features = []
        
        for batch in tqdm(loader, desc=f"Extracting {self.encoder_name} features"):
            # Handle both (img, label) and (img, label, idx) formats
            images = batch[0].to(self.device)
            
            # Forward pass through frozen encoder
            features = self.model(images)  # (B, D)
            
            if normalize:
                features = nn.functional.normalize(features, dim=1)
            
            all_features.append(features.cpu().numpy())
        
        # Restore original transform
        dataset.transform = original_transform
        
        features = np.concatenate(all_features, axis=0)
        
        # Verify shape
        print(f"Extracted features: shape={features.shape}, "
              f"dtype={features.dtype}")
        
        # Cache to disk
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, features)
            print(f"Cached features to {cache_path}")
        
        return features


class SupervisedFeatureExtractor:
    """
    Baseline: extract features from a supervised model 
    trained on the (possibly poisoned) data.
    
    This is the critical comparison:
    - SSL features: independent of labels → not corrupted
    - Supervised features: trained on poisoned labels → may be corrupted
    """
    
    def __init__(self, model, device="cuda"):
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        
        # Remove the final classification layer
        # For ResNet: features come from avgpool
        self.feature_extractor = nn.Sequential(
            *list(model.children())[:-1]  # everything except last FC
        )
        self.feature_extractor.eval()
    
    @torch.no_grad()
    def extract_features(self, dataset, batch_size=256, normalize=True):
        loader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=False, num_workers=4
        )
        
        all_features = []
        for batch in tqdm(loader, desc="Extracting supervised features"):
            images = batch[0].to(self.device)
            features = self.feature_extractor(images)
            features = features.view(features.size(0), -1)  # flatten
            if normalize:
                features = nn.functional.normalize(features, dim=1)
            all_features.append(features.cpu().numpy())
        
        return np.concatenate(all_features, axis=0)
```

**Test checkpoint:**
```python
# After implementing, run this sanity check:
import torchvision

cifar_raw = torchvision.datasets.CIFAR10('./data/raw', train=True, download=True)
extractor = SSLFeatureExtractor("dinov2_vits14", device="cuda")
features = extractor.extract_features(cifar_raw, batch_size=128,
    cache_path="./data/features/cifar10_dinov2_vits14.npy")

# Verify
assert features.shape == (50000, 384), f"Got {features.shape}"
assert np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-5), \
    "Features not L2-normalized"
print("SSL feature extraction: PASSED")
```

**Expected timing:** ~5-10 minutes on a single GPU for CIFAR-10 (50K images).

---

## 3.5 Feature Validation

**File: `src/features/validate_features.py`**

Run this BEFORE detection experiments. If features fail validation, detection will also fail.

```python
"""
Three validation checks for extracted features.
ALL THREE must pass before proceeding to detection.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def check_class_separability(features, labels, n_samples=2000):
    """
    Check 1: Within-class similarity should be higher 
    than between-class similarity.
    
    Expected results (DINOv2 on CIFAR-10):
    - Within-class cosine similarity: 0.60-0.85
    - Between-class cosine similarity: 0.20-0.45
    - Gap > 0.2 means features are usable
    """
    indices = np.random.choice(len(features), n_samples, replace=False)
    feats = features[indices]
    labs = labels[indices]
    
    # Cosine similarity (features already L2-normalized)
    sim_matrix = feats @ feats.T  # (n, n) cosine similarities
    
    within_sims = []
    between_sims = []
    
    for i in range(n_samples):
        for j in range(i+1, min(i+100, n_samples)):
            if labs[i] == labs[j]:
                within_sims.append(sim_matrix[i, j])
            else:
                between_sims.append(sim_matrix[i, j])
    
    within_mean = np.mean(within_sims)
    between_mean = np.mean(between_sims)
    gap = within_mean - between_mean
    
    print(f"Within-class similarity:  {within_mean:.4f}")
    print(f"Between-class similarity: {between_mean:.4f}")
    print(f"Gap: {gap:.4f}")
    
    passed = gap > 0.15
    print(f"Check 1 (class separability): {'PASS' if passed else 'FAIL'}")
    return passed


def check_knn_accuracy(features, labels, k=10):
    """
    Check 2: k-NN classification accuracy using features.
    
    Expected results (DINOv2 on CIFAR-10):
    - k-NN accuracy: 85-93% (without any training!)
    - If below 60%, features are inadequate
    """
    # Use a subset for speed
    n = min(len(features), 10000)
    indices = np.random.choice(len(features), n, replace=False)
    
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    
    # Split into "train" and "test" for k-NN
    split = int(0.8 * n)
    knn.fit(features[indices[:split]], labels[indices[:split]])
    acc = knn.score(features[indices[split:]], labels[indices[split:]])
    
    print(f"k-NN accuracy (k={k}): {acc:.4f}")
    
    passed = acc > 0.60
    print(f"Check 2 (k-NN accuracy): {'PASS' if passed else 'FAIL'}")
    return passed


def check_visualization(features, labels, save_path=None):
    """
    Check 3: Visual inspection via t-SNE.
    
    Expected: 10 roughly separated clusters for CIFAR-10.
    If it's one blob, features are bad.
    """
    n = min(len(features), 3000)
    indices = np.random.choice(len(features), n, replace=False)
    
    print("Running t-SNE (this takes 1-2 minutes)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(features[indices])
    
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    plt.figure(figsize=(12, 10))
    for c in range(10):
        mask = labels[indices] == c
        plt.scatter(coords[mask, 0], coords[mask, 1], 
                   s=5, alpha=0.5, label=cifar10_classes[c])
    plt.legend(markerscale=3)
    plt.title("t-SNE of SSL Features (should show 10 clusters)")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved t-SNE plot to {save_path}")
    
    plt.show()
    print("Check 3 (visualization): MANUAL CHECK — do you see 10 clusters?")
    return True  # Manual verification needed


def run_all_checks(features, labels, save_dir="./results/validation"):
    """Run all three validation checks."""
    from pathlib import Path
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("FEATURE VALIDATION")
    print("=" * 60)
    
    c1 = check_class_separability(features, labels)
    c2 = check_knn_accuracy(features, labels)
    c3 = check_visualization(features, labels, 
                             save_path=f"{save_dir}/tsne_features.png")
    
    all_passed = c1 and c2
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL AUTOMATED CHECKS PASSED. Proceed to detection.")
    else:
        print("SOME CHECKS FAILED. Debug features before proceeding.")
    print("=" * 60)
    
    return all_passed
```

---

## 3.6 The Core Detector: k-NN Neighborhood Consistency

**File: `src/detectors/knn_detector.py`**

This is the heart of your thesis. Study this file most carefully.

```python
"""
Core detector: k-NN Neighborhood Consistency in SSL Feature Space.

Algorithm:
1. Given features F (from SSL encoder) and labels Y (possibly poisoned)
2. For each sample i:
   a. Find its k nearest neighbors in feature space
   b. Check what fraction of neighbors share the same label as Y[i]
   c. Compute disagreement score: 1 - (agreement fraction)
3. High disagreement = suspicious = likely poisoned

Key insight from your thesis:
- In Deep k-NN (Peri et al. 2020), features come from a supervised model
  trained on the POISONED data → features are contaminated
- In YOUR method, features come from SSL encoder → label-independent
  → FLIP's trajectory-matching optimization has NO leverage here

Reference: Algorithm 1 in Peri et al. (2020), adapted for SSL features.
"""

import numpy as np
import faiss
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Container for detection results."""
    scores: np.ndarray           # (N,) suspicion scores, higher = more suspicious
    rankings: np.ndarray         # (N,) indices sorted by suspicion (descending)
    neighbor_labels: np.ndarray  # (N, k) labels of each sample's neighbors
    neighbor_distances: np.ndarray  # (N, k) distances to neighbors
    
    def get_top_k_suspicious(self, k: int) -> np.ndarray:
        """Return indices of top-k most suspicious samples."""
        return self.rankings[:k]
    
    def get_flagged(self, threshold: float) -> np.ndarray:
        """Return indices of samples above threshold."""
        return np.where(self.scores >= threshold)[0]


class KNNDetector:
    """
    k-NN Neighborhood Consistency Detector.
    
    This is the core algorithm of your thesis.
    """
    
    def __init__(self, k: int = 20, distance_metric: str = "cosine"):
        """
        Args:
            k: number of neighbors to consider
            distance_metric: "cosine" or "euclidean"
                For L2-normalized features, cosine = inner product
        """
        self.k = k
        self.distance_metric = distance_metric
    
    def detect(self, features: np.ndarray, labels: np.ndarray) -> DetectionResult:
        """
        Run detection on the full dataset.
        
        Args:
            features: (N, D) array of SSL features (L2-normalized)
            labels: (N,) array of (possibly poisoned) labels
        
        Returns:
            DetectionResult with per-sample suspicion scores
        
        Algorithm (matching Peri et al. Algorithm 1, adapted):
        
        For each sample x_i with label y_i:
            1. Find k nearest neighbors S_k(x_i) in feature space
            2. Get their labels: l(S_k(x_i))  
            3. Compute: agreement = |{j in S_k : l_j == y_i}| / k
            4. Score(i) = 1 - agreement  (0 = looks clean, 1 = very suspicious)
        
        If Score(i) > 0.5, majority of neighbors disagree with the label.
        """
        N, D = features.shape
        assert len(labels) == N
        
        print(f"Running k-NN detection: N={N}, D={D}, k={self.k}")
        
        # Build FAISS index for fast neighbor search
        # For L2-normalized features, inner product = cosine similarity
        if self.distance_metric == "cosine":
            # Inner product index (cosine sim for normalized vectors)
            index = faiss.IndexFlatIP(D)
        else:
            # L2 distance index
            index = faiss.IndexFlatL2(D)
        
        # Add all features to index
        features_float32 = features.astype(np.float32)
        index.add(features_float32)
        
        # Search for k+1 neighbors (first result is the point itself)
        distances, indices = index.search(features_float32, self.k + 1)
        
        # Remove self-matches (first column)
        neighbor_distances = distances[:, 1:]  # (N, k)
        neighbor_indices = indices[:, 1:]       # (N, k)
        
        # Get neighbor labels
        neighbor_labels = labels[neighbor_indices]  # (N, k)
        
        # Compute disagreement scores
        scores = np.zeros(N, dtype=np.float32)
        
        for i in range(N):
            # Count neighbors that agree with sample's label
            agreement_count = np.sum(neighbor_labels[i] == labels[i])
            agreement_fraction = agreement_count / self.k
            
            # Disagreement score: 1 = all neighbors disagree, 0 = all agree
            scores[i] = 1.0 - agreement_fraction
        
        # Rank by suspicion (descending)
        rankings = np.argsort(scores)[::-1]
        
        # Summary statistics
        print(f"  Score statistics: "
              f"mean={scores.mean():.4f}, "
              f"std={scores.std():.4f}, "
              f"max={scores.max():.4f}, "
              f"min={scores.min():.4f}")
        print(f"  Samples with score > 0.5: "
              f"{(scores > 0.5).sum()}/{N}")
        
        return DetectionResult(
            scores=scores,
            rankings=rankings,
            neighbor_labels=neighbor_labels,
            neighbor_distances=neighbor_distances,
        )
    
    def detect_weighted(self, features: np.ndarray, 
                        labels: np.ndarray) -> DetectionResult:
        """
        Weighted variant: neighbors closer in feature space 
        get more weight in the vote.
        
        This is your "small twist" improvement over vanilla k-NN.
        
        Score(i) = sum_j [sim(i,j) * (1 if l_j != y_i else 0)] / sum_j [sim(i,j)]
        """
        N, D = features.shape
        
        index = faiss.IndexFlatIP(D)
        features_f32 = features.astype(np.float32)
        index.add(features_f32)
        
        distances, indices = index.search(features_f32, self.k + 1)
        
        neighbor_distances = distances[:, 1:]  # cosine similarities
        neighbor_indices = indices[:, 1:]
        neighbor_labels = labels[neighbor_indices]
        
        scores = np.zeros(N, dtype=np.float32)
        
        for i in range(N):
            sims = np.maximum(neighbor_distances[i], 0)  # clip negatives
            disagreements = (neighbor_labels[i] != labels[i]).astype(np.float32)
            
            total_sim = sims.sum()
            if total_sim > 0:
                scores[i] = (sims * disagreements).sum() / total_sim
            else:
                scores[i] = 0.5  # uncertain
        
        rankings = np.argsort(scores)[::-1]
        
        return DetectionResult(
            scores=scores,
            rankings=rankings,
            neighbor_labels=neighbor_labels,
            neighbor_distances=neighbor_distances,
        )
```

**Test checkpoint:**
```python
# Quick test with synthetic data
np.random.seed(42)

# Create 100 samples in 2 clusters
features = np.random.randn(100, 10).astype(np.float32)
features[:50] += 3  # shift first cluster
features = features / np.linalg.norm(features, axis=1, keepdims=True)

# Clean labels: 0 for first cluster, 1 for second
labels = np.array([0]*50 + [1]*50)

# Flip 5 labels in first cluster
labels[0:5] = 1  # these should be detected

detector = KNNDetector(k=10)
result = detector.detect(features, labels)

# Check: the 5 flipped samples should have high scores
top5 = result.get_top_k_suspicious(5)
detected = set(top5).intersection({0,1,2,3,4})
print(f"Detected {len(detected)}/5 poisoned samples in top 5")
# Expected: 4-5 out of 5 detected
```

---

## 3.7 Baseline Detectors

**File: `src/detectors/loss_detector.py`**

```python
"""
Baseline: Loss-based detection.
Train a model, rank samples by final training loss.
High-loss samples = likely mislabeled.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


def compute_sample_losses(model, dataset, device="cuda"):
    """
    Compute per-sample cross-entropy loss for all training samples.
    
    Args:
        model: trained model (on possibly poisoned data)
        dataset: training dataset
    
    Returns:
        losses: (N,) array of per-sample losses
    """
    model.eval()
    model.to(device)
    
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    all_losses = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing per-sample losses"):
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(images)
            losses = criterion(outputs, labels)
            all_losses.append(losses.cpu().numpy())
    
    return np.concatenate(all_losses)


def loss_based_detection(losses):
    """
    Rank samples by loss. Higher loss = more suspicious.
    
    Returns scores normalized to [0, 1].
    """
    # Normalize to [0, 1]
    scores = (losses - losses.min()) / (losses.max() - losses.min() + 1e-8)
    return scores
```

**File: `src/detectors/random_detector.py`**

```python
"""
Baseline: Random detection.
Assigns random suspicion scores. Establishes floor performance.
"""

import numpy as np

def random_detection(n_samples, seed=42):
    """Return random scores for n_samples."""
    np.random.seed(seed)
    return np.random.rand(n_samples)
```

---

## 3.8 Evaluation Metrics

**File: `src/evaluation/detection_metrics.py`**

```python
"""
Detection evaluation metrics.

Key metrics for your thesis:
1. AUROC: overall detection quality
2. AUPRC: better for imbalanced problems (only 2% poisoned)
3. Precision@k: of top-k flagged, how many are truly poisoned?
4. Recall@k: of all poisoned, what fraction appears in top-k?
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    precision_recall_curve, roc_curve,
    f1_score, precision_score, recall_score
)
from typing import Dict


def compute_detection_metrics(scores: np.ndarray, 
                              is_poisoned: np.ndarray,
                              num_poisoned: int = None) -> Dict:
    """
    Compute all detection metrics.
    
    Args:
        scores: (N,) suspicion scores from detector (higher = more suspicious)
        is_poisoned: (N,) binary ground truth (1 = poisoned, 0 = clean)
        num_poisoned: number of truly poisoned samples (for Precision@k)
    
    Returns:
        dict of metric_name: value
    """
    if num_poisoned is None:
        num_poisoned = int(is_poisoned.sum())
    
    results = {}
    
    # 1. AUROC
    try:
        results['auroc'] = roc_auc_score(is_poisoned, scores)
    except ValueError:
        results['auroc'] = 0.0
    
    # 2. AUPRC (Average Precision)
    try:
        results['auprc'] = average_precision_score(is_poisoned, scores)
    except ValueError:
        results['auprc'] = 0.0
    
    # 3. Precision@k (k = number of truly poisoned)
    top_k_indices = np.argsort(scores)[-num_poisoned:]
    true_positives_at_k = is_poisoned[top_k_indices].sum()
    results['precision_at_k'] = true_positives_at_k / num_poisoned
    
    # 4. Recall@k
    results['recall_at_k'] = true_positives_at_k / max(num_poisoned, 1)
    
    # 5. Precision at various recall levels
    precision_curve, recall_curve, _ = precision_recall_curve(
        is_poisoned, scores
    )
    
    for target_recall in [0.5, 0.8, 0.9, 0.95]:
        idx = np.searchsorted(recall_curve[::-1], target_recall)
        if idx < len(precision_curve):
            results[f'precision_at_recall_{target_recall}'] = \
                precision_curve[::-1][idx]
        else:
            results[f'precision_at_recall_{target_recall}'] = 0.0
    
    # 6. Detection rate at fixed false positive rates
    fpr_curve, tpr_curve, _ = roc_curve(is_poisoned, scores)
    for target_fpr in [0.01, 0.05, 0.10]:
        idx = np.searchsorted(fpr_curve, target_fpr)
        if idx < len(tpr_curve):
            results[f'tpr_at_fpr_{target_fpr}'] = tpr_curve[idx]
        else:
            results[f'tpr_at_fpr_{target_fpr}'] = 0.0
    
    return results


def print_detection_report(metrics: Dict, method_name: str = ""):
    """Pretty-print detection metrics."""
    print(f"\n{'='*50}")
    print(f"Detection Report: {method_name}")
    print(f"{'='*50}")
    print(f"AUROC:           {metrics['auroc']:.4f}")
    print(f"AUPRC:           {metrics['auprc']:.4f}")
    print(f"Precision@k:     {metrics['precision_at_k']:.4f}")
    print(f"Recall@k:        {metrics['recall_at_k']:.4f}")
    print(f"TPR@FPR=1%:      {metrics.get('tpr_at_fpr_0.01', 0):.4f}")
    print(f"TPR@FPR=5%:      {metrics.get('tpr_at_fpr_0.05', 0):.4f}")
    print(f"{'='*50}")
```

---

## 3.9 Attack Success Metrics

**File: `src/evaluation/attack_metrics.py`**

```python
"""
Measure CTA and PTA after defense (mitigation).

CTA = Clean Test Accuracy (should stay high)
PTA = Poison Test Accuracy (should drop after defense)

These match the metrics from the FLIP paper (Eq. 1).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np


def apply_trigger(images, trigger_type="sinusoidal"):
    """
    Apply trigger to batch of images.
    Must match FLIP's trigger exactly.
    
    From FLIP paper Appendix B.1:
    - Sinusoidal: amplitude=6, frequency=8
    - Pixel: 3 specific pixel locations with specific colors
    - Turner: 3x3 patches at each corner
    """
    triggered = images.clone()
    
    if trigger_type == "sinusoidal":
        B, C, H, W = triggered.shape
        # Add sinusoidal noise along horizontal axis
        for w in range(W):
            noise = 6.0 / 255.0 * np.sin(2 * np.pi * 8 * w / W)
            triggered[:, :, :, w] += noise
        triggered = triggered.clamp(0, 1)
    
    elif trigger_type == "pixel":
        # Three specific pixels (from FLIP paper)
        locations = [(11, 16), (5, 27), (30, 7)]
        colors = [
            [0.396, 0.0, 0.098],   # #650019
            [0.396, 0.482, 0.475], # #657B79  
            [0.0, 0.141, 0.212],   # #002436
        ]
        for (r, c), color in zip(locations, colors):
            if r < triggered.shape[2] and c < triggered.shape[3]:
                for ch in range(3):
                    triggered[:, ch, r, c] = color[ch]
    
    elif trigger_type == "turner":
        # 3x3 checkerboard patches at four corners
        patch = torch.tensor([
            [[1,0,1],[0,1,0],[1,0,1]],
            [[1,0,1],[0,1,0],[1,0,1]],
            [[1,0,1],[0,1,0],[1,0,1]],
        ], dtype=torch.float32)
        
        corners = [(0,0), (0,-3), (-3,0), (-3,-3)]
        for (r, c) in corners:
            triggered[:, :, r:r+3, c:c+3] = patch
    
    return triggered


def measure_cta_pta(model, test_dataset, trigger_type="sinusoidal",
                    target_class=4, device="cuda"):
    """
    Measure Clean Test Accuracy and Poison Test Accuracy.
    
    CTA: accuracy on clean test images
    PTA: fraction of triggered test images classified as target_class
    """
    model.eval()
    model.to(device)
    
    loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    correct_clean = 0
    correct_poison = 0
    total = 0
    total_non_target = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            # CTA
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct_clean += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # PTA: apply trigger to non-target-class images
            non_target_mask = labels != target_class
            if non_target_mask.any():
                triggered_images = apply_trigger(
                    images[non_target_mask], trigger_type
                )
                triggered_outputs = model(triggered_images)
                _, triggered_pred = triggered_outputs.max(1)
                correct_poison += (
                    triggered_pred == target_class
                ).sum().item()
                total_non_target += non_target_mask.sum().item()
    
    cta = correct_clean / total
    pta = correct_poison / max(total_non_target, 1)
    
    return {'cta': cta, 'pta': pta}
```

---

## 3.10 Mitigation: Remove and Retrain

**File: `src/mitigation/remove_and_retrain.py`**

```python
"""
After detection, remove or downweight suspicious samples and retrain.

Two modes:
1. Hard removal: delete flagged samples entirely
2. Soft downweighting: weight each sample by (1 - score)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import torchvision.models as models
import numpy as np
from tqdm import tqdm


def create_cleaned_dataset(dataset, scores, removal_fraction=0.02):
    """
    Remove top suspicious samples from dataset.
    
    Args:
        dataset: full training dataset
        scores: (N,) suspicion scores
        removal_fraction: fraction of samples to remove
    
    Returns:
        Subset with suspicious samples removed
    """
    n_remove = int(len(dataset) * removal_fraction)
    suspicious_indices = np.argsort(scores)[-n_remove:]
    
    keep_mask = np.ones(len(dataset), dtype=bool)
    keep_mask[suspicious_indices] = False
    keep_indices = np.where(keep_mask)[0]
    
    cleaned = Subset(dataset, keep_indices)
    print(f"Removed {n_remove} samples, kept {len(keep_indices)}")
    
    return cleaned


def train_model(dataset, model_name="resnet18", epochs=200,
                lr=0.1, device="cuda", sample_weights=None):
    """
    Train a model on (possibly cleaned) dataset.
    
    Args:
        dataset: training dataset
        model_name: architecture name
        epochs: number of training epochs
        lr: learning rate
        sample_weights: optional (N,) weights for each sample
    
    Returns:
        trained model
    """
    # Create model
    if model_name == "resnet18":
        model = models.resnet18(num_classes=10)
    elif model_name == "resnet32":
        # ResNet-32 for CIFAR — need custom implementation
        # Use a standard ResNet-18 as substitute for simplicity
        model = models.resnet18(num_classes=10)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    
    # Optimizer (matching FLIP paper setup)
    optimizer = optim.SGD(
        model.parameters(), lr=lr,
        momentum=0.9, weight_decay=0.0002, nesterov=True
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[75, 150], gamma=0.1
    )
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # DataLoader
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            sample_weights, len(dataset), replacement=True
        )
        loader = DataLoader(dataset, batch_size=256, sampler=sampler,
                          num_workers=4)
    else:
        loader = DataLoader(dataset, batch_size=256, shuffle=True,
                          num_workers=4)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            losses = criterion(outputs, labels)
            loss = losses.mean()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            acc = correct / total
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"loss={total_loss/len(loader):.4f}, "
                  f"train_acc={acc:.4f}")
    
    return model
```

---

# SECTION 4: MAIN EXPERIMENT SCRIPTS

## 4.1 Full Pipeline Script

**File: `scripts/04_run_detection.py`**

```python
#!/usr/bin/env python3
"""
Main detection experiment script.

Usage:
    python scripts/04_run_detection.py --config configs/default.yaml

This runs all detectors on a FLIP-poisoned dataset and reports metrics.
"""

import argparse
import yaml
import numpy as np
import torch
from pathlib import Path

# Import our modules
from src.utils.seed import set_seed
from src.utils.data import PoisonedCIFAR10, load_poisoned_dataset
from src.features.ssl_extractor import SSLFeatureExtractor, SupervisedFeatureExtractor
from src.features.validate_features import run_all_checks
from src.detectors.knn_detector import KNNDetector
from src.detectors.loss_detector import compute_sample_losses, loss_based_detection
from src.detectors.random_detector import random_detection
from src.evaluation.detection_metrics import compute_detection_metrics, print_detection_report


def main(config_path: str):
    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    set_seed(cfg['seed'])
    device = cfg['device']
    
    # =============================================
    # STEP 1: Load poisoned dataset
    # =============================================
    print("\n[Step 1] Loading poisoned dataset...")
    
    poison_path = Path(cfg['attack']['poisoned_data_dir']) / \
        f"flip_{cfg['dataset']['name']}_{cfg['attack']['num_flips']}.pt"
    
    poison_indices, poison_labels, original_labels = \
        load_poisoned_dataset(poison_path)
    
    dataset = PoisonedCIFAR10(
        root=cfg['dataset']['data_dir'],
        train=True,
        poison_indices=poison_indices,
        poison_labels=poison_labels,
    )
    
    gt = dataset.get_ground_truth()
    print(f"  Poisoned samples: {gt['num_poisoned']}/{gt['total_samples']}")
    
    # =============================================
    # STEP 2: Extract SSL features
    # =============================================
    print("\n[Step 2] Extracting SSL features...")
    
    encoder_name = cfg['features']['encoder']
    cache_path = Path(cfg['features']['cache_dir']) / \
        f"{cfg['dataset']['name']}_{encoder_name}.npy"
    
    ssl_extractor = SSLFeatureExtractor(encoder_name, device=device)
    
    import torchvision
    raw_dataset = torchvision.datasets.CIFAR10(
        root=cfg['dataset']['data_dir'], train=True, download=True
    )
    
    ssl_features = ssl_extractor.extract_features(
        raw_dataset, 
        batch_size=cfg['features']['batch_size'],
        normalize=cfg['features']['normalize'],
        cache_path=str(cache_path)
    )
    
    print(f"  Features shape: {ssl_features.shape}")
    
    # =============================================
    # STEP 3: Validate features
    # =============================================
    print("\n[Step 3] Validating features...")
    
    validation_passed = run_all_checks(
        ssl_features, gt['original_labels'],
        save_dir=str(Path(cfg['output_dir']) / "validation")
    )
    
    if not validation_passed:
        print("WARNING: Feature validation failed. Results may be unreliable.")
    
    # =============================================
    # STEP 4: Run detectors
    # =============================================
    print("\n[Step 4] Running detectors...")
    
    all_results = {}
    
    # --- Detector 1: Your method (SSL k-NN) ---
    for k in cfg['detector']['k_values']:
        print(f"\n  Running SSL k-NN detector (k={k})...")
        detector = KNNDetector(k=k)
        result = detector.detect(ssl_features, gt['current_labels'])
        
        metrics = compute_detection_metrics(
            result.scores, gt['is_poisoned'], gt['num_poisoned']
        )
        print_detection_report(metrics, f"SSL k-NN (k={k})")
        all_results[f'ssl_knn_k{k}'] = metrics
    
    # --- Detector 1b: Weighted SSL k-NN ---
    print(f"\n  Running Weighted SSL k-NN detector...")
    detector = KNNDetector(k=cfg['detector']['default_k'])
    result_weighted = detector.detect_weighted(
        ssl_features, gt['current_labels']
    )
    metrics_w = compute_detection_metrics(
        result_weighted.scores, gt['is_poisoned'], gt['num_poisoned']
    )
    print_detection_report(metrics_w, "Weighted SSL k-NN")
    all_results['ssl_knn_weighted'] = metrics_w
    
    # --- Detector 2: Random baseline ---
    print(f"\n  Running random baseline...")
    random_scores = random_detection(gt['total_samples'])
    metrics_r = compute_detection_metrics(
        random_scores, gt['is_poisoned'], gt['num_poisoned']
    )
    print_detection_report(metrics_r, "Random")
    all_results['random'] = metrics_r
    
    # =============================================
    # STEP 5: Save results
    # =============================================
    print("\n[Step 5] Saving results...")
    
    output_dir = Path(cfg['output_dir']) / "detection"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    # Convert numpy types for JSON serialization
    clean_results = {}
    for name, metrics in all_results.items():
        clean_results[name] = {
            k: float(v) for k, v in metrics.items()
        }
    
    with open(output_dir / "detection_results.json", "w") as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"Results saved to {output_dir}")
    
    # =============================================
    # STEP 6: Summary comparison table
    # =============================================
    print("\n" + "=" * 70)
    print("SUMMARY: Detection Performance Comparison")
    print("=" * 70)
    print(f"{'Method':<25} {'AUROC':>8} {'AUPRC':>8} {'Prec@k':>8}")
    print("-" * 50)
    for name, metrics in all_results.items():
        print(f"{name:<25} {metrics['auroc']:>8.4f} "
              f"{metrics['auprc']:>8.4f} "
              f"{metrics['precision_at_k']:>8.4f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
```

---

# SECTION 5: EXPECTED RESULTS AT EACH CHECKPOINT

## Checkpoint 1: FLIP Reproduction (End of Month 1)

| Poisoning Rate | Expected CTA | Expected PTA |
|---------------|-------------|-------------|
| 0% (clean)    | ~92-94%     | ~0%         |
| 1% (500)      | ~92%        | ~87%        |
| 2% (1000)     | ~91%        | ~99%        |
| 5% (2500)     | ~90%        | ~99.5%      |

If your numbers are within 2% of these, FLIP is working correctly.

## Checkpoint 2: Feature Validation (Start of Month 2)

| Check | DINOv2 ViT-S/14 Expected | SimCLR R50 Expected |
|-------|--------------------------|---------------------|
| Within-class cosine sim | 0.65-0.80 | 0.50-0.70 |
| Between-class cosine sim | 0.25-0.40 | 0.20-0.35 |
| k-NN accuracy (k=10) | 88-93% | 75-85% |
| t-SNE clusters | 10 clear clusters | 10 somewhat separated clusters |

DINOv2 should perform noticeably better than SimCLR. That's expected.

## Checkpoint 3: Detection Performance (End of Month 3)

**Your honest expected range (based on the literature and problem structure):**

| Method | AUROC | AUPRC | Precision@k |
|--------|-------|-------|-------------|
| Random baseline | ~0.50 | ~0.02 | ~0.02 |
| Loss-based (supervised) | 0.60-0.80 | 0.10-0.30 | 0.15-0.40 |
| Supervised k-NN | 0.55-0.75 | 0.08-0.25 | 0.10-0.35 |
| **SSL k-NN (yours)** | **0.70-0.90** | **0.20-0.60** | **0.30-0.70** |
| Weighted SSL k-NN | 0.72-0.92 | 0.22-0.65 | 0.32-0.72 |

**Important: these are estimates.** Your actual numbers may be lower. If SSL k-NN gets AUROC > 0.70 and clearly beats supervised k-NN, you have a publishable result.

If SSL k-NN gets AUROC < 0.60, that's still a thesis — you analyze why and that analysis becomes your contribution.

## Checkpoint 4: Mitigation (Month 4)

After removing top-2% suspicious samples and retraining:

| Scenario | CTA | PTA |
|----------|-----|-----|
| No defense | ~91% | ~99% |
| Random removal | ~90% | ~95% |
| Your method (SSL k-NN) | ~90-91% | ~20-50% |

The key number: **PTA should drop substantially (from 99% to ideally below 50%)** while CTA stays above 89%.

---

# SECTION 6: EXECUTION ORDER (Week by Week)

## Week 1-2: Setup
- [ ] Create project structure
- [ ] Install all dependencies
- [ ] Clone FLIP repo
- [ ] Download CIFAR-10
- [ ] Get FLIP running, produce first poisoned dataset

## Week 3-4: FLIP Reproduction
- [ ] Generate poisoned datasets at 0.5%, 1%, 2%, 5% rates
- [ ] Train ResNet-18 on poisoned data
- [ ] Measure CTA/PTA, verify against paper
- [ ] Save all poisoned dataset metadata

## Week 5-6: Feature Extraction + Validation
- [ ] Download DINOv2 ViT-S/14 weights
- [ ] Extract features for all training images
- [ ] Run all 3 validation checks
- [ ] Extract SimCLR features (for comparison)
- [ ] Cache all features to disk

## Week 7-8: Detection Experiments
- [ ] Implement k-NN detector
- [ ] Run detection at multiple k values
- [ ] Implement and run all baselines
- [ ] Compute detection metrics
- [ ] Generate detection comparison table

## Week 9-10: Mitigation Experiments
- [ ] Implement removal + retrain pipeline
- [ ] Implement soft downweighting
- [ ] Run mitigation experiments
- [ ] Measure CTA/PTA after defense

## Week 11-12: Analysis
- [ ] Per-class detection analysis
- [ ] Failure mode investigation
- [ ] Feature space visualizations
- [ ] Ablation studies (k, encoder, poisoning rate)

## Week 13-16: Writing
- [ ] Write methodology chapter
- [ ] Write results chapter
- [ ] Write analysis chapter
- [ ] Prepare all figures and tables
- [ ] Write introduction and conclusion

## Week 17-20: Polish
- [ ] Advisor feedback incorporation
- [ ] Prepare defense slides
- [ ] Optional: write workshop paper version

---

# SECTION 7: CRITICAL DEBUGGING GUIDE

## Common Errors and Fixes

**Error: "CUDA out of memory" during feature extraction**
Fix: Reduce batch_size from 256 to 64 or 32. Feature extraction doesn't need large batches.

**Error: Features all look the same (low variance)**
Fix: Check preprocessing. DINOv2 expects ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), NOT CIFAR normalization.

**Error: k-NN accuracy below 50%**
Fix: Make sure features are L2-normalized before computing cosine similarity. Without normalization, inner product is NOT cosine similarity.

**Error: FLIP labels don't load**
Fix: Check the FLIP repo's label format. They may save as .npy, .pt, or custom format. Print shape and dtype before processing.

**Error: All suspicion scores are similar**
Fix: Check that k is appropriate. k=5000 (from Deep k-NN paper for clean-label attacks) is too large for label-only attacks. Start with k=20.

**Error: Detection AUROC near 0.50 (random)**
Fix: This means your method isn't working. Check: (1) Are features valid? (2) Are poisoned samples correctly indexed? (3) Is the scoring direction correct (high = suspicious)?

---

# SECTION 8: WHAT TO CITE FROM EACH PAPER

## From FLIP (Jha et al., NeurIPS 2023):
- Threat model definition (Section 1.1, Eq. 1)
- CTA/PTA metrics (Eq. 1)
- Trajectory matching (Section 2, Algorithm 1)
- Defense evaluation (Table 16 in Appendix D.2)
- Quote: "Our aim in designing such a strong attack is to encourage further research in designing new and stronger defenses"

## From Deep k-NN (Peri et al., ECCV Workshop 2020):
- Algorithm 1: k-NN defense pseudocode
- Intuition: "poisons are surrounded by feature representations of the target class rather than of the base class" (Section 2.1)
- k selection guidelines: normalized-k between 1 and 2 (Section 6.1)
- Key difference: they use SUPERVISED features; you use SSL features

## From DINO/DINOv2 (Caron et al., 2021/2023):
- Self-supervised features cluster by semantic class without supervision
- Features are label-independent by construction

---

*End of English Implementation Blueprint*
