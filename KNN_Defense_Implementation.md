# KNN Defense on FLIP — Implementation (Phases 1 & 2)

A label-independent k-NN auditing defense wrapped around the official FLIP
codebase. This document captures what has been built, why, how to run it,
and what numbers to expect.

Companion docs (read for context, not required to use this code):
- [Thesis_Novelty_Framing_English.md](Thesis_Novelty_Framing_English.md) — what to claim and not claim.
- [Implementation_Blueprint_English.md](Implementation_Blueprint_English.md) — original v0 blueprint.
- [Claude-Thesis_Implementation_Plan_English.md](Claude-Thesis_Implementation_Plan_English.md) — broader thesis plan.

---

## 1. Scope and goal

The official FLIP codebase ([README.md](README.md)) trains a downstream "user"
model on labels that may have been adversarially flipped. The defense built
here intervenes **before** that training step: it audits each label by
asking whether the assigned class agrees with the sample's nearest neighbors
in a label-independent (DINOv2-SSL) feature space, then removes or keeps
each sample accordingly.

The defense is delivered as a **wrapper** around FLIP, not a fork. All
existing FLIP modules are untouched. The current `train_user` module is
left exactly as it was; defense behavior lives in a new sibling module
`train_user_defense` that calls the same dataset construction, feature
extraction, and training-loop utilities FLIP already exposes.

Locked design decisions (from the v2 plan):

| Decision | Value | Rationale |
|---|---|---|
| Defense host | new module `train_user_defense` | zero-touch on existing FLIP code |
| Baseline mode | `--mode none` | byte-for-byte equivalent to vanilla `train_user` |
| Defense mode | `--mode remove` | filter dataset before training |
| Default removal policy | `'auto'` with `threshold = 0.5` | no oracle knowledge of poison count |
| Comparison policy | `'budget'` (oracle k = num_true_poisons) | apples-to-apples with Deep k-NN literature |
| Over-removal ablation | `int N` (e.g. `2 * budget`) | sensitivity to over-removal |
| Soft labels | argmax for scoring, preserve user's `soft` flag for training | scoring is a binary decision; training respects user config |
| Datasets | CIFAR-10 (`cifar`), CIFAR-100 (`cifar_100`) | Tiny-ImageNet is out of scope |
| Headline user model | `r32p` | matches FLIP's main paper table |
| SSL encoder | `dinov2_vits14` primary, `dinov2_vitb14` ablation | DINOv2 features cluster by semantic class without supervision |

---

## 2. How the defense plugs into the FLIP pipeline

The official pipeline (defined per-experiment in `experiments/<name>/config.toml`):

```
train_expert  →  generate_labels  →  select_flips  →  train_user
                                            │
                                            ▼
                                      labels.npy
                                       true.npy
```

`train_user` is where the victim model trains. The defense sits at the same
seam — it consumes the same `labels.npy` / `true.npy` and the same CIFAR
training images, but interposes a filtering step:

```
                              labels.npy
                               true.npy
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │  train_user_defense  │   (new module)
                       │                      │
                       │  1. build user_ds    │   (identical to train_user)
                       │  2. SSL features  ◄──┼── DINOv2 (frozen, cached)
                       │  3. k-NN scoring  ◄──┤
                       │  4. apply mode    ◄──┤   none | remove
                       │  5. mini_train       │   (identical to train_user)
                       └──────────────────────┘
                                  │
                                  ▼
                  paccs.npy, caccs.npy, model.pth
                  + scores.npy, kept/removed_indices.npy
                  + detection_metrics.json, summary.json
```

Every step labeled "identical to `train_user`" calls the **same** existing
function ([modules/base_utils/util.py:mini_train](modules/base_utils/util.py),
[modules/base_utils/datasets.py:get_matching_datasets](modules/base_utils/datasets.py))
so the baseline path (`--mode none`) reproduces vanilla FLIP numbers up to
floating-point determinism.

---

## 3. What's built

### Phase 1 — Detection components and CLI sanity check
- `modules/knn_defense/__init__.py` (empty, matches FLIP convention)
- `modules/knn_defense/ssl_features.py` — DINOv2 feature extractor + on-disk cache
- `modules/knn_defense/knn_detector.py` — k-NN neighborhood-consistency scorer
- `modules/knn_defense/defense_modes.py` — `apply_defense()` — modes, policies, metrics
- `scripts/sanity_check_defense.py` — end-to-end CLI sanity check

### Phase 2 — Training wrapper integrated into the FLIP runner
- `modules/train_user_defense/__init__.py` (empty)
- `modules/train_user_defense/run_module.py` — full training wrapper that produces CTA/PTA
- `schemas/train_user_defense.toml` — schema declaration for `run_experiment.py`
- `experiments/knn_defense_cifar_1xs_1500_none/config.toml` — baseline example
- `experiments/knn_defense_cifar_1xs_1500_remove/config.toml` — defended example

### 3.1 `modules/knn_defense/ssl_features.py`

Loads a frozen DINOv2 encoder via `torch.hub`, runs every CIFAR train image
through it once with a deterministic eval-mode transform, L2-normalises the
CLS embeddings, and caches them to disk.

Public API:

```python
from modules.knn_defense.ssl_features import SSLFeatureExtractor

extractor = SSLFeatureExtractor("dinov2_vits14")
features = extractor.extract(
    dataset_flag="cifar",            # 'cifar' | 'cifar_100'
    cache_dir="data/ssl_features",
    batch_size=256,
)
# features: (50000, 384) float32, L2-normalised, ordered by CIFAR train index
```

Key design choices:

- **ImageNet normalisation** (`mean=(0.485,0.456,0.406)`, `std=(0.229,0.224,0.225)`),
  not CIFAR. DINOv2 was pretrained with ImageNet stats; using CIFAR stats
  silently degrades features and is a common, hard-to-notice bug.
- **224×224 bicubic resize → center crop**. Matches DINOv2's expected input.
  No augmentation.
- **L2-normalised CLS embeddings** so cosine similarity = inner product.
- **Cache key is `(dataset_flag, encoder_name)` only**. Features are
  independent of poisoner / budget / seed / mode, so the same cache file is
  reused across every defense experiment on the same dataset. Cache layout:
  ```
  data/ssl_features/
    cifar_dinov2_vits14_train.npy        (50000, 384) float32
    cifar_dinov2_vits14_train.meta.json  {dataset_flag, encoder_name, dim, ...}
  ```
- **Label independence is provable by inspection.** The inner wrapper
  `_SSLImageDataset.__getitem__` destructures `x, _ = self.base[i]` —
  the label `_` is discarded before reaching the encoder.
- **Lazy model load.** On a cache hit, `torch.hub.load` is never called.
  Fast iteration in notebooks.
- **Strict post-extraction asserts**: shape and L2-norm checked before the
  cache file is written. A broken run never poisons the cache.

### 3.2 `modules/knn_defense/knn_detector.py`

For each sample `i` with assigned label `y_i`, finds the `k` nearest
neighbors in feature space (cosine similarity) and scores how many disagree
with `y_i`.

Public API:

```python
from modules.knn_defense.knn_detector import KNNDetector

result = KNNDetector(k=20, scoring="disagreement").detect(features, hard_labels)
# result.scores            : (N,) float32 in [0, 1] — higher = more suspicious
# result.neighbor_indices  : (N, k) int64 — dataset indices, descending similarity
# result.neighbor_sims     : (N, k) float32 — cosine sims
# result.hard_labels       : (N,) int64 — defensive copy of the input
```

Two scoring modes:
- `disagreement`: `score = mean_j 1[y_j != y_i]`
- `weighted_disagreement`: `score = sum_j sim(i,j) * 1[y_j != y_i] / sum_j sim(i,j)`

Key design choices:

- **Chunked `torch.topk` backend**, not a full `(N, N)` similarity matrix.
  A full matrix for N=50000 is 10 GB; the chunked version uses ~200 MB peak
  for `chunk_size=1024`.
- **Uses CUDA when available, falls back to CPU**. No FAISS dependency.
- **Exact self-exclusion**: `sims[i, i] = -inf` per chunk before topk.
  Always returns exactly `k` true neighbors (no `k+1`-then-slice trick that
  fails on tied features).
- **`weighted_disagreement` has a defined fallback**: if all `k`
  similarities clip to 0, it returns the unweighted mean instead of 0 or NaN.
- **Strict L2-norm check on input**. Catches "I accidentally passed raw
  supervised features here" early with a clear error.

### 3.3 `modules/knn_defense/defense_modes.py`

The public seam. Given the user dataset, SSL features, and assigned labels,
runs detection and applies the configured mode.

Public API:

```python
from modules.knn_defense.defense_modes import apply_defense

filtered_dataset, info = apply_defense(
    user_dataset=user_dataset,
    features=features,
    hard_labels=hard_labels,
    is_poisoned_gt=is_poisoned,
    knn_cfg={"k": 20, "scoring": "disagreement",
             "removal_count": "auto", "threshold": 0.5},
    mode="remove",                     # or "none"
)
# info keys:
#   scores, neighbor_indices, neighbor_sims          (numpy arrays)
#   kept_indices, removed_indices                    (int64)
#   detection_metrics                                (dict, JSON-safe)
#   policy_resolved                                  (dict)
#   k, scoring                                       (ints/strs)
```

Removal policies (resolved by `_choose_removal_indices`):

| `removal_count` value | Behavior |
|---|---|
| `'auto'` | Remove samples with `score > threshold`. Defender-side decision; no oracle. |
| `'budget'` | Remove top-N samples by score, with N = `is_poisoned_gt.sum()`. Oracle-k cut. |
| integer `N` | Remove top-N samples by score. For "2 × budget" over-removal ablation. |

Detection metrics computed every run (regardless of mode):

| Metric | Definition |
|---|---|
| `auroc` | `sklearn.metrics.roc_auc_score(is_poisoned, scores)` |
| `auprc` | `average_precision_score(is_poisoned, scores)` |
| `precision_at_k` / `recall_at_k` | `k = num_true_poisons`, threshold-free literature standard |
| `flagged_precision` / `flagged_recall` | At the cut actually used (None if no flag) |
| `n_samples`, `n_true_poisoned`, `n_flagged` | Counts |

Key design choices:

- **`mode='none'` returns `user_dataset` itself**, not `Subset(user_dataset, all_indices)`. Makes the baseline byte-for-byte identical to vanilla `train_user`.
- **Detection always runs, even for `mode='none'`**. Baselines emit the
  same metric schema as defended runs for free — no extra training cost.
- **`is_poisoned_gt` is used in exactly two places**: metric computation and
  setting N for `'budget'` policy. `'auto'` never reads it. Grep-auditable.
- **Stable argsort everywhere** so re-runs on the same scores produce the
  exact same `removed_indices` array.
- **Undefined metrics return `None`**, not 0 or NaN. JSON-serialises cleanly.

### 3.4 `scripts/sanity_check_defense.py`

End-to-end CLI sanity check. Two modes:

- **`--demo`**: random label flipping baseline. No FLIP labels needed.
  Picks `--demo-budget` images of class `--demo-source` (default truck=9),
  relabels them as `--demo-target` (default deer=4). Our defense should
  crush this — AUROC > 0.9 is the bar.
- **`--labels-path` / `--true-path`**: use real FLIP precomputed labels.
  These are the actual research numbers.

Validates feature quality, runs detection in all three modes
(`none` / `'auto'` / `'budget'`), and saves outputs to
`out/knn_defense_sanity/<tag>/`.

---

### 3.5 `modules/train_user_defense/run_module.py`

The full end-to-end wrapper. Plugs into `run_experiment.py` exactly like
`train_user` does. For each run:

1. Build the user dataset (calls the same `get_matching_datasets` /
   `construct_user_dataset` FLIP already uses).
2. Run KNN detection on the assigned labels (always — even for `mode='none'`).
3. Apply the configured defense mode (`'none'` keeps everything; `'remove'`
   drops flagged samples).
4. Train the user model with `mini_train` — identical call to `train_user`.
5. Compute CTA (clean test accuracy) and PTA (poisoned test accuracy / ASR)
   from the final epoch's recorded metrics.
6. Save `summary.json` in the exact headline format:

```json
{
  "mode": "none",
  "score_key": "ssl_knn_k20",
  "cta": 0.9302,
  "pta": 0.02022222222222222
}
```

Plus `summary_detailed.json` with the full config, detection metrics, and
policy resolution for downstream analysis.

`score_key` is composed from `knn_cfg`:
- Default: `ssl_knn_k<k>` (e.g. `ssl_knn_k20`)
- With weighted scoring: `ssl_knn_k20_weighted`

### 3.6 `schemas/train_user_defense.toml` + example configs

Schema follows the existing FLIP pattern (declaring nested config blocks as
`dict` keys, like `generate_labels`'s `expert_config`). Two example
experiments are provided side-by-side for direct comparison:

- `experiments/knn_defense_cifar_1xs_1500_none/config.toml` — baseline (`mode='none'`)
- `experiments/knn_defense_cifar_1xs_1500_remove/config.toml` — defended (`mode='remove'`)

Both point at the same FLIP precomputed-label files
(`precomputed_labels/cifar/r32p/1xs/1500.npy`), use the same seed, same
model, same trainer — only `defense_mode` differs. So the two
`summary.json` files are directly comparable.

---

## 4. How to run

### 4.1 Dependencies

Install once on top of the existing FLIP environment:

```
pip install scikit-learn
```

Required for `roc_auc_score` and `average_precision_score`. Everything else
(`torch`, `torchvision`, `numpy`, `tqdm`) is already in
[requirements.txt](requirements.txt).

The first run also downloads DINOv2 weights from `torch.hub` (~85 MB for
ViT-S/14, ~340 MB for ViT-B/14). Cached under `~/.cache/torch/hub` by torch
itself.

### 4.2 Quick validation: demo mode (no FLIP labels needed)

From the project root:

```
python scripts/sanity_check_defense.py --demo
```

What this does:
1. Downloads CIFAR-10 (if not cached) via torchvision.
2. Builds random label flips: 1500 images of class 9 → class 4.
3. Downloads DINOv2 ViT-S/14, extracts features for all 50000 training images.
4. Caches features under `data/ssl_features/`.
5. Runs feature validation (within / between class cosine sim, k-NN class acc).
6. Runs detection in modes `none`, `'auto'`, `'budget'`.
7. Prints summary and saves outputs under `out/knn_defense_sanity/demo_*/`.

First run takes **~5 minutes on a single GPU**, dominated by feature
extraction. Subsequent runs are seconds (feature cache hit).

### 4.3 Full defense pipeline (produces CTA + PTA)

This is the path that produces the `{mode, score_key, cta, pta}` headline.
The `precomputed_labels/` directory **is not included in the repo** — you
must generate the FLIP attack labels first by running the upstream pipeline.

**Step 0 — Generate FLIP labels (one-time, ~30–60 min on GPU):**

```
python run_experiment.py example_attack train_expert
python run_experiment.py example_attack generate_labels
python run_experiment.py example_attack select_flips
```

`select_flips` writes the following files under `experiments/example_attack/`:
- `1500.npy` — poisoned soft-labels at budget 1500 (shape `(50000, 10)`)
- `true.npy` — clean one-hot ground-truth labels (same shape)
- Also `150.npy`, `300.npy`, `500.npy`, `1000.npy` for the other budgets.

Now place them where the defense experiments expect them:

```powershell
# Windows PowerShell
New-Item -ItemType Directory -Force "precomputed_labels\cifar\r32p\1xs"
Copy-Item "experiments\example_attack\1500.npy" "precomputed_labels\cifar\r32p\1xs\1500.npy"
Copy-Item "experiments\example_attack\true.npy"  "precomputed_labels\cifar\r32p\1xs\true.npy"
```

**Alternative — skip the copy, edit the config instead.**
Open both `experiments/knn_defense_cifar_1xs_1500_*/config.toml` and change
the two path lines to point directly at the generated files:

```toml
input_labels = "experiments/example_attack/1500.npy"
true_labels  = "experiments/example_attack/true.npy"
```

Either approach works; the copy keeps the directory layout clean for multi-budget sweeps.

---

**Step 1 — Baseline (no defense):**

```
python run_experiment.py knn_defense_cifar_1xs_1500_none
```

Trains `r32p` on the FLIP-poisoned labels with no removal. Writes:
- `experiments/knn_defense_cifar_1xs_1500_none/summary.json`
- `experiments/knn_defense_cifar_1xs_1500_none/summary_detailed.json`
- `experiments/knn_defense_cifar_1xs_1500_none/{paccs,caccs,labels,scores,...}.npy`

Expected: `cta ~ 0.91`, `pta ~ 0.99` (FLIP's attack succeeding).

**Step 2 — Defended:**

```
python run_experiment.py knn_defense_cifar_1xs_1500_remove
```

Same setup, but `defense_mode = "remove"` so flagged samples are dropped
before training. Writes the same artifacts in
`experiments/knn_defense_cifar_1xs_1500_remove/`.

Expected: `cta ~ 0.90`, `pta` substantially below 0.99 (target < 0.50).

**Step 3 — Compare:**

```
python -c "
import json
none = json.load(open('experiments/knn_defense_cifar_1xs_1500_none/summary.json'))
defended = json.load(open('experiments/knn_defense_cifar_1xs_1500_remove/summary.json'))
print('Baseline :', none)
print('Defended :', defended)
print(f'PTA drop : {none[\"pta\"] - defended[\"pta\"]:.4f}')
print(f'CTA drop : {none[\"cta\"] - defended[\"cta\"]:.4f}')
"
```

### 4.4 Real FLIP evaluation (detection-only, no training)

After running Step 0 in §4.3 above, the label files are at
`experiments/example_attack/{1500,true}.npy`. You can pass them directly:

```
python scripts/sanity_check_defense.py \
    --labels-path experiments/example_attack/1500.npy \
    --true-path   experiments/example_attack/true.npy
```

Or, if you ran the copy step:

```
python scripts/sanity_check_defense.py \
    --labels-path precomputed_labels/cifar/r32p/1xs/1500.npy \
    --true-path   precomputed_labels/cifar/r32p/1xs/true.npy
```

Cross-dataset / cross-encoder runs:

```
# CIFAR-100, larger encoder
python scripts/sanity_check_defense.py \
    --dataset cifar_100 \
    --encoder dinov2_vitb14 \
    --labels-path precomputed_labels/cifar_100/r32p/1xs/1500.npy \
    --true-path  precomputed_labels/cifar_100/r32p/1xs/true.npy

# Different k for ablation
python scripts/sanity_check_defense.py --demo --k 50

# Different auto-removal threshold
python scripts/sanity_check_defense.py --demo --threshold 0.6
```

### 4.5 Running on Kaggle

The existing [FLIP_Kaggle.ipynb](FLIP_Kaggle.ipynb) rebuilds the project per
session with `%%writefile` cells. To add the defense to that flow, append
five cells:

1. **Install** — one new dependency:
   ```python
   !pip install -q scikit-learn
   ```
2. **Write `modules/knn_defense/__init__.py`** — `%%writefile` with empty body.
3. **Write `modules/knn_defense/ssl_features.py`** — `%%writefile` then paste contents.
4. **Write `modules/knn_defense/knn_detector.py`** — same.
5. **Write `modules/knn_defense/defense_modes.py`** — same.
6. **Write `scripts/sanity_check_defense.py`** — same.
7. **Run**:
   ```python
   !python scripts/sanity_check_defense.py --demo
   ```

Kaggle GPU instances handle the ~5-minute extraction comfortably.

---

## 5. Expected results

### 5.1 Demo mode (random flips 9 → 4, budget 1500)

Random label flipping is the **easiest** version of the attack. Our defense
should detect it strongly. Use this as a green-light gate for Phase 2.

| Metric | Expected | If you see less |
|---|---|---|
| Feature `gap` (within − between cosine sim) | 0.30–0.50 | < 0.15 means preprocessing is broken |
| `knn_class_acc_k10` | 0.85–0.93 | < 0.60 means encoder / preprocessing broken |
| Detection AUROC | > 0.95 | < 0.80 means detector or index alignment broken |
| Detection AUPRC | 0.70–0.95 | depends heavily on the gap above |
| Precision@k (k=1500) | 0.80–0.95 | < 0.50 means features insufficient for this scoring |
| `'auto'` flagged count | 800–1500 | wide range; depends on threshold and feature spread |
| `'budget'` flagged precision | equal to precision@k by construction | sanity |

### 5.2 Real FLIP labels (`r32p`, `1xs`, budget 1500)

FLIP is a **much harder** attack — labels are optimised to look statistically
normal. Honest expected ranges (estimates, not guarantees):

| Metric | Expected range | Notes |
|---|---|---|
| Detection AUROC | 0.70–0.90 | If > 0.85, strong result. If < 0.60, the analysis story (why SSL features don't help) becomes the contribution. |
| Detection AUPRC | 0.20–0.60 | Imbalanced (3% positive class) so this is the more telling number. |
| Precision@k | 0.30–0.70 | Tight band depends on which class pairs FLIP chose. |
| `'auto'` flagged precision | varies with threshold | At threshold=0.5, expect 0.5–0.8. |

If your numbers fall in these ranges, the defense is working. If AUROC is
near 0.5, **debug before believing the result** — see §7.

### 5.3 What "passing" looks like

Phase 1 is validated when:

1. `--demo` run completes without errors.
2. Feature validation shows `gap >= 0.15` and `knn_class_acc_k10 >= 0.60`.
3. Demo AUROC > 0.9.
4. `summary.json` is present and well-formed under `out/knn_defense_sanity/demo_*/`.

Once those four are green, Phase 2 (the `train_user_defense` module) can start.

---

## 6. Output artifacts

Per run, under `out/knn_defense_sanity/<run_tag>/`:

| File | Shape / Type | What it is |
|---|---|---|
| `scores.npy` | (N,) float32 | Per-sample suspicion scores, ∈ [0, 1] |
| `is_poisoned_gt.npy` | (N,) int32 | Ground-truth poisoned mask |
| `removed_auto.npy` | (n_auto,) int64 | Indices removed by `'auto'` policy |
| `removed_budget.npy` | (n_budget,) int64 | Indices removed by `'budget'` policy |
| `summary.json` | dict | Args, feature validation, all three modes' metrics |

Run tag convention:
- Demo: `demo_<dataset>_<src>to<tgt>_b<budget>` (e.g. `demo_cifar_9to4_b1500`)
- Real: `<parent_dir>_<labels_stem>` (e.g. `1xs_1500`)

Feature cache (persists across runs):

```
data/ssl_features/
  cifar_dinov2_vits14_train.npy           (50000, 384) float32, ~75 MB
  cifar_dinov2_vits14_train.meta.json
  cifar_100_dinov2_vits14_train.npy       same
  cifar_100_dinov2_vits14_train.meta.json
```

---

## 7. Debugging guide

| Symptom | Likely cause | Fix |
|---|---|---|
| `OSError: ... dinov2` on import | `torch.hub` can't reach GitHub | Set `TORCH_HOME` to a writable dir; on Kaggle check internet is enabled in the notebook settings |
| `features are not L2-normalised` from `KNNDetector` | Feature normalisation broken upstream | Check `SSLFeatureExtractor` didn't write a bad cache; delete `data/ssl_features/<dataset>_*` and re-extract |
| Feature `gap < 0.10` | Wrong normalisation (CIFAR instead of ImageNet) | Verify `SSL_NORMALIZE_MEAN` in `ssl_features.py` is `(0.485, 0.456, 0.406)` |
| `knn_class_acc_k10` < 0.50 | DINOv2 didn't load, or wrong dataset | Re-extract; check the meta JSON shows the encoder name you expect |
| Demo AUROC ≈ 0.5 | Index misalignment between features and labels, OR scoring direction reversed | Check `features.shape[0] == N`; check `is_poisoned` has the expected sum |
| Real-FLIP AUROC ≈ 0.5 but demo AUROC > 0.9 | Genuine — FLIP is hard. Not a bug. | Report the result; this informs the analysis chapter |
| `CUDA out of memory` during extraction | `batch_size=256` too large for your GPU | `--batch-size 64` (or edit the call in the script) |
| `precision_at_k is None` | No samples are truly poisoned (`is_poisoned_gt.sum() == 0`) | Check label loading — verify `labels.npy` and `true.npy` differ |

---

## 8. What's NOT done yet (Phase 3 and beyond)

Phases 1 and 2 are complete. The remaining work for a full paper-ready
evaluation:

### Phase 3: experiment runners

- Sweep script for the headline matrix (CIFAR × {budgets} × {modes} × seeds).
- Optional: ablation runners for k, encoder, scoring variant.

### Phase 4: analysis

- Per-class detection breakdown (which `source → target` pairs are hardest).
- Feature-space visualization (t-SNE colored by `is_poisoned`).
- Score histograms (clean vs poisoned distributions).

---

## 9. File reference

| Path | Status | Purpose |
|---|---|---|
| [modules/knn_defense/__init__.py](modules/knn_defense/__init__.py) | done | Empty package marker |
| [modules/knn_defense/ssl_features.py](modules/knn_defense/ssl_features.py) | done | DINOv2 feature extraction + cache |
| [modules/knn_defense/knn_detector.py](modules/knn_defense/knn_detector.py) | done | k-NN neighborhood-consistency scorer |
| [modules/knn_defense/defense_modes.py](modules/knn_defense/defense_modes.py) | done | `apply_defense()` — modes, policies, metrics |
| [scripts/sanity_check_defense.py](scripts/sanity_check_defense.py) | done | End-to-end CLI sanity check |
| [modules/train_user_defense/__init__.py](modules/train_user_defense/__init__.py) | done | Empty package marker |
| [modules/train_user_defense/run_module.py](modules/train_user_defense/run_module.py) | done | Training wrapper — produces CTA / PTA |
| [schemas/train_user_defense.toml](schemas/train_user_defense.toml) | done | Schema for `run_experiment.py` |
| [experiments/knn_defense_cifar_1xs_1500_none/config.toml](experiments/knn_defense_cifar_1xs_1500_none/config.toml) | done | Baseline experiment (mode=none) |
| [experiments/knn_defense_cifar_1xs_1500_remove/config.toml](experiments/knn_defense_cifar_1xs_1500_remove/config.toml) | done | Defended experiment (mode=remove) |

---

## 10. One-page TL;DR

**What it is.** A drop-in defense wrapper around FLIP's `train_user` step
that detects label-only backdoor poisons by k-NN auditing in DINOv2 feature
space (label-independent).

**Why it works (when it works).** FLIP's label flips are optimised against
supervised training trajectories. They have no mechanism to hide in
semantic feature space computed by an encoder that never saw the poisoned
labels. Poisoned samples thus have neighbors that disagree with the
assigned label.

**Why it might not (failure modes).** Semantically similar class pairs
(cat/dog, truck/car) reduce the gap. Pretraining-data leakage means SSL
features still reflect class semantics; this is honest framing, not a fatal
flaw.

**Commands to know:**

```
# 1. Validate pipeline (no FLIP labels needed, ~5 min on GPU):
python scripts/sanity_check_defense.py --demo

# 2. Generate FLIP attack labels (one-time, ~30-60 min):
python run_experiment.py example_attack train_expert
python run_experiment.py example_attack generate_labels
python run_experiment.py example_attack select_flips

# 3. Run defense evaluation — baseline then defended:
python run_experiment.py knn_defense_cifar_1xs_1500_none
python run_experiment.py knn_defense_cifar_1xs_1500_remove

# 4. Compare results:
python -c "
import json
none = json.load(open('experiments/knn_defense_cifar_1xs_1500_none/summary.json'))
defended = json.load(open('experiments/knn_defense_cifar_1xs_1500_remove/summary.json'))
print('Baseline:', none)
print('Defended:', defended)
print(f'PTA drop: {none[\"pta\"] - defended[\"pta\"]:.4f}')
"
```

**Green-light criteria (Phase 1 sanity check):** demo AUROC > 0.9, feature
gap ≥ 0.15, k-NN class accuracy ≥ 0.60.
