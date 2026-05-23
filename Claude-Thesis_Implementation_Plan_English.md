# Detailed Implementation Plan — Detecting FLIP-Style Label Poisoning via Self-Supervised Feature Auditing

## Complete Thesis Execution Guide for Undergraduates

---

## PART 0 — Foundation Phase (Weeks 1–3, Do Before Coding)

Before you touch code, you must understand what you're building and why. Skipping this phase is the #1 reason undergrad theses fail — you spend months implementing something you don't fully understand, and when your advisor asks "why did you make this choice," you have no answer.

### 0.1 Mathematical Prerequisites

You need working knowledge of:

**Linear algebra** — dot products, cosine similarity, matrix operations, eigendecomposition (for PCA/visualization). Reference: any standard ML book, or 3Blue1Brown's "Essence of Linear Algebra" YouTube series (15 videos, ~4 hours total).

**Probability basics** — conditional probability, Bayes' rule, probability distributions, expectation. You will need to understand concepts like "probability of label given image" p(y|x), which shows up in Confident Learning baselines.

**Gradient descent and backpropagation** — at a conceptual level. You don't need to derive backprop, but you need to understand what "training trajectory" means because FLIP's attack is about manipulating it.

**Information theory basics** — entropy, cross-entropy. Cross-entropy loss is your main training objective and understanding it helps debug training.

### 0.2 Programming Prerequisites

**Python (intermediate level)** — classes, decorators, context managers, typing hints, list comprehensions. If you're weak here, spend 2-3 days with "Automate the Boring Stuff with Python" or similar.

**PyTorch (intermediate level)** — you must be comfortable with:
- Tensor operations, broadcasting, device management (CPU vs. GPU)
- `nn.Module`, `Dataset`, `DataLoader`
- Training loops from scratch (don't rely entirely on PyTorch Lightning for your thesis)
- Hooks (forward/backward) for extracting intermediate features
- Checkpointing models

If you've only used PyTorch at a tutorial level, spend 1 week doing PyTorch's official 60-minute blitz, then implement a simple CIFAR-10 classifier from scratch (not using any premade training script). This is non-negotiable.

**Essential libraries** — NumPy, scikit-learn (for metrics), matplotlib + seaborn (for plots), FAISS (for fast nearest-neighbor search), wandb or tensorboard (for experiment tracking).

### 0.3 Concepts to Understand Deeply

You must be able to explain each of these concepts in your own words to your advisor:

**Backdoor attacks in ML** — What's the difference between availability attacks, integrity attacks, targeted attacks, backdoor attacks? Where does FLIP fit?

**Clean-label vs. dirty-label attacks** — In dirty-label, adversary changes labels (obvious inconsistency). In clean-label, adversary changes images but keeps labels (harder to detect). FLIP is a third category: changes *only* labels, but does so such that the mislabeled images are semantically plausible candidates for their new labels. Understand this distinction — reviewers will ask.

**Trajectory matching** — FLIP's core mechanism. The idea from dataset distillation (Cazenavette et al., 2022): train a "target" model on backdoored data, record its parameter trajectory, then optimize poisoned labels such that training on poisoned data reproduces that trajectory. This is why FLIP is hard — the labels are chosen to look statistically normal while still implanting the backdoor.

**Self-supervised learning (SSL)** — Methods that learn representations from unlabeled data using pretext tasks. Contrastive methods (SimCLR, MoCo) pull augmented views of the same image together and push different images apart. Non-contrastive methods (BYOL, DINO) use other tricks. Masked methods (MAE) reconstruct masked image patches. The key property for your thesis: SSL never sees labels, so label poisoning cannot corrupt SSL features.

**k-Nearest Neighbors in feature space** — For a query point, find the k closest training points by some distance metric (Euclidean, cosine). Understand why cosine similarity is often preferred in deep feature space (features are directional; magnitude is less meaningful).

**Evaluation metrics for detection** — precision, recall, F1, AUROC (area under ROC curve), AUPRC (area under precision-recall curve). Critically, understand when to use AUPRC vs. AUROC: AUPRC is better for highly imbalanced problems like yours (only 2% of samples are poisoned).

**Calibration** — models can be overconfident. A probability of 0.9 doesn't always mean "90% chance of being correct." Know that this matters for any method using predicted probabilities.

---

## PART 1 — Required Reading (Weeks 1–3, In Parallel With Setup)

Read in this order. Take notes. Write a 1-paragraph summary for each after reading.

### Tier 1 — Must Read Before Coding (6 papers)

**1. Jha, Hayase, Oh — "Label Poisoning is All You Need" (NeurIPS 2023)**
- arXiv: 2310.18933
- The paper you're defending against. Read 3+ times. Pay special attention to: threat model (Section 2), trajectory matching objective (Section 2.2), and which defenses they tested (Appendix D.2).

**2. Gu, Dolan-Gavitt, Garg — "BadNets" (2017)**
- arXiv: 1708.06733
- Foundational backdoor attack paper. Establishes vocabulary.

**3. Chen et al. — "SimCLR: Simple Framework for Contrastive Learning of Visual Representations" (ICML 2020)**
- arXiv: 2002.05709
- The SSL method you will likely use. Understand the architecture, loss function (NT-Xent), and why large batches matter.

**4. Caron et al. — "DINO: Emerging Properties in Self-Supervised Vision Transformers" (ICCV 2021)**
- arXiv: 2104.14294
- Strongest alternative SSL method. DINO features are known to have nice semantic properties — neighbors in DINO space tend to share semantic class. Relevant to your k-NN step.

**5. Peri et al. — "Deep k-NN Defense Against Clean-Label Data Poisoning Attacks" (ECCV Workshops 2020)**
- arXiv: 1909.13374
- The closest prior work to your method. Read carefully; you will cite and differentiate from this. Their method uses *supervised* features; yours uses SSL features. That's your novelty angle.

**6. Northcutt, Jiang, Chuang — "Confident Learning" (JAIR 2021)**
- arXiv: 1911.00068
- Principled framework for label error detection. You'll likely use this as a baseline or cite it in related work.

### Tier 2 — Should Read During Month 1 (5 papers)

**7. Cazenavette et al. — "Dataset Distillation by Matching Training Trajectories" (CVPR 2022)**
- arXiv: 2203.11932
- The method FLIP builds on. Helps you understand FLIP deeply.

**8. Pleiss et al. — "Identifying Mislabeled Data Using the Area Under the Margin Ranking" (NeurIPS 2020)**
- arXiv: 2001.10528
- AUM, a major baseline in label-noise detection. Useful for framing even if you don't use it.

**9. Turner, Tsipras, Madry — "Clean-Label Backdoor Attacks" (2019)**
- Establishes clean-label attack paradigm. Useful for related work section.

**10. He et al. — "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022)**
- arXiv: 2111.06377
- MAE is an alternative SSL method; worth knowing but you probably won't use it.

**11. Cinà et al. — "Wild Patterns Reloaded" (ACM Computing Surveys 2023)**
- Comprehensive survey. Read the sections relevant to backdoor and label attacks.

### Tier 3 — Reference As Needed (read only relevant parts)

- Tran et al., "Spectral Signatures in Backdoor Attacks" (NeurIPS 2018)
- Chen et al., "Activation Clustering" (SafeAI workshop 2019)
- Wang et al., "Neural Cleanse" (IEEE S&P 2019)
- Goldblum et al., "Dataset Security for ML: A Survey" (TPAMI 2022)

---

## PART 2 — Environment Setup (Week 2–3)

### 2.1 Hardware Requirements

**Minimum viable:** Google Colab Pro+ ($50/month) — gives you A100 or V100 access. Adequate for CIFAR-10 experiments.

**Preferred:** Access to your university's GPU cluster. Ask your advisor now.

**Rough budget estimate:** You will do about 50-80 full training runs (10 poisoning rates × 2-3 datasets × 3-5 detectors × 2-3 seeds). Each run on CIFAR-10 with ResNet-18 takes about 1 hour on an A100, 3 hours on a V100. So total: ~100-300 GPU hours. Budget accordingly.

### 2.2 Software Stack

Install in this order:

```
# Base
Python 3.10+
CUDA 11.8 or 12.1 (matching your GPU drivers)

# Core ML
PyTorch 2.1+ (latest stable)
torchvision 0.16+

# Utilities
numpy, scipy, pandas, scikit-learn
matplotlib, seaborn
tqdm  # progress bars

# Specialized
faiss-gpu  # for fast k-NN search
wandb  # experiment tracking (strongly recommended)

# SSL models
timm  # PyTorch Image Models library; has pretrained DINO
```

Create a `requirements.txt` the day you start. Update it whenever you install something new. Your future self will thank you.

### 2.3 Project Structure

Set up your repo like this from day 1:

```
thesis-flip-ssl/
├── configs/           # YAML config files for experiments
├── data/              # datasets (usually .gitignore'd)
├── src/
│   ├── attacks/       # FLIP attack code
│   ├── defenses/      # your detector implementations
│   ├── models/        # ResNet, ViT, SSL encoders
│   ├── training/      # training loops
│   ├── evaluation/    # metrics, plotting
│   └── utils/         # helper functions
├── experiments/       # experiment scripts (one per figure/table)
├── notebooks/         # Jupyter notebooks for exploration
├── results/           # saved results (metrics, checkpoints)
├── paper/             # thesis + paper drafts
├── requirements.txt
└── README.md
```

Use git from day 1. Commit every time something works. You will accidentally break things, and git saves you.

---

## PART 3 — FLIP Reproduction (Month 1)

This is where many theses die. The goal: get FLIP running end-to-end and reproduce their main result before building anything else.

### 3.1 Clone and Inspect FLIP

```
git clone https://github.com/SewoongLab/FLIP
cd FLIP
```

Read the README thoroughly. Read their code, especially:
- How they define the attacker's trajectory-matching loss
- How they generate the poisoned label set
- Which datasets and models they support

Don't touch anything yet. Just read and understand.

### 3.2 Run Their Main Experiment

Follow their README to reproduce the CIFAR-10 + ResNet-32 result. Expected: ~99% attack success rate (ASR) with ~2% label corruption, and clean test accuracy (CTA) drop of less than 2%.

**Common problems and fixes:**
- CUDA version mismatch — check their requirements against your install
- Missing precomputed labels — they provide these; download separately
- Dataset path issues — always use absolute paths during debugging
- Running out of GPU memory — reduce batch size in their config

Allocate up to 2 weeks for this phase. If you're stuck after 2 weeks, message me or your advisor. Do not try to push through.

### 3.3 Validate You Have The Attack Working

You should be able to produce a table like this:

| Poisoning Rate | Clean Test Accuracy | Attack Success Rate |
|----------------|---------------------|---------------------|
| 0% (clean)     | ~94%                | N/A                 |
| 0.5%           | ~93%                | ~50-70%             |
| 1%             | ~93%                | ~80-95%             |
| 2%             | ~92%                | ~99%                |
| 5%             | ~91%                | ~99.5%              |

If your numbers are way off, don't proceed. Debug first.

### 3.4 Generate Your Own Poisoned Datasets

You need poisoned datasets at multiple poisoning rates for your experiments. Save them as PyTorch tensors or HDF5 files. Maintain a clean mapping: for each sample index, record (original_label, poisoned_label, is_poisoned). You will need this ground truth for evaluation.

Make sure to use multiple random seeds (at least 3) for each poisoning rate. Variance matters.

---

## PART 4 — SSL Feature Extraction Pipeline (Month 2, Weeks 1–2)

### 4.1 Choosing the SSL Model

My recommendation: start with **DINO v2** (from Meta AI). Reasons:
- Stronger features than SimCLR
- Available pretrained on various backbones (ViT-S, ViT-B, ViT-L)
- Well-documented, easy to load
- Features are known to cluster by semantic class without fine-tuning

Alternative: SimCLR — more "vanilla," easier to explain in your thesis, widely cited.

Do not train an SSL model from scratch. That would take weeks and isn't the point of your thesis. Use pretrained weights.

### 4.2 Critical Caveat — Read Carefully

Pretrained SSL models were trained on large-scale image datasets (ImageNet, LAION, etc.). CIFAR-10 and CIFAR-100 classes overlap with these datasets. So the SSL encoder has implicitly "seen" cat-like and truck-like images before. Technically labels never entered SSL training, but the data distribution does include your classes.

**Why this matters for your thesis:** You must be honest about this in your methodology section. Frame it as: "We use a foundation SSL encoder whose training data excludes explicit labels. While the pretraining data distribution overlaps with our evaluation classes, no supervised signal from our poisoned dataset influences the feature extractor."

This is a real nuance. Don't hide it; address it head-on.

### 4.3 Feature Extraction Procedure

For each training image in your (possibly poisoned) dataset:
1. Apply the SSL model's standard preprocessing (center crop, normalize with ImageNet stats typically)
2. Pass through the frozen SSL encoder
3. Take the [CLS] token embedding (for ViTs) or the pooled feature (for CNNs)
4. Store as a (N, D) tensor where N is number of samples and D is feature dimension (usually 384, 768, or 1024)

Store features on disk. Don't recompute them every experiment.

### 4.4 Validating Your Features

Before running detection experiments, sanity check your features:

**Check 1: Class separability.** Compute the average cosine similarity between:
- pairs within the same class: should be high (0.6–0.9)
- pairs across different classes: should be lower (0.2–0.5)

If within-class and across-class similarities are similar, your features are not semantically meaningful. Debug before proceeding.

**Check 2: Visualization.** Run t-SNE or UMAP on a random subset (2000 samples) of your features, colored by true class label. You should see 10 roughly-separated clusters for CIFAR-10. If it looks like one big blob, something is wrong.

**Check 3: k-NN class prediction accuracy.** Using clean (non-poisoned) labels: for each sample, find its 10 nearest neighbors (excluding itself) and predict its class via majority vote. Your accuracy should be 70-90% for a good encoder on CIFAR-10. If it's below 50%, your features are inadequate.

---

## PART 5 — Implementing the Detector (Month 2, Weeks 3–4)

### 5.1 Core Algorithm (Pseudocode)

```
Input: 
  X = training images (N samples)
  Y = (possibly poisoned) training labels
  encoder = frozen SSL encoder
  k = number of neighbors (hyperparameter, e.g., 10–50)

Step 1: Extract features
  features = encoder(X)                      # (N, D) tensor
  features = normalize(features, dim=1)      # unit vectors, enables cosine sim

Step 2: Build fast nearest neighbor index
  index = FAISS_IndexFlatIP(D)              # inner product = cosine sim on unit vectors
  index.add(features)

Step 3: For each sample, compute neighborhood-disagreement score
  for i in range(N):
    _, neighbor_indices = index.search(features[i], k+1)
    neighbor_indices = neighbor_indices[1:]   # exclude self
    neighbor_labels = Y[neighbor_indices]
    
    # Score = fraction of neighbors whose label disagrees with Y[i]
    disagreement = sum(neighbor_labels != Y[i]) / k
    
    # Alternative: weighted by similarity
    # weighted_disagreement = sum over neighbors j of:
    #   sim(i,j) * (1 if Y[j] != Y[i] else 0)
    # then normalize
    
    scores[i] = disagreement

Step 4: Rank samples by score (descending)
  suspicious_indices = argsort(scores)[::-1]

Step 5: Use threshold or top-k selection
  flagged = suspicious_indices[:num_flagged]
```

### 5.2 Hyperparameters to Explore

**k (number of neighbors):** Try k ∈ {5, 10, 20, 50, 100}. Don't commit early. Report performance across k values in an ablation.

**Scoring variant:** 
- Unweighted disagreement (as above)
- Similarity-weighted disagreement
- Soft disagreement using label distributions (if you later extend to soft labels)

**Distance metric:**
- Cosine similarity (standard for deep features)
- Euclidean distance (may work; worth one ablation)

### 5.3 Efficiency Notes

For CIFAR-10 (50K samples), exact k-NN is fast enough with FAISS. For Tiny-ImageNet or CIFAR-100 (100K+), consider FAISS's IVF or HNSW indices for speed. But always verify approximate results match exact k-NN on a subset.

### 5.4 Calibrating to Probabilistic Scores

Raw disagreement fractions are not well-calibrated probabilities. Optional refinement: fit a simple logistic regression on a held-out validation set (where you know the true poisoned samples) to map scores to probabilities. Frame this as "calibrated label trust score" in your paper — sounds more sophisticated than "k-NN disagreement ratio."

---

## PART 6 — Core Experiments (Month 3)

This is the heart of your thesis. Run these experiments carefully and document every decision.

### 6.1 Experimental Protocol

For each configuration, you will:
1. Generate a poisoned dataset (with fixed seed for reproducibility)
2. Extract SSL features (one-time cost per dataset)
3. Run your detector → get per-sample scores
4. Compute metrics against ground truth poisoned mask
5. Repeat with 3 different seeds, report mean ± std

### 6.2 Experimental Matrix

| Variable | Values to Test |
|----------|---------------|
| Dataset | CIFAR-10 (primary), CIFAR-100 (secondary) |
| Model architecture | ResNet-18 (for FLIP training) |
| SSL encoder | DINOv2-ViT-S/14 (primary), SimCLR-ResNet50 (secondary) |
| Poisoning rate | 0.5%, 1%, 2%, 5% |
| k (neighbors) | 10, 20, 50 |
| Seeds | 3 seeds per config |

Don't run all combinations — that's 720+ runs. Prioritize:
- Main result: one SSL encoder, all poisoning rates, all datasets, 3 seeds, k=20
- SSL comparison: both encoders, one poisoning rate (2%), one dataset, 3 seeds
- k ablation: one encoder, one poisoning rate, one dataset, all k values, 3 seeds

### 6.3 Baselines (Critical — Don't Skip)

You must compare against at least these:

**Baseline 1: Random detection** — flag k% of samples at random. Establishes floor performance.

**Baseline 2: Loss-based ranking** — train a classifier, rank samples by final training loss, flag highest-loss samples. Simple and sometimes surprisingly competitive.

**Baseline 3: Deep k-NN with supervised features** — same k-NN audit, but use features from a classifier trained on the poisoned data. This is *the* critical comparison because it isolates the SSL-vs-supervised question.

**Baseline 4 (if time):** AUM — use Pleiss et al.'s margin-tracking method.

**Baseline 5 (if time):** Confident Learning — use the cleanlab library.

### 6.4 Metrics to Report

Primary:
- **AUROC**: overall detection quality
- **AUPRC**: more informative given class imbalance (only 2% poisoned)
- **Precision@k**: for k = (true number of poisoned samples). Tells you "if I flag exactly as many as there are, how many are actually poisoned?"
- **Recall@k for fixed precision** (e.g., recall when precision = 80%)

Secondary:
- **Detection rate across poisoning rates** — curve plot
- **Per-class detection rate** — bar chart showing which classes are easier/harder

### 6.5 Mitigation Experiments

After detection, the second half: show that acting on detection reduces attack success.

For each detection method:
1. Remove top-k% flagged samples (k = detected count, or 2× expected)
2. Retrain a ResNet-18 on the cleaned dataset
3. Measure: clean test accuracy, attack success rate

Compare:
- No defense (baseline): CTA ~92%, ASR ~99%
- Your method: ideally CTA > 90%, ASR < 40%
- Random removal: should be worse than your method
- Deep k-NN (supervised features): your comparison

Also try soft downweighting: weight each sample by (1 - score), retrain. Often better than hard removal.

---

## PART 7 — Analysis (Month 4, Weeks 1–2)

Detection numbers alone are not enough. Your thesis needs insight into *why* things work or don't.

### 7.1 Failure Mode Analysis

For samples your method missed (false negatives):
- Are they concentrated in specific classes?
- Is there a visual pattern? (Look at actual images)
- Are they in regions of feature space where classes overlap?

For false positives (clean samples you wrongly flagged):
- Are they naturally ambiguous? (e.g., an image of a cat sitting in a car labeled "cat")
- Are they already known label errors in CIFAR-10? (see Northcutt et al.'s analysis)

### 7.2 Per-Class Analysis

Make a confusion-matrix-style plot: which (original_class → flipped_class) pairs are easiest/hardest to detect? Expectation:
- Easy: airplane → frog (semantically very different, features very far apart)
- Hard: cat → dog, truck → car (semantically similar, features overlap)

This analysis will be one of the most valuable parts of your thesis.

### 7.3 Visualizations to Include

- **Feature space visualization:** t-SNE or UMAP of features, colored by true label, with poisoned samples highlighted. Shows whether poisoned samples are visually distinguishable in feature space.
- **Score distribution histograms:** suspicion scores for clean vs. poisoned samples, overlaid. Good separation = good detection.
- **ROC curves and PR curves:** your method vs. baselines.
- **Ablation plots:** performance vs. k, performance vs. poisoning rate.

---

## PART 8 — Thesis Writing (Month 5)

### 8.1 Thesis Chapter Structure

**Chapter 1 — Introduction**
- Motivation: why ML security matters, scale of label-poisoning threat
- Specific problem: label-only backdoors are hard to detect
- Your contribution (3-4 bullets)
- Thesis organization

**Chapter 2 — Background and Related Work**
- ML security landscape (brief)
- Backdoor attacks: evolution from BadNets to FLIP
- Label noise detection methods (AUM, CL, Deep k-NN)
- Self-supervised learning (SimCLR, DINO)
- Gap you're filling

**Chapter 3 — Methodology**
- Threat model (reproduce FLIP's)
- Your approach: SSL features + k-NN audit
- Detection algorithm (with pseudocode)
- Mitigation strategies
- Rationale for each design choice

**Chapter 4 — Experimental Setup**
- Datasets, models, SSL encoders
- Implementation details (enough for reproducibility)
- Metrics
- Baselines
- Compute budget

**Chapter 5 — Results**
- Detection performance (main table)
- Mitigation results (ASR reduction table)
- Ablations (k, scoring variants, SSL choice)
- Comparison to baselines

**Chapter 6 — Analysis**
- Failure modes
- Per-class breakdown
- Feature space visualizations
- Why SSL features help (or don't)

**Chapter 7 — Discussion**
- Limitations (honest)
- Connections to related work
- Adaptive attacks (brief speculation)
- Future work

**Chapter 8 — Conclusion**

### 8.2 Paper (Workshop Version) Outline

Same content, compressed to 8 pages:
1. Intro (0.5 page)
2. Related work (0.5 page)
3. Method (1.5 pages)
4. Experiments (3 pages)
5. Analysis (1.5 pages)
6. Discussion + conclusion (1 page)

Target venues:
- NeurIPS SafeML Workshop
- ICML AdvML-Frontiers Workshop
- ICLR Workshop on Backdoor Attacks
- IEEE SaTML (dedicated trust-in-ML conference)

### 8.3 Figures to Create

Make these early, refine throughout:

1. **Threat model diagram** — attacker flips labels only; images unchanged.
2. **Method overview diagram** — SSL encoder → feature extraction → k-NN audit → scores.
3. **Feature space visualization** — t-SNE with poisoned samples highlighted.
4. **Main results bar chart** — AUROC for your method vs. baselines across poisoning rates.
5. **ROC/PR curves** — all methods on one plot.
6. **Ablation: k value** — line plot.
7. **Per-class detection accuracy** — heatmap or bar chart.
8. **Mitigation results** — CTA vs. ASR scatter plot, your method vs. baselines.

### 8.4 Tables to Create

1. **Main detection table** — methods × poisoning rates × metrics.
2. **Mitigation table** — methods × (CTA, ASR after defense).
3. **SSL encoder comparison** — DINOv2 vs. SimCLR vs. supervised features.
4. **Confusion table** — easy vs. hard class pairs for detection.

---

## PART 9 — Risk Management

### 9.1 Risks and Contingencies

**Risk: FLIP reproduction fails.**  
Contingency: use simpler label-flipping attack (Xiao et al. 2012 optimal label flipping). Thesis becomes "defending against optimal label flipping using SSL features" — still publishable.

**Risk: SSL features don't separate CIFAR classes well.**  
Contingency: try different encoders (DINOv2, CLIP, MAE). If all fail, pivot to supervised features trained on held-out clean data as the backbone.

**Risk: Your method underperforms Deep k-NN baseline.**  
Contingency: this is still publishable as a negative result — "SSL features do not provide the expected robustness benefit because [reason]." Analysis becomes your main contribution.

**Risk: Running out of time.**  
Contingency: drop CIFAR-100, drop one baseline, drop additional SSL encoders. Core result (CIFAR-10, one encoder, main baselines) is non-negotiable.

### 9.2 Things That Will Go Wrong (and are normal)

- Environment problems costing 2-3 days
- Off-by-one errors in poisoned sample indexing
- Features loaded with wrong preprocessing
- Metric computation bugs (precision vs. recall swapped)
- Plots that look different in the paper PDF than in your notebook

Budget 20% slack time. Don't plan to finish "exactly on time."

---

## PART 10 — Writing and Advisor Tips

### 10.1 How to Work With Your Advisor

- Send weekly updates, even short ones
- When asking for feedback, frame specific questions (not "what do you think?")
- Bring results, not plans, to meetings
- Keep a decisions log: every time you make a design choice, write down why. You'll forget otherwise.

### 10.2 When to Start Writing

Start writing chapters 3 and 4 (methodology, experimental setup) during month 3. This serves two purposes: forces you to clarify your own understanding, and reduces panic in month 5.

### 10.3 Citation Hygiene

Use BibTeX. Keep a single `references.bib`. When you read any paper, add it immediately. This saves hours of pain later.

---

## Quick Reference — Daily/Weekly Checklist

**Every day:**
- Git commit working code
- Log experiments in wandb/tensorboard
- Update notes document

**Every week:**
- Update advisor (even if just 2 sentences)
- Review what worked and what didn't
- Re-plan next week based on reality

**Every month:**
- Re-read your thesis statement. Is your work still answering it?
- Back up everything (code, results, writing) to multiple locations

---

## Final Note

The goal is not to produce a flawless thesis. The goal is to produce an honest, reproducible piece of research that you understand deeply. An undergrad thesis with clean negative results and good analysis is worth far more than a pretty-looking thesis with inflated claims. Reviewers and graders can always tell the difference.

You have enough to start. Begin this week.

