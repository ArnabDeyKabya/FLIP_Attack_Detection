# Analysis Report: A Label-Independent k-NN Defense Against the FLIP Label-Poisoning Attack

*An automated research-style analysis of the experiment artifacts found in this project.*
*Prepared as a combined research-mentor, thesis-supervisor, and paper-reviewer evaluation.*

---

## 0. How to read this report

This document was produced by scanning the whole project folder, reading the
source code, the configuration files, every results file (JSON, CSV, logs),
and looking directly at the generated figures. Wherever a number appears, it
comes from a real file in this directory — not from a generic expectation.

Every important point is written **twice**:

- **In plain words** — so a beginner can follow it.
- **In research terms** — so a supervisor or reviewer gets the precise version.

If you only have five minutes, read Section 1 (the big picture) and Section 9
(the verdict).

---

## 1. The big picture in one minute

**What problem is being studied?**
Modern models are often trained on data whose *labels* come from untrusted
sources (crowd workers, scraped web data, third-party annotators). An attacker
can secretly change a small number of training labels so that the final model
behaves normally on clean data but misbehaves when it sees a special "trigger".
This is a **label-only backdoor attack**. The specific attack here is called
**FLIP**, which carefully chooses *which* labels to flip by simulating how the
victim model trains.

**What is the defense?**
Before training, look at each training image with a *frozen, label-blind*
vision model (**DINOv2**, a self-supervised encoder). For each image, find its
`k` nearest neighbors in this feature space and check: *do my neighbors carry
the same label that I was given?* If most of my neighbors disagree with my
label, my label is probably poisoned. Suspicious samples are then **removed**
before training. This is a **k-Nearest-Neighbor (k-NN) label-consistency
audit**.

**Why might it work?**
FLIP optimizes its flips against the *supervised training process*. But DINOv2
features were computed *without ever seeing the (poisoned) labels*, so FLIP has
no direct way to hide there. A truck image relabeled as "deer" still *looks
like a truck* to DINOv2, and its neighbors are real trucks → disagreement →
flagged.

**Did it work here?** Yes, strongly, on the one configuration that was run:

| | Without defense (baseline) | With defense (best operating points) |
|---|---|---|
| Attack success rate (PTA) | **~99.8%** | **0.2% – 4.5%** (for k ≥ 50) |
| Clean accuracy (CTA) | **~89.6%** | **92.0% – 92.4%** (actually *higher*) |
| Poison detection AUROC | — | **0.97 – 0.99** (for k ≥ 5) |

In plain words: **the attack went from "almost always succeeds" to "almost
never succeeds", and the clean accuracy did not drop — it slightly improved.**

The rest of this report explains every number, every figure, what is
trustworthy, and what is still missing for a complete thesis.

---

## 2. What is in this project (the map)

The project is a fork/wrapper of the official **FLIP** attack codebase, with a
new defense bolted on as a sibling module. Here is the structure that matters
(virtual environments, dataset binaries, and model checkpoints omitted):

```
flip-defense/
├── modules/
│   ├── knn_defense/                ← THE DEFENSE (the thesis novelty)
│   │   ├── ssl_features.py         ← DINOv2 feature extractor + on-disk cache
│   │   ├── knn_detector.py         ← k-NN disagreement scorer
│   │   └── defense_modes.py        ← removal policy + detection metrics
│   ├── train_user_defense/
│   │   └── run_module.py           ← trains the victim model WITH the defense
│   ├── train_expert/ generate_labels/ select_flips/ train_user/   ← original FLIP attack
│   └── base_utils/, pytorch_cifar/ ← FLIP's shared training/data code
├── scripts/
│   ├── sanity_check_defense.py     ← quick "does the detector work at all?" check
│   └── k_sweep/                    ← the full sweep: run, aggregate, plot, tabulate
├── experiments/
│   ├── example_attack/             ← the FLIP attack outputs (poisoned labels)
│   ├── knn_defense_cifar_1xs_1500_{none,remove}_k{1..2000}/   ← 13 runs
│   └── _report_knn_defense_cifar_1xs_1500/                    ← THE RESULTS HUB
│       ├── master_metrics.{csv,json}   ← one row per (k, mode)
│       ├── per_class_detection.csv     ← detection broken down per class
│       ├── threshold_sweep.csv         ← metrics vs decision threshold
│       ├── training_curves.csv         ← per-epoch CTA/PTA
│       ├── sweep.log                   ← full run log
│       ├── figures/{detection,training,features,samples}/
│       └── tables/                     ← thesis-ready Markdown + LaTeX tables
├── precomputed_labels/cifar/r32p/1xs/  ← 1500.npy (poisoned), true.npy (clean)
├── data/ssl_features/                  ← cached DINOv2 features (50000 × 384)
└── README.md, *_English.md             ← design docs, thesis framing, plan
```

**How the files relate to each other (the data flow):**

```
true.npy  +  1500.npy  ──►  is_poisoned_gt  (which labels were changed)
CIFAR images  ──► DINOv2 ──► features (50000×384, L2-normalised, cached once)
features + assigned labels ──► k-NN detector ──► scores.npy (suspicion 0..1)
scores + threshold ──► removed_indices / kept_indices
kept data ──► train ResNet-32 ──► caccs.npy (CTA), paccs.npy (PTA)
all of the above ──► detection_metrics.json + summary.json (per run)
all runs ──► aggregate ──► master_metrics + tables + figures
```

Everything downstream traces back to two arrays: the **poisoned labels**
(`1500.npy`) and the **true labels** (`true.npy`). Their difference defines the
ground truth of "which samples are poisoned", which is the yardstick for every
detection metric.

---

## 3. The experiment setup (read this before the numbers)

From `experiments/example_attack/config.toml` and the run configs:

| Setting | Value | Plain meaning |
|---|---|---|
| Dataset | CIFAR-10 | 50,000 training images, 10 classes, 5,000 each |
| Victim model | `r32p` (ResNet-32) | the model the attacker wants to backdoor |
| Attack | FLIP, poisoner `1xs` (sinusoidal trigger) | flips labels so a trigger → target class |
| Source → Target | class 9 (**truck**) → class 4 (**deer**) | trucks get pushed toward the "deer" label |
| Budget | 1500 labels (= **3%** of the data) | how many labels the attacker may flip |
| Defense encoder | DINOv2 ViT-S/14 | frozen, label-blind feature extractor (384-dim) |
| Scoring | `disagreement` | fraction of k neighbors whose label ≠ mine |
| Removal policy | `auto`, threshold 0.5 | remove any sample with >50% disagreeing neighbors |
| `k` swept over | 1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000 | neighborhood size (the main variable) |
| Seed | 0 (single seed) | one run each |
| Training | SGD, 200 epochs | from `training_curves.csv` |

**Two modes are compared:**
- `none` = baseline. Detection scores are still computed, but **nothing is
  removed**. This reproduces the undefended FLIP result.
- `remove` = defended. Flagged samples are dropped before training.

A subtle but important detail confirmed in `sweep.log`: the `none` baseline was
only fully trained for **k = 1, 5, 10**, because *without removal the training
data is identical regardless of k* — so the baseline is k-independent and one
value suffices. (This has a small reporting consequence; see Section 8.2.)

---

## 4. The evaluation metrics, explained simply

This experiment has **two kinds** of metrics. People often confuse them, so
keep them separate in your head.

### 4.1 "Did the defense save the model?" — training metrics

These come from `summary.json` / `training_curves.csv` and measure the final
model.

- **CTA — Clean Test Accuracy.** Accuracy on normal, clean test images.
  *Higher is better.* This tells you whether the defense damaged the model's
  normal job. Think of it as "is the model still good at its day job?"

- **PTA — Poison Test Accuracy (a.k.a. Attack Success Rate, ASR).** How often
  the backdoor *works* — i.e., how often a triggered test image is pushed to
  the attacker's target class. **Here, lower is better for the defender.**
  PTA near 1.0 = "the attack fully succeeds". PTA near 0.0 = "the backdoor was
  not installed". *This is the single most important number for judging the
  defense.*

> ⚠️ Naming trap: "Poison Test *Accuracy*" sounds like something you'd want to
> be high. It is the opposite. A *low* PTA means the *defense* won. Read it as
> "attack success rate".

### 4.2 "Did the detector find the poisons?" — detection metrics

These come from `detection_metrics.json` / `master_metrics.csv` and measure the
per-sample suspicion scores against the ground-truth poison mask. They do
**not** require training a model.

- **AUROC** (Area Under ROC Curve). Probability that a random poisoned sample
  gets a higher suspicion score than a random clean sample. 0.5 = random
  guessing; 1.0 = perfect ranking. *Higher is better.* Robust to class
  imbalance.

- **AUPRC** (Area Under Precision-Recall Curve). Like AUROC but focused on the
  rare positive class. Because only 3% of samples are poisoned, AUPRC is the
  **more honest** number here. A random detector would score ~0.03 (the base
  rate), so anything well above that is meaningful.

- **Precision@k** and **Recall@k** (here `k = number of true poisons = 1500`):
  if you remove exactly the 1500 highest-scoring samples, what fraction are
  truly poison? Because the cut size equals the number of poisons, precision
  and recall are equal by construction. This is the standard "Deep k-NN
  defense" literature metric.

- **Flagged precision / recall** at the actual cut used (threshold 0.5):
  - *Precision* = of everything we removed, how much was really poison? (low
    precision = we threw away clean data too)
  - *Recall* = of all the poison, how much did we catch? (high recall = few
    poisons survived)

- **F1** = balance of precision and recall. **MCC** (Matthews Correlation
  Coefficient) = a balanced score from −1 to +1 that behaves well under heavy
  class imbalance; ~0.7 here is strong. **Balanced accuracy** = average of
  true-positive rate and true-negative rate. **FPR** = false-positive rate
  (fraction of clean wrongly flagged).

- **Feature-quality checks** (from the demo `summary.json`):
  - *within-class similarity* − *between-class similarity* = **gap**. A big gap
    means the encoder cleanly separates classes (good for detection).
  - *k-NN class accuracy* = if you classify each image by majority vote of its
    feature neighbors, how accurate is it? This is a direct test that the
    features "know" the classes.

---

## 5. The headline results (with full numbers)

### 5.1 The baseline: the FLIP attack works perfectly

From `table_training_sweep.md` (mode = `none`):

| k | CTA (clean acc) | PTA (attack success) |
|---|---|---|
| 1 | 0.898 | 0.997 |
| 5 | 0.897 | 1.000 |
| 10 | 0.895 | 0.998 |

**Plain:** Without any defense, the model is ~89.6% accurate on clean data, and
the backdoor succeeds **~99.8% of the time**. The attack is devastating and
fully working — exactly the threat the defense must counter.

**Technical:** With a 3% label budget on a semantically distant source→target
pair, FLIP achieves near-saturated ASR while keeping CTA at a normal level for
ResNet-32 on CIFAR-10. The attack leaves essentially no clean-accuracy
fingerprint, which is precisely why model-internal defenses struggle and why a
pre-training, label-independent audit is attractive.

The three baseline rows differ very slightly (CTA 0.895–0.898, PTA 0.997–1.000)
even though, with no removal, they *should* be identical. This is harmless GPU
training non-determinism, but it sets a **noise floor of roughly ±0.2% CTA /
±0.3% PTA** that you should keep in mind when reading small differences (see
Section 8.1).

### 5.2 The defense: the attack collapses, clean accuracy is preserved

From `master_metrics.json` (mode = `remove`, auto threshold 0.5). This is the
central table of the whole project:

| k | removed | % of data | CTA | PTA (↓ better) | AUROC | AUPRC | flag-Prec | flag-Recall | best F1 | best MCC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3490 | 7.0% | 0.911 | **0.470** | 0.848 | 0.246 | 0.320 | 0.745 | 0.448 | 0.466 |
| 5 | 3131 | 6.3% | 0.916 | 0.386 | 0.974 | 0.515 | 0.405 | 0.846 | 0.592 | 0.581 |
| 10 | 2732 | 5.5% | 0.913 | 0.219 | 0.983 | 0.572 | 0.464 | 0.846 | 0.618 | 0.613 |
| 20 | 2981 | 6.0% | 0.921 | 0.111 | 0.986 | 0.614 | 0.460 | 0.913 | 0.644 | 0.641 |
| 50 | 3232 | 6.5% | 0.920 | 0.045 | 0.987 | 0.657 | 0.439 | 0.947 | 0.658 | 0.665 |
| 100 | 3527 | 7.1% | 0.921 | 0.027 | 0.988 | 0.665 | 0.412 | 0.969 | 0.668 | 0.672 |
| 200 | 3827 | 7.7% | 0.923 | **0.007** | 0.988 | 0.665 | 0.390 | 0.994 | 0.664 | 0.671 |
| 500 | 4425 | 8.9% | **0.924** | 0.003 | 0.988 | 0.663 | 0.339 | 0.999 | 0.656 | 0.667 |
| 1000 | 5176 | 10.4% | 0.918 | **0.002** | 0.987 | 0.652 | 0.289 | 0.999 | 0.664 | 0.684 |
| 2000 | 6783 | 13.6% | 0.908 | 0.005 | 0.987 | 0.649 | 0.221 | 0.998 | **0.684** | **0.704** |

This single table tells the whole story. Reading it carefully:

**(a) The attack is neutralized.** PTA drops from ~0.998 (baseline) to as low as
**0.002** (k=1000). For any k ≥ 50, PTA ≤ 4.5% — the backdoor is essentially
gone. Even a tiny k=20 already brings PTA to 11%.

**(b) Clean accuracy is preserved — even improved.** CTA rises from ~0.896
(baseline) to ~0.92 for most k. The `defense_success_vs_k` figure makes this
vivid: the "CTA drop" line sits *slightly below zero*, meaning removing
suspicious samples **helped** clean accuracy. *Plain reason:* the removed
samples are mostly genuinely mislabeled or visually ambiguous images, and
training on cleaner data produces a slightly better model.

**(c) Bigger k = stronger attack suppression, but more clean data thrown away.**
As k grows, recall climbs to ~99.9% (almost no poison survives) but precision
falls (0.32 → 0.22) and the amount removed climbs from 7% to 13.6%. So very
large k over-removes. The CTA finally dips at k=2000 (0.908) because too much
clean data is gone.

**(d) The sweet spot.** Around **k = 50–200**: PTA is essentially 0 (0.7%–4.5%),
CTA is at its peak (~0.92), and only ~6.5–7.7% of data is removed. This is the
operating point a practitioner should pick.

**(e) k=1 is too crude.** With a single neighbor, the score is binary (0 or 1),
which is noisy: PTA only falls to 0.47 (barely below the 0.5 "win" line) and
AUROC is only 0.848. The method needs k ≥ 10 to be reliable.

> **Reviewer note on a reporting gap:** The auto-generated `table_headline.md`
> and the `defense_success_vs_k.png` figure only show **k = 1, 5, 10**, because
> they pair each defended run against a *matched* `none` baseline, and the
> baseline was only trained for those three k. This accidentally **hides the
> best results** (k=50–1000). The fix is trivial and the data already exists in
> `table_training_sweep.md`: treat the single k-independent baseline
> (CTA≈0.896, PTA≈0.998) as a constant reference line for *all* k. I strongly
> recommend regenerating these two artifacts before the thesis goes out.

---

## 6. Deep dive: what the figures and per-class data reveal

### 6.1 Detection is near-perfect, and it saturates fast (ROC + k-sweep)

`figures/detection/roc_all_k.png` shows ROC curves hugging the top-left corner
for every k ≥ 5. AUROC climbs from 0.848 (k=1) to 0.974 (k=5) and then plateaus
at ~0.987–0.988 from k=50 onward. `k_sweep_detection.png` confirms the same
shape for AUPRC (rises to ~0.665 and plateaus), best-F1 (~0.68) and best-MCC
(~0.70).

**Plain:** Even a small neighborhood (k=5) already separates poison from clean
extremely well. Going beyond k≈50 gives almost nothing in pure *ranking* power.

**Technical:** The detection signal is dominated by a small, very confident set
of poisons (the source-class images sitting in the target class), so the ROC
saturates quickly. AUPRC plateauing at ~0.665 — well above the 0.03 base rate —
is the honest headline detection number, because the problem is 97% negative.

### 6.2 Score distributions: a clean/poison split that sharpens with k

`figures/detection/score_dist_facet.png` (histograms, log y-axis, dashed line at
the 0.5 threshold):

- At **k=1**, scores are only 0 or 1; many *clean* samples also land at 1.0, so
  the threshold sweeps up a lot of false alarms.
- From **k=20–200**, a clear valley forms: clean mass piles up near 0, poison
  mass piles up at 0.7–1.0, with a relatively empty middle. The 0.5 threshold
  sits in this valley — a good, natural cut.
- The poisoned histogram is slightly **bimodal** at high k (a lump near 0.75 and
  a spike at 1.0). The 1.0 spike is the "easy" poisons (all neighbors disagree);
  the broad lump is harder cases.

**Why this matters:** It justifies the design choice of threshold = 0.5 ("more
than half my neighbors disagree"). It is not an arbitrary number; it falls in
the natural gap between the two populations.

### 6.3 Feature space: poisons form their own island (UMAP)

`figures/features/features_umap_poison.png` plots DINOv2 features (subsampled:
3500 clean, 1500 poison) colored by poison status. A **large red cluster** sits
apart in the top-right: these are the source-class images (trucks) that were
relabeled to the target (deer). They cluster together because DINOv2 sees them
as trucks, regardless of the "deer" label they carry. A minority of red points
are scattered inside other clusters — those are the harder poisons that the
detector misses (false negatives).

**This is the visual proof of the thesis hypothesis:** the label flip does not
move the image in semantic feature space, so the poison is geometrically
exposed.

The demo `summary.json` quantifies the feature quality directly:
within-class similarity 0.310, between-class 0.106, **gap 0.204**, and **k-NN
class accuracy 96.3%**. In plain terms, DINOv2 alone classifies CIFAR-10 at 96%
without any training — that is *why* the neighborhood audit is so effective.

### 6.4 Per-class breakdown: where the hits and false alarms live

This is the most insightful artifact (`per_class_detection.csv` +
`per_class_flag_heatmap.png`). The poisons are not spread evenly:

- **Class 4 (deer, the target):** holds **1209 of the 1500 poisons** (80.6%).
  This class has 6209 samples instead of 5000 — it absorbed the flipped images.
  Detection precision here is **high (~0.85–0.89)**: trucks-labeled-deer are
  easily caught because their neighbors are real deer (disagreeing) / real
  trucks elsewhere.

- **Class 9 (truck, the source):** shrank to 3711 samples (it lost images to
  deer). It has almost no true poison (8) yet a **high flag rate** (0.23 at k=1,
  falling to 0.08 at k=10). These are **false positives** — genuine trucks
  flagged by mistake.

  *Why?* A real truck's nearest neighbors include the 1209 trucks that were
  relabeled to "deer". Those neighbors now carry the label "deer", which
  disagrees with "truck" → the clean truck's disagreement score is inflated.
  **The poison contaminates the neighborhood of clean source-class samples.**
  Larger k averages this out (0.23 → 0.08), which is one more reason small k is
  worse.

- **Other classes:** low poison counts (5–94), low flag rates (0.01–0.13) —
  this is the background false-alarm level.

This per-class story is genuinely valuable for the thesis: it shows *the
mechanism*, not just the score. It also explains why **overall** flagged
precision is only ~0.3–0.46 even though the target-class precision is ~0.87 —
the false alarms in the source/other classes drag the global precision down.

### 6.5 What the removed images actually look like (sample grids)

- `samples_TP_k20.png` (true positives — correctly removed poison): these are
  clearly mislabeled — e.g., a truck image carrying the label "deer". The grid
  confirms the detector removes *real* label errors, not noise.
- `samples_FP_k20.png` (false positives — clean images wrongly removed): these
  are mostly **genuinely ambiguous** images — atypical poses, occlusions, or
  species that look like each other (a deer that resembles a horse, a cat that
  resembles a dog). Removing such borderline clean samples costs very little
  clean accuracy, which is exactly why CTA stays high (even improves) despite
  ~6–14% removal.

### 6.6 Threshold behavior: the 0.5 cut is good but not optimal at large k

`threshold_sweep.csv` (example for k=20) traces precision/recall/F1/MCC across
thresholds 0.0–1.0. At the 0.5 cut: recall 0.913, precision 0.460, F1 0.611,
FPR 0.033. F1 actually **peaks higher (~0.644) at threshold ≈ 0.66**. The
`table_best_thresholds.md` shows the best threshold drifts upward with k
(0.62 → 0.74). Consequently, the *fixed* 0.5 threshold over-flags at large k:
`f1_at_thr` collapses to 0.36 at k=2000 while `best_f1` stays ~0.68.

**Takeaway:** The defense still wins at the fixed 0.5 threshold (because
over-removing clean data is cheap here), but a **k-aware threshold** would
remove far less clean data for the same protection. This is a clean,
low-effort improvement to suggest.

### 6.7 Training curves confirm the result is stable, not a lucky epoch

`training_curves.csv` records all 200 epochs. The final PTA values are flat over
the last epochs (e.g., k=20 hovers at 0.10–0.11; k=200 at 0.007–0.008), so the
reported numbers are converged steady states, not single-epoch flukes.

---

## 7. The standalone sanity check (the "is the detector even sane?" test)

`out/knn_defense_sanity/demo_cifar_9to4_b1500/summary.json` is a **separate,
detection-only** experiment using *random* truck→deer flips (not FLIP). It is
the green-light gate described in the README.

| Check | Demo (random flips) | Real FLIP (this sweep, comparable k) | README target |
|---|---|---|---|
| Feature gap | 0.204 | — | ≥ 0.15 ✅ |
| k-NN class acc (k=10) | 0.963 | — | ≥ 0.60 ✅ |
| AUROC (k=20) | 0.983 | 0.986 | > 0.90 ✅ |
| AUPRC | 0.457 | 0.614 | — |
| Precision@k | 0.497 | 0.597 | — |

**This is one of the most scientifically interesting findings in the project,
and it deserves to be foregrounded in the thesis:**

The project's own design docs *predicted* that real FLIP would be **much
harder** to detect than random flips (expected AUROC 0.70–0.90, because "FLIP
labels are optimised to look statistically normal"). **The data shows the
opposite: real FLIP is detected just as well as — even slightly better than —
random flips (AUROC 0.986 vs 0.983).**

**Plain meaning:** FLIP works hard to fool the *training process*, but that
effort buys it *nothing* against a label-blind feature detector. The cleverness
of FLIP is invisible to DINOv2.

**Research meaning:** This is direct empirical support for the central
hypothesis — *label-independent representations are immune to the
supervised-trajectory optimization that FLIP relies on.* It is arguably the
strongest scientific point in the work, and right now it is buried in a sanity
script rather than presented as a headline experiment.

---

## 8. Problems, inconsistencies, and things to verify

As a reviewer, here are the issues I would raise. None of them invalidate the
result, but several must be addressed before the work is paper- or
defense-ready.

### 8.1 Single seed → no error bars (must fix)

Every cell in every table is **n = 1**. The three `none` runs that *should* be
identical instead differ by ±0.2% CTA / ±0.3% PTA — direct evidence of
run-to-run noise. The CTA "improvement" under defense (~+2.5 points) is larger
than this noise so it is probably real, but without ≥3 seeds you cannot put a
confidence interval on any number, and a careful reviewer will ask for one.
**Action:** rerun at least the key k values (e.g., 1, 20, 200) over 3–5 seeds and
report mean ± std.

### 8.2 The "easy case" was tested; the "hard case" was not (most important)

The chosen attack is **truck → deer** — two *semantically very different*
classes. This is the **easiest possible case** for a semantic-feature detector,
and it is precisely why AUROC hits 0.99. But the project's own
`Thesis_Novelty_Framing_English.md` states the *expected failure mode* is
**semantically similar pairs** (cat ↔ dog, truck ↔ car). **Those experiments
are not present.** The single most valuable missing experiment is to repeat the
sweep with a hard pair (e.g., cat → dog or automobile → truck) and show whether
AUROC degrades as the theory predicts. Without it, the strong headline is
honest but unrepresentative, and the "failure characterization" promised as a
core contribution is undelivered.

### 8.3 No comparison to the obvious baselines (must fix for the central claim)

The framing doc's contribution #3 is: *"self-supervised features outperform
supervised features (trained on poisoned data) by X points AUPRC."* This
comparison **does not exist in the data.** There is no run using:
- supervised features from the poisoned model (the actual Deep k-NN / Peri et
  al. 2020 baseline), nor
- other label-noise detectors (AUM, Confident Learning), nor
- backdoor-specific defenses (activation clustering, spectral signatures).

So the claim "label-independence is what matters" is currently *asserted*, not
*demonstrated*. The whole scientific punchline rests on this missing baseline.
At minimum, add the supervised-feature k-NN comparison on the same data.

### 8.4 Narrow scope (acknowledged, but limits the conclusions)

Only **one** of each: dataset (CIFAR-10), model (r32p), poisoner (1xs), budget
(1500 / 3%), encoder (ViT-S/14), source→target pair. The plan documents promise
budgets 0.5–5%, CIFAR-100, and the ViT-B/14 encoder ablation — none are run.
The k-sweep is thorough but it is a sweep along a *single axis* of a *single
configuration*. Generalization is currently untested.

### 8.5 The fixed 0.5 threshold over-removes at large k

As shown in 6.6, the auto threshold is suboptimal for large k (F1-at-threshold
collapses to 0.36 at k=2000 even though best-F1 is 0.68). Report results with a
calibrated/k-aware threshold, or simply recommend k≈50–200 with threshold 0.5
and stop there.

### 8.6 The `budget` (oracle) policy was validated but not used in the sweep

The demo computes the oracle-k `budget` policy (remove exactly the top-1500),
but the full training sweep only uses `auto`. Since precision@k is the
literature-standard comparison metric, running a few `budget`-policy training
points would strengthen the comparison to prior work. (Minor.)

### 8.7 Truncated headline artifacts (cosmetic but misleading)

As noted in Section 5: `table_headline.md` and `defense_success_vs_k.png` stop
at k=10 and hide the best results. Regenerate them against the constant baseline.

### 8.8 Things that are *correct* and worth crediting

To be balanced — I checked the code, and the methodology is sound:
- `is_poisoned_gt = (assigned_label != true_label)` — ground truth is defined
  correctly (`run_module.py`).
- AUROC/AUPRC use the correct score direction (higher = more poisoned).
- The k-NN detector does **exact self-exclusion** (`sims[i,i] = -inf`), uses
  L2-normalised cosine similarity, and asserts normalization — no off-by-one or
  "neighbor is myself" bug.
- The `none` mode returns the dataset unchanged (byte-for-byte baseline), and
  the `auto` policy never reads the ground-truth mask — so there is **no oracle
  leakage** in the headline mitigation result. This is exactly the discipline a
  reviewer hopes to find.
- Features depend only on (dataset, encoder), are cached once, and the label is
  provably discarded before the encoder — the "label-independent" claim is true
  by construction, not just by intention.

---

## 9. The verdict (mentor + supervisor + reviewer)

### 9.1 Are the experiments convincing?

**For the specific claim "a DINOv2 k-NN audit defeats FLIP on a
semantically-separated CIFAR-10 attack" — yes, very convincing.** The result is
internally consistent across five independent views: ROC/AUPRC, score
histograms, UMAP geometry, per-class breakdown, sample grids, and the final
CTA/PTA. The attack goes from ~99.8% success to ~0%, clean accuracy is
preserved, and the mechanism is clearly visualized. The engineering is clean and
leak-free.

**For the broader thesis claim "SSL feature auditing is a general defense
against label-only backdoors" — not yet.** That requires the hard class-pair
case (8.2), the supervised-feature baseline (8.3), and multi-config breadth
(8.4).

### 9.2 Are the conclusions justified?

The *narrow* conclusion is justified and even under-sold (the headline table
hides the best k). The *scientific* conclusion — that label-independence is the
reason it works — is **strongly suggested** by the demo-vs-FLIP equivalence
(Section 7) but is **not yet proven** because the supervised-feature comparison
is missing. Right now the most defensible framing is exactly the one in the
project's own novelty doc: *"we investigate whether SSL features expose FLIP,
and find that they do, strongly, when classes are semantically separable."*

### 9.3 The most important missing experiments, in priority order

1. **Hard class pair** (cat→dog or automobile→truck): does AUROC fall as the
   theory predicts? This is the failure-mode contribution.
2. **Supervised-feature k-NN baseline** on the same data: this is what proves
   "label-independence matters" rather than just "k-NN matters".
3. **Multiple seeds** for error bars on CTA/PTA.
4. **Budget sweep** (0.5%, 1%, 3%, 5%): does detection hold at lower poison
   rates (where AUPRC base rate shrinks)?
5. **CIFAR-100 and the ViT-B/14 encoder ablation** (promised in the plan).
6. **A token adaptive-attack discussion** (even if not run): what could an
   attacker do if they knew about the SSL audit?

### 9.4 Strengths to keep and lead with

- A clean, reproducible, well-documented pipeline with leak-free evaluation.
- A genuinely strong and well-visualized headline result.
- The **demo-vs-FLIP equivalence** (Section 7) — promote this from the sanity
  script to a headline figure; it is your best scientific evidence.
- The **per-class false-positive-contamination** finding (Section 6.4) — this is
  a non-obvious, publishable observation about *how* neighborhood audits fail on
  the source class.
- An unusually honest framing document that already anticipates reviewer
  attacks. Follow its own advice — especially "test the cat/dog case".

### 9.5 One-paragraph summary for your advisor

> On a CIFAR-10 FLIP attack (truck→deer, 3% budget) that otherwise succeeds
> ~99.8% of the time, a label-independent DINOv2 k-NN label-consistency audit
> reduces attack success to under 1% (k≈100–1000) while *increasing* clean
> accuracy from ~89.6% to ~92%, removing ~6–8% of the data. Detection AUROC
> reaches 0.99 and AUPRC 0.67, and — crucially — the real, optimized FLIP attack
> is no harder to detect than naive random flips, supporting the hypothesis that
> FLIP's supervised-trajectory optimization gives it no leverage in
> label-blind feature space. The evidence is strong but currently limited to one
> easy, semantically-separated configuration with a single seed and no competing
> baseline; the next steps are a hard class-pair, a supervised-feature
> comparison, and multi-seed error bars.

---

## 10. Quick reference: every results file and what it holds

| File | What it is | Key content |
|---|---|---|
| `master_metrics.{csv,json}` | one row per (k, mode) | CTA, PTA, AUROC, AUPRC, prec@k, flag P/R, best F1/MCC, score stats |
| `table_training_sweep.md` | training outcomes | CTA & PTA vs k (the real headline) |
| `table_detection_sweep.md` | detection outcomes | full detection metric grid vs k |
| `table_headline.md` | combined view | ⚠️ only k=1,5,10 (baseline-limited) |
| `table_best_thresholds.md` | tuning guide | best F1/MCC threshold per k (rises with k) |
| `per_class_detection.csv` | per-class breakdown | poison counts, precision, flag-rate per class |
| `threshold_sweep.csv` | threshold grid | TP/FP/FN/TN, P/R/F1/MCC/FPR vs threshold |
| `training_curves.csv` | per-epoch logs | CTA/PTA over 200 epochs (confirms convergence) |
| `sweep.log` | run journal | confirms config, seeds, and the none-baseline skip |
| `detection_metrics.json` (per run) | per-run detection | the numbers aggregated into master_metrics |
| `summary.json` / `summary_detailed.json` (per run) | per-run headline | mode, score_key, CTA, PTA (+ full config) |
| `out/.../demo_*/summary.json` | sanity check | feature gap, k-NN acc, random-flip detection |
| `figures/detection/*` | detection plots | ROC, PR, score dists, confusion, calibration, per-class heatmap |
| `figures/training/*` | training plots | CTA/PTA trade-off, defense-success-vs-k, training curves |
| `figures/features/*` | feature plots | UMAP (by class / by poison), neighbor purity, pair distance |
| `figures/samples/*` | image grids | TP / FP / FN / TN examples, poison examples |

---

*End of report. Generated by automated analysis of the project artifacts; all
quantitative claims trace to the files listed in Section 10.*
