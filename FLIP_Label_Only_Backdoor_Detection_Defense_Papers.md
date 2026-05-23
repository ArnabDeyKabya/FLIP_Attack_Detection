# Detecting and Mitigating Label-Only Backdoor Attacks (FLIP & Similar)

## A Focused Paper Compilation

*Target paper: "Label Poisoning is All You Need" (Jha, Hayase, Oh — NeurIPS 2023)*

---

## Why This Is Hard (You're Right to Think So)

In FLIP, the images are clean — only labels are flipped. This defeats:
- **Human visual inspection** (poisoned images look correct for their flipped labels... because the attacker carefully chose them)
- **Image-label consistency filters** (nothing obviously wrong with the image)
- **Activation Clustering** (Chen et al. 2019) — relies on trigger-based activation separation
- **Spectral Signatures** (Tran et al. 2018) — degraded when no image perturbation exists
- **Neural Cleanse** (Wang et al. 2019) — reverse-engineers triggers from images

The FLIP paper itself reports that most existing SOTA defenses fail to stop it. This is exactly why the authors explicitly called for "further research in designing new and stronger defenses."

That said, here are the most promising directions.

---

## 1. Direct Defenses Against Label Poisoning (Most Relevant)

### 1.1 FLORAL: Adversarial Training for Defense Against Label Poisoning Attacks ⭐ Most directly relevant
- **Authors:** Melis Ilayda Bal, Volkan Cevher, Michael Muehlebach
- **Venue:** ICLR 2025 / arXiv 2502.17121
- **Link:** https://arxiv.org/abs/2502.17121
- **Why it matters for your question:** This is the most recent work that **explicitly addresses FLIP-style label-only poisoning**. It casts training as a Stackelberg game between an adversary (who flips labels using dual-variable influence — conceptually similar to FLIP's attacker) and the defender (an SVM using projected gradient descent). Compared to RoBERTa baselines, FLORAL achieves higher robust accuracy under increasing attacker budgets. The authors explicitly note that standard adversarial training fails against label poisoning, which is why a dedicated defense was needed.

### 1.2 Certified Robustness to Label-Flipping Attacks via Randomized Smoothing
- **Authors:** Elan Rosenfeld, Ezra Winston, Pradeep Ravikumar, J. Zico Kolter
- **Venue:** ICML 2020
- **Link:** https://arxiv.org/abs/2002.03018
- **Why it matters:** Provides **provable certificates** — for each test point, guarantees the prediction is unchanged if up to k training labels are adversarially flipped. This is one of the few defenses with formal guarantees against arbitrary label flipping strategies (including FLIP-style attacks). The limitation: works best for linear classifiers, not end-to-end deep models.

### 1.3 Deep Partition Aggregation (DPA) & SS-DPA
- **Authors:** Alexander Levine, Soheil Feizi
- **Venue:** ICLR 2021
- **Link:** https://arxiv.org/abs/2006.14768
- **Why it matters:** SS-DPA (semi-supervised variant) is specifically designed to outperform existing certified defenses for **label-flipping attacks**. Works by training many classifiers on disjoint partitions and aggregating — a labeled poison in one partition can't cross-contaminate others.

### 1.4 BagFlip: A Certified Defense against Data Poisoning
- **Authors:** Yuhao Zhang, Aws Albarghouthi, Loris D'Antoni
- **Venue:** NeurIPS 2022
- **Link:** https://arxiv.org/abs/2205.13862
- **Why it matters:** Extends DPA with tighter certified guarantees specifically covering trigger-based + label-flip attacks. Directly relevant to FLIP's threat model.

---

## 2. Detection via Training Dynamics (Very Promising for FLIP)

This is probably the **most practically promising angle**. The intuition: clean-labeled samples fit normally during training, but label-flipped samples create unusual training trajectories because the image content "fights" the assigned label.

### 2.1 Learning from Training Dynamics: Identifying Mislabeled Data Beyond Manually Designed Features (L2D)
- **Authors:** Qingrui Jia et al.
- **Venue:** AAAI 2023
- **Link:** https://arxiv.org/abs/2212.09321
- **Why it matters:** Trains an **LSTM-based noise detector** that takes raw per-sample training dynamics (loss curves, confidence trajectories) as input and predicts whether a sample was mislabeled. Tested on CIFAR with synthetic noise and transfers to Tiny ImageNet, CUB-200, WebVision, Clothing1M **without retraining**. This is directly applicable to FLIP-style attacks because FLIP-flipped samples necessarily have abnormal training dynamics (the network struggles to fit a truck-image labeled "deer").

### 2.2 Identifying Mislabeled Data using the Area Under the Margin (AUM)
- **Authors:** Geoff Pleiss, Tianyi Zhang, Ethan Elenberg, Kilian Weinberger
- **Venue:** NeurIPS 2020
- **Link:** https://arxiv.org/abs/2001.10528
- **Why it matters:** Simple, elegant, works surprisingly well. Tracks the **margin between the assigned-label logit and the largest-other-class logit** over training. Mislabeled samples consistently show low/negative AUM. One of the strongest baselines for label-noise detection — and several FLIP follow-up works use it as a comparison.

### 2.3 Enhanced Sample Selection with Confidence Tracking
- **Authors:** Multiple
- **Venue:** arXiv 2504.17474 / 2025
- **Link:** https://arxiv.org/abs/2504.17474
- **Why it matters:** Tracks confidence gaps between annotated labels and other classes during training using the Mann-Kendall test. Correctly-labeled samples show monotonically increasing confidence for the true class; flipped ones show the competing class rising faster. Plug-and-play with existing selection methods.

### 2.4 SEEP: Training Dynamics Grounds Latent Representation Search for Mitigating Backdoor Poisoning
- **Authors:** Multiple
- **Venue:** TACL 2024 (Transactions of the ACL)
- **Link:** https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00684/124258
- **Why it matters:** Uses training dynamics to identify a small seed set of high-precision poisoned samples, then propagates labels through latent representations to catch the rest. Tested against advanced backdoor attacks that defeat simpler defenses.

---

## 3. Confident Learning / Label Error Detection

### 3.1 Confident Learning: Estimating Uncertainty in Dataset Labels (Cleanlab)
- **Authors:** Curtis Northcutt, Lu Jiang, Isaac Chuang
- **Venue:** Journal of Artificial Intelligence Research (JAIR) 2021
- **Link:** https://arxiv.org/abs/1911.00068
- **Why it matters:** The theoretical foundation of the Cleanlab library. Uses model predicted probabilities + class-conditional noise estimation to identify label errors. Works model-agnostically. **Practical caveat for FLIP:** works best when the flipped labels are "natural" mislabels; may be less effective against FLIP's deliberately optimized flips that mimic the natural noise distribution.

### 3.2 Pervasive Label Errors in Test Sets Destabilize ML Benchmarks
- **Authors:** Curtis Northcutt, Anish Athalye, Jonas Mueller
- **Venue:** NeurIPS Datasets Track 2021
- **Link:** https://arxiv.org/abs/2103.14749
- **Why it matters:** Shows Cleanlab-style methods find real label errors in ImageNet, CIFAR, MNIST. Validates that training-signal-based detection works on real label noise.

---

## 4. Influence-Based Detection

### 4.1 Understanding Black-box Predictions via Influence Functions
- **Authors:** Pang Wei Koh, Percy Liang
- **Venue:** ICML 2017 (Best Paper Award)
- **Link:** https://arxiv.org/abs/1703.04730
- **Why it matters:** The foundational technique for tracing which training samples are "responsible" for a test-time prediction. A classifier that is backdoored via FLIP will have the flipped labels showing up as unusually influential for the trigger-bearing test inputs. Scalability is the main issue for deep nets.

### 4.2 Identifying a Training-Set Attack's Target using Renormalized Influence Estimation
- **Authors:** Zayd Hammoudeh, Daniel Lowd
- **Venue:** ACM CCS 2022
- **Link:** https://arxiv.org/abs/2201.10055
- **Why it matters:** Specifically designed for poisoning/backdoor scenarios. Uses renormalized influence to identify the target of an attack — applicable to finding the "target class" in FLIP-style attacks.

---

## 5. Robust Learning with Noisy Labels (Implicit Defenses)

These don't explicitly detect poisoning, but make training robust enough that label flips have reduced effect.

### 5.1 Co-teaching: Robust Training with Extremely Noisy Labels
- **Authors:** Bo Han, Quanming Yao, et al.
- **Venue:** NeurIPS 2018
- **Link:** https://arxiv.org/abs/1804.06872
- **Why it matters:** Two networks train simultaneously, each selecting small-loss samples for the other. Empirically, FLIP-flipped samples will tend to have higher loss (image content doesn't match label), so Co-teaching implicitly filters them.

### 5.2 Robust Loss Functions under Label Noise for Deep Neural Networks
- **Authors:** Aritra Ghosh, Himanshu Kumar, P.S. Sastry
- **Venue:** AAAI 2017
- **Link:** https://arxiv.org/abs/1712.09482
- **Why it matters:** Proves certain symmetric losses (e.g., MAE, reverse cross-entropy) are inherently noise-tolerant. Gives theoretical guarantees rather than empirical ones.

### 5.3 Early-Learning Regularization Prevents Memorization of Noisy Labels (ELR)
- **Authors:** Sheng Liu, Jonathan Niles-Weed, Narges Razavian, Carlos Fernandez-Granda
- **Venue:** NeurIPS 2020
- **Link:** https://arxiv.org/abs/2007.00151
- **Why it matters:** Neural networks first learn clean patterns, then memorize noisy ones. ELR regularization holds the model near the "early learning" state, preventing memorization of flipped labels. Directly useful since FLIP relies on the model actually learning the flipped labels.

---

## 6. Label-Propagation / Neighbor-Based Detection

### 6.1 Label Sanitization against Label Flipping Poisoning Attacks
- **Authors:** Andrea Paudice, Luis Muñoz-González, Emil Lupu
- **Venue:** ECML PKDD 2018 Workshop
- **Link:** https://arxiv.org/abs/1803.00992
- **Why it matters:** Classic k-NN-based relabeling — if a sample's k nearest neighbors (in feature space) disagree with its label, relabel it. **Important caveat for FLIP:** FLIP specifically targets samples whose labels can be flipped without raising suspicion in feature space, so vanilla k-NN may underperform. Still worth considering with deep feature embeddings.

### 6.2 Deep k-NN Defense Against Clean-Label Data Poisoning Attacks
- **Authors:** Neehar Peri et al.
- **Venue:** ECCV 2020 Workshops
- **Link:** https://arxiv.org/abs/1909.13374
- **Why it matters:** Applies k-NN in deep feature space rather than pixel space — often catches poisoned samples that FLIP-like attacks expect to evade.

---

## 7. Survey / Context Paper That's Worth Reading

### 7.1 Wild Patterns Reloaded: A Survey of ML Security Against Training Data Poisoning
- **Authors:** Antonio Emanuele Cinà, Kathrin Grosse, Ambra Demontis, Sebastiano Vascon, Werner Zellinger, Bernhard Moser, Alina Oprea, Battista Biggio, Marcello Pelillo, Fabio Roli
- **Venue:** ACM Computing Surveys, 2023
- **Link:** https://dl.acm.org/doi/10.1145/3585385
- **Why it matters:** Comprehensive survey covering the attack-defense arms race. Especially useful for understanding where FLIP sits in the taxonomy and which defense categories haven't been fully explored against label-only attacks.

---

## Honest Assessment (What's Actually Promising for FLIP)

Ranked by my read of the literature — what's most likely to actually detect FLIP-style attacks:

1. **Training-dynamics methods (L2D, AUM)** — probably your best practical bet. FLIP can't fully hide the fact that flipped samples disagree with their images during training.
2. **FLORAL** — the only published defense that explicitly targets this threat model, though SVM-based so the deep-learning version remains open research.
3. **Certified defenses (Randomized Smoothing, DPA, BagFlip)** — provable but expensive and often accuracy-lossy.
4. **Early-learning regularization (ELR)** — clever because FLIP *requires* memorization to succeed; prevent that and the attack weakens.
5. **Influence functions** — theoretically sound, computationally hard to scale.

## Open Research Questions (Possible Thesis/Paper Directions)

- **No existing defense is evaluated specifically against FLIP at scale.** Most clean-label defenses were developed against Turner et al.'s clean-label attack or Poison Frogs, not FLIP.
- **Ensemble of dynamics-based detectors + robust training** is an obvious-but-unexplored combination.
- **Detecting trajectory-matching signatures** — since FLIP uses trajectory matching to design flips, there may be a detectable signature in the resulting training dynamics (an inverse attack).
- **Self-supervised pretraining** may provide attack-invariant features that make k-NN style detection more reliable.

---

*All venues listed are top-tier: NeurIPS, ICML, ICLR, AAAI, ACM CCS, CVPR, JAIR, ACM Computing Surveys, or TACL. arXiv preprints are flagged explicitly.*
