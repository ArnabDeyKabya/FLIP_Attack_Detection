# Novelty, Framing, and Contribution

## How to Present Your Thesis Honestly and Strongly

*For thesis proposal, defense, and paper submission*

---

## 1. The One-Sentence Pitch

**Use this as your thesis abstract's first sentence and in every introduction:**

> We systematically evaluate whether self-supervised representations provide robustness against label-only backdoor attacks such as FLIP, and propose a pre-training label auditing framework that exposes inconsistencies between poisoned labels and semantic feature neighborhoods.

This sentence does five things at once:
- Frames the work as scientific investigation (not a novel algorithm claim)
- Names the specific threat (FLIP, label-only)
- Identifies the specific tool (self-supervised representations)
- Delivers a concrete contribution (auditing framework)
- Avoids overclaiming

Memorize this sentence. Use it in your proposal, thesis introduction, paper abstract, and defense slides.

---

## 2. What Is Actually Novel (Be Honest)

Your novelty lies in four places. Know them exactly. Don't claim more than this.

### 2.1 Problem-Level Novelty

You are the first to systematically ask: **does self-supervised feature space expose FLIP-style label-only poisoning?** This is a previously unasked research question. The FLIP authors tested defenses operating on images or activations from the poisoned model, but not external semantic representations.

**Why this matters:** FLIP optimizes its label flips against supervised training trajectories. Self-supervised features are computed independently of labels, so FLIP's optimization has no direct leverage against them. Testing this hypothesis is scientifically meaningful regardless of the outcome.

### 2.2 Empirical Novelty

You produce the first empirical characterization of **which FLIP-style poisoned samples are detectable and which are not** by semantic auditing. Your failure-mode analysis (which class pairs are hardest, why) is itself a contribution because no prior work has mapped this terrain.

### 2.3 Methodological Novelty (Modest)

You combine self-supervised representations with neighborhood consistency auditing. Each piece exists separately in prior work:
- SSL encoders (DINO, SimCLR) — existing
- k-NN label consistency (Deep k-NN defense) — existing
- Feature-based anomaly detection — existing

**Your specific combination applied to label-only backdoors is new, but it's an incremental combination, not a new paradigm.** Be honest about this in your methodology section.

### 2.4 Framework Novelty

You propose a complete **pre-training auditing pipeline** — feature extraction, per-sample scoring, calibrated label trust, and downstream mitigation — specifically targeting FLIP-style attacks. Prior work addresses these pieces in isolation; you integrate them for this specific threat.

---

## 3. What You Should NOT Claim

Do not say these things. Each one will be attacked by a careful reviewer:

**Do not claim:** "We propose a novel defense algorithm."  
**Reason:** Your algorithm is k-NN in feature space. That's not novel. The novelty is the evaluation and framing.

**Do not claim:** "Our method detects label poisoning without any supervised information."  
**Reason:** Your SSL encoder was trained on data (ImageNet/LAION) whose distribution overlaps with your evaluation classes. You use external semantic knowledge, just not labels from the poisoned set.

**Do not claim:** "Our defense is robust to adaptive attacks."  
**Reason:** You don't test this. Don't promise what you haven't shown.

**Do not claim:** "We solve the FLIP attack."  
**Reason:** Your method has known failure modes (semantically similar classes). Acknowledge this prominently.

**Do not claim:** "Our method is state-of-the-art."  
**Reason:** You probably haven't compared against every defense. Say "competitive with strong baselines" or "outperforms baselines we evaluated."

---

## 4. How to Frame Each Section of Your Thesis

### 4.1 Introduction Framing

Structure your introduction in this logical chain:

1. **Threat:** Modern ML trains on untrusted data from crowd-sourcing and third-party annotators.
2. **Specific threat:** Jha et al. (NeurIPS 2023) introduced FLIP — a label-only backdoor attack that flips <2% of labels and achieves >99% attack success, and for which existing defenses are inadequate.
3. **Observation:** FLIP optimizes label flips against supervised training, but has no mechanism to hide in semantic feature space.
4. **Question:** Can semantic feature consistency, computed via label-independent representations, detect FLIP-poisoned samples before training?
5. **Contribution:** We answer this question empirically, analyze when and why it succeeds or fails, and propose a practical auditing framework.

This framing positions you as a researcher asking a question, not a vendor selling an algorithm. Much stronger.

### 4.2 Related Work Framing

Organize around three threads:
- Backdoor attacks (BadNets → clean-label → FLIP)
- Label error and noisy-label detection (AUM, Confident Learning, Deep k-NN)
- Self-supervised representation learning (SimCLR, DINO)

Then end the related work section with an explicit gap statement:

> While prior work has explored label error detection in benign noise settings and defense against traditional backdoor attacks, no systematic study examines whether self-supervised representations can detect label-only backdoor attacks specifically designed to evade supervised-feature defenses. This thesis fills that gap.

### 4.3 Method Framing

Present the method modestly and clearly. Do not call it "novel" — call it "adapted" or "applied to this setting."

Suggested language: "We adapt neighborhood-consistency auditing, previously used for label error detection, to the specific setting of label-only backdoor attacks by computing consistency in self-supervised feature space."

### 4.4 Results Framing

Lead with honest observations, not victory claims:

Good: "We find that self-supervised feature auditing detects X% of FLIP-poisoned samples at Y% precision, outperforming supervised-feature baselines by Z points on AUPRC."

Bad: "Our method dramatically outperforms all baselines."

### 4.5 Limitations Framing

Be proactive and explicit. A dedicated limitations section with these points:

1. Semantic overlap: class pairs like cat/dog, truck/car reduce detection rate.
2. Pretrained encoder dependence: requires access to a foundation SSL model; features may implicitly encode class semantics from large-scale pretraining.
3. No adaptive-attack evaluation: a sophisticated attacker aware of this defense might optimize flips against semantic consistency.
4. Dataset scope: evaluation limited to CIFAR-10/100; larger datasets may behave differently.

Reviewers respect honesty. Hidden weaknesses get exposed in reviews.

---

## 5. Your Contribution Paragraph (For Proposal/Paper)

Use this template, filling in X/Y/Z with your actual results:

> **Our contributions are:**
> 1. **Problem formulation:** We formalize pre-training label auditing for label-only backdoor attacks as a neighborhood-consistency problem in label-independent feature space.
> 2. **Empirical study:** We present the first systematic evaluation of self-supervised feature auditing against FLIP, testing across poisoning rates (0.5%–5%), encoder choices (DINOv2, SimCLR), and datasets (CIFAR-10, CIFAR-100).
> 3. **Comparative analysis:** We demonstrate that self-supervised features outperform supervised features (trained on poisoned data) by X points AUPRC, establishing that label-independence of features matters for detecting label-only attacks.
> 4. **Failure characterization:** We identify class-pair-specific failure modes (e.g., cat/dog, truck/car) and quantify detection degradation as a function of semantic class distance.
> 5. **Mitigation pipeline:** We integrate our auditing method with both hard removal and soft downweighting, reducing attack success rate from ~99% to Y% while preserving Z% of clean test accuracy.

If you can deliver on these five points, you have a thesis worth defending and a paper worth submitting.

---

## 6. Anticipating Reviewer Attacks (And Preparing Answers)

Your advisor and external reviewers will ask these questions. Prepare answers now.

**Q: "Why is this not just Deep k-NN (Peri et al. 2020)?"**  
A: Deep k-NN uses supervised features from a model trained on the poisoned dataset — features that are themselves corrupted. We use features from a label-independent self-supervised encoder, and we show empirically that this distinction matters for label-only attacks specifically.

**Q: "Your SSL encoder was trained on ImageNet, which contains cat/dog/truck images. You're using supervised-distribution knowledge."**  
A: True that pretraining data overlaps with evaluation classes. However, no *labels* from our poisoned dataset ever inform the feature extractor. This is a meaningfully different setting than using features from a classifier trained on the poisoned labels. We discuss this nuance explicitly in Section [X].

**Q: "What if the attacker knows you're using SSL features and adapts the attack?"**  
A: We do not evaluate adaptive attacks; this is a limitation we acknowledge. An adaptive attacker could attempt to select flips that are semantically plausible in SSL feature space, but this constrains the attacker significantly and is left to future work.

**Q: "Why not combine multiple signals (AUM, CL, k-NN)?"**  
A: We chose to study one signal in depth to isolate whether SSL feature space alone exposes FLIP. Multi-signal ensembles are a natural extension, but would obscure the core scientific question we address.

**Q: "CIFAR-10 is small. Does this scale?"**  
A: We evaluate on CIFAR-10 and CIFAR-100. Scaling to ImageNet-scale evaluation was beyond our compute budget but is clearly future work.

**Q: "Your detection rate isn't 100%. Is this really a defense?"**  
A: No defense in this literature achieves perfect detection. We report precision/recall curves and show that even partial detection, when combined with mitigation, reduces attack success substantially while preserving clean accuracy. We position this as an auditing tool, not a perfect filter.

---

## 7. Positioning Against FLIP Authors' Implicit Challenge

The FLIP paper explicitly calls for new defenses. You are responding to that call. State this directly:

> "Jha et al. conclude their NeurIPS 2023 paper by calling for new defenses against label-only backdoor attacks, noting that existing defenses based on model activations and image anomalies are inadequate. This thesis responds to that call by investigating whether external, label-independent representations provide the distinguishing signal that model-internal defenses lack."

Using their own words to motivate your work is stronger than positioning yourself as a competitor.

---

## 8. The Final Framing Hierarchy

Think of your contribution at three levels:

**Level 1 (Technical):** A k-NN audit in self-supervised feature space with calibrated scoring and downweighting-based mitigation.

**Level 2 (Methodological):** A pre-training label auditing framework applicable to label-only backdoor attacks.

**Level 3 (Scientific):** An empirical investigation of whether label-independent representations expose poisoning that label-dependent representations cannot.

When writing for your thesis, lead with Level 3 in the introduction and Level 2 in the method section. Level 1 goes in implementation details. Presenting the technical details as the "main contribution" weakens your work. Presenting the scientific question as the main contribution strengthens it.

---

## 9. Words to Use vs. Words to Avoid

**Use these words (they sound researcher-like):**
- "We investigate whether..."
- "We systematically evaluate..."
- "We provide empirical evidence that..."
- "Our analysis reveals..."
- "We characterize the conditions under which..."
- "Our approach adapts..."

**Avoid these words (they oversell):**
- "Novel algorithm"
- "Breakthrough"
- "Solves the problem of..."
- "Robust defense"
- "State-of-the-art"
- "Complete solution"

Subtle changes in vocabulary substantially affect how reviewers perceive your work. Researcher language earns trust; marketing language invites skepticism.

---

## 10. Summary: Your Thesis in Three Bullets

If your advisor asks "what is your thesis in three bullets," say:

- Label-only backdoor attacks like FLIP corrupt labels in ways that evade existing defenses, which operate on images or activations from the poisoned model.
- We investigate whether features from a self-supervised encoder — computed independently of labels — expose these corruptions via neighborhood consistency auditing.
- Our empirical study shows [when this works, when it fails, and by how much], yielding the first systematic characterization of semantic-space detectability for label-only backdoor attacks.

Memorize this answer. Deliver it with confidence in your defense.

---

## Final Note

Your thesis is not weak. Your thesis is honest. In research, those are the same thing done properly. The students who overclaim get torn apart in defenses. The students who frame their work accurately pass with strong evaluations even when their results are modest.

You are positioned correctly. Now execute.
