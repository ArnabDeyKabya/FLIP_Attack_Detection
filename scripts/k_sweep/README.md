# Deep k-NN defense — thesis evaluation sweep

A complete, reproducible pipeline that takes the existing
`train_user_defense` module and runs it across a sweep of `k` values in both
`none` (baseline) and `remove` (defended) modes, then generates every metric,
curve, image, and table needed for the thesis chapter on the defense.

The sweep is built on top of the existing `run_experiment.py` and the existing
`train_user_defense` schema — no changes to the upstream modules are required.

---

## Quick start

```bash
# Full sweep — runs every (k, mode) combination, then produces all artifacts.
bash scripts/k_sweep/run_full_sweep.sh

# Plots/tables only — assumes the experiments already ran.
bash scripts/k_sweep/run_full_sweep.sh --skip-train

# Narrow sweep, force re-train.
bash scripts/k_sweep/run_full_sweep.sh --k-values "5 20 100 500" --force
```

PowerShell variant:

```powershell
pwsh -File scripts\k_sweep\run_full_sweep.ps1
pwsh -File scripts\k_sweep\run_full_sweep.ps1 -SkipTrain
pwsh -File scripts\k_sweep\run_full_sweep.ps1 -KValues 5,20,100,500 -Force
```

Default sweep: `k ∈ {1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000}`,
`mode ∈ {none, remove}`. Each `k` adds two trainings: a baseline (no removal)
and a defended (auto-threshold removal). On a single GPU the full sweep takes
roughly the same time as 20 vanilla `train_user` runs — generally ~6-12h on a
single RTX 3090.

---

## What gets produced

After a full sweep, `experiments/_report_<name_prefix>/` contains:

```
master_metrics.csv               # one row per (k, mode) — every aggregated metric
master_metrics.json              # same in JSON
threshold_sweep.csv              # fine threshold grid per experiment
per_class_detection.csv          # per (k, mode, class) confusion stats
training_curves.csv              # per (k, mode, epoch) CTA/PTA/loss
sweep.log                        # tee'd stdout of the whole sweep
figures/
    detection/
        roc_all_k.{png,pdf}
        pr_all_k.{png,pdf}
        k_sweep_detection.{png,pdf}        # AUROC/AUPRC/best F1/best MCC/P@k/R@k vs k
        k_sweep_flagged.{png,pdf}          # flagged precision/recall vs k
        k_sweep_removed.{png,pdf}          # poisons vs clean removed vs k
        score_dist_facet.{png,pdf}         # small-multiples per-k
        per_class_flag_heatmap.{png,pdf}
        score_dist_k{k}.{png,pdf}          # one per k (linear + log scale)
        confusion_matrix_k{k}.{png,pdf}
        threshold_sweep_k{k}.{png,pdf}
        calibration_k{k}.{png,pdf}
    training/
        training_curve_k{k}.{png,pdf}
        training_curve_facet_cta.{png,pdf}
        training_curve_facet_pta.{png,pdf}
        k_sweep_cta_pta.{png,pdf}
        cta_pta_tradeoff.{png,pdf}
        defense_success_vs_k.{png,pdf}
        bar_compare_none_remove.{png,pdf}
    features/
        features_tsne_class.{png,pdf}
        features_tsne_poison.{png,pdf}
        features_tsne_score_k{k}.{png,pdf}
        features_umap_*.{png,pdf}          # if umap-learn installed
        features_pair_distance.{png,pdf}
        features_neighbor_purity_k{k}.{png,pdf}
    samples/
        samples_TP_k{k}.{png,pdf}          # correctly flagged poisons
        samples_FN_k{k}.{png,pdf}          # missed poisons
        samples_FP_k{k}.{png,pdf}          # false alarms
        samples_TN_k{k}.{png,pdf}          # correctly kept clean
        samples_top_scores_k{k}.{png,pdf}
        samples_target_class_overview.{png,pdf}
        samples_poison_examples.{png,pdf}
tables/
    table_headline.{md,tex}                # 1-page main result
    table_detection_sweep.{md,tex}
    table_training_sweep.{md,tex}
    table_best_thresholds.{md,tex}
    table_per_class_{remove,none}.md
```

Each LaTeX table uses `booktabs` and is `\input{}`-ready.

---

## Pipeline stages

The shell script wires together six stages; each is also a standalone
Python entrypoint you can re-run by itself.

| # | Script                                          | What it does                                                                          |
|---|-------------------------------------------------|---------------------------------------------------------------------------------------|
| 1 | `scripts/k_sweep/generate_config.py`            | Writes per-(k, mode) `experiments/<name>/config.toml` files.                          |
| 2 | `run_experiment.py <name>`                      | (Existing) trains the user model + saves detection artifacts.                          |
| 3 | `scripts/k_sweep/aggregate_results.py`          | Builds master CSV + JSON tables consumed by every plot script.                         |
| 4 | `scripts/k_sweep/plot_detection.py`             | All detection-side figures.                                                            |
| 5 | `scripts/k_sweep/plot_training.py`              | Training-dynamics, CTA/PTA, defense-success figures.                                   |
| 6 | `scripts/k_sweep/plot_features.py`              | t-SNE / UMAP / pair-distance / neighbor-purity figures.                                |
| - | `scripts/k_sweep/plot_samples.py`               | Sample-image grids (TP / FP / FN / TN, top scores, target-class).                      |
| - | `scripts/k_sweep/make_tables.py`                | LaTeX + Markdown tables.                                                               |

Idempotent: stage 2 skips any experiment whose `summary.json` already exists.
Pass `--force` to re-train.

---

## Configuration knobs

The orchestrator forwards every flag to `generate_config.py`:

| Flag                  | Default                                         | Description                                                    |
|-----------------------|-------------------------------------------------|----------------------------------------------------------------|
| `--k-values`          | `1 5 10 20 50 100 200 500 1000 2000`            | Sweep over these k values.                                     |
| `--modes`             | `none remove`                                   | Defense modes per k.                                           |
| `--dataset`           | `cifar`                                         | `cifar` or `cifar_100`.                                        |
| `--user-model`        | `r32p`                                          | Victim architecture.                                           |
| `--trainer`           | `sgd`                                           | `sgd` or `adam`.                                               |
| `--poisoner`          | `1xs`                                           | FLIP poisoner flag (sinusoidal trigger here).                  |
| `--budget`            | `1500`                                          | # label flips (FLIP attack budget).                            |
| `--source-label`      | `9` (truck)                                     | FLIP source class.                                             |
| `--target-label`      | `4` (deer)                                      | FLIP target class.                                             |
| `--encoder`           | `dinov2_vits14`                                 | SSL encoder for k-NN feature space.                            |
| `--scoring`           | `disagreement`                                  | `disagreement` or `weighted_disagreement`.                     |
| `--removal-count`     | `auto`                                          | `auto`, `budget` (oracle), or an int top-N.                    |
| `--threshold`         | `0.5`                                           | `auto`-mode threshold.                                         |
| `--seed`              | `0`                                             | Reproducibility seed.                                          |
| `--skip-train`        | —                                               | Plots only, no retraining.                                     |
| `--skip-plots`        | —                                               | Aggregate + tables only.                                       |
| `--skip-tables`       | —                                               | Skip table generation.                                         |
| `--skip-none`         | —                                               | Only run `remove`-mode trainings.                              |
| `--force`             | —                                               | Re-train even if summary.json exists.                          |

---

## Why this set of metrics, curves, and images?

The metric panel mirrors what label-poisoning defense papers report:

- **Detection-side** (Peri et al. 2020 §5–6, Floral 2024 §5):
  AUROC, AUPRC, P@k, R@k, F1, MCC, balanced accuracy, confusion matrix,
  threshold sweep, calibration. The MCC + best-F1 are the operating-point
  free measures the Deep k-NN paper specifically recommends.
- **Training-side** (FLIP 2023 §5, Floral 2024 §6.2):
  CTA, PTA, the (CTA, PTA) scatter trade-off, per-epoch trajectories,
  CTA/PTA drop vs k as the defense-success summary.
- **Geometric** (Peri et al. 2020 Fig. 1, 3; Floral 2024 Fig. 1):
  t-SNE / UMAP projections, pairwise SSL similarity distributions,
  neighbor purity distributions — the qualitative evidence the
  "neighborhood-consistency" assumption actually holds.
- **Qualitative** (FLIP 2023 Fig. 8, Floral 2024 Fig. 4):
  TP / FP / FN / TN image grids, target-class overview, FLIP poison
  examples.

---

## Cost and runtime notes

Training (stage 2) is by far the most expensive stage; each run
fine-tunes a ResNet-32p on ~50k CIFAR-10 images for ~200 epochs.
On a single RTX 3090 that's ~25 min/run. With the default 10 ks ×
2 modes = 20 trainings, the full sweep takes ~8h. To shorten:

- `--skip-none` halves the wall-clock — the `none` baseline is
  k-independent for the trained model (only the logged detection
  metrics change with k), so you only need one `none` run.
- `--k-values "5 20 100 500"` is a good 4-point sweep that still
  produces a meaningful curve.

Detection scoring inside `run_experiment.py` is essentially free
(a few seconds per k after the SSL feature cache is warm).

All plotting stages are cheap (~minutes total) and can be re-run
independently of training via `--skip-train`.

---

## Files in this directory

```
scripts/k_sweep/
    README.md                # this file
    run_full_sweep.sh        # bash orchestrator
    run_full_sweep.ps1       # PowerShell orchestrator
    generate_config.py       # per-(k, mode) config writer
    aggregate_results.py     # CSV + JSON aggregator
    plot_detection.py        # ROC, PR, score dist, k-sweep, confusion, calibration
    plot_training.py         # CTA/PTA, training curves, trade-off
    plot_features.py         # t-SNE / UMAP / distance / purity
    plot_samples.py          # TP / FP / FN / TN image grids
    make_tables.py           # LaTeX + Markdown tables
```
