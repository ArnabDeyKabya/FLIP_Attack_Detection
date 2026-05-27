#!/usr/bin/env bash
# =============================================================================
# Deep k-NN defense — full thesis sweep
# =============================================================================
# Runs the FLIP-poisoned CIFAR-10 user-model training across a sweep of k
# values, in both `none` (no removal) and `remove` (filtered) modes, then
# generates every metric / curve / image / table needed for the thesis.
#
# Stages
#   1. generate_config.py  -> per-(k, mode) configs under experiments/
#   2. run_experiment.py   -> training + detection per config
#   3. aggregate_results.py -> master CSV + JSON of all summary metrics
#   4. plot_detection.py    -> ROC, PR, score histograms, k-sweep detection
#                              metric curves, confusion matrices, threshold
#                              sweeps, per-class breakdown, calibration
#   5. plot_training.py     -> training curves (per-epoch CTA/PTA),
#                              CTA-PTA trade-off scatter, bar comparisons
#   6. plot_features.py     -> t-SNE / UMAP of SSL features (poison/clean,
#                              per-class), within/between-class distance
#                              distributions
#   7. plot_samples.py      -> sample image grids (TP/FP/FN/TN at chosen k)
#   8. make_tables.py       -> LaTeX + Markdown tables for the paper
#
# Usage
#   bash scripts/k_sweep/run_full_sweep.sh
#   bash scripts/k_sweep/run_full_sweep.sh --skip-train         # plots only
#   bash scripts/k_sweep/run_full_sweep.sh --k-values "5 20 100"
#   bash scripts/k_sweep/run_full_sweep.sh --modes "remove"
#   bash scripts/k_sweep/run_full_sweep.sh --skip-none          # remove only
#
# The script is idempotent: an experiment directory that already contains
# summary.json is skipped unless --force is given.
# =============================================================================

set -euo pipefail

# ----- defaults ---------------------------------------------------------------
K_VALUES=(1 5 10 20 50 100 200 500 1000 2000)
MODES=(none remove)
DATASET="cifar"
USER_MODEL="r32p"
TRAINER="sgd"
POISONER="1xs"
BUDGET=1500
SOURCE_LABEL=9
TARGET_LABEL=4
ENCODER="dinov2_vits14"
SCORING="disagreement"
REMOVAL_COUNT="auto"
THRESHOLD=0.5
SEED=0

NAME_PREFIX=""             # auto-derived if empty
SKIP_TRAIN=0
SKIP_PLOTS=0
SKIP_TABLES=0
SKIP_AGGREGATE=0
FORCE=0
PYTHON="${PYTHON:-python}"

# Resume policy: experiments already generated in a previous sweep run that
# must NOT be retrained.  All other pipeline stages still operate over the
# full K_VALUES x MODES grid so the final report includes these existing
# outputs alongside the newly trained remove-mode runs.
ALREADY_DONE_MODES=(none)        # done for every k value
ALREADY_DONE_REMOVE_K=(1 5)      # done only for remove mode

# ----- arg parsing ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --k-values)         shift; IFS=' ' read -ra K_VALUES <<< "$1"; shift ;;
        --modes)            shift; IFS=' ' read -ra MODES    <<< "$1"; shift ;;
        --dataset)          shift; DATASET="$1"; shift ;;
        --user-model)       shift; USER_MODEL="$1"; shift ;;
        --trainer)          shift; TRAINER="$1"; shift ;;
        --poisoner)         shift; POISONER="$1"; shift ;;
        --budget)           shift; BUDGET="$1"; shift ;;
        --source-label)     shift; SOURCE_LABEL="$1"; shift ;;
        --target-label)     shift; TARGET_LABEL="$1"; shift ;;
        --encoder)          shift; ENCODER="$1"; shift ;;
        --scoring)          shift; SCORING="$1"; shift ;;
        --removal-count)    shift; REMOVAL_COUNT="$1"; shift ;;
        --threshold)        shift; THRESHOLD="$1"; shift ;;
        --seed)             shift; SEED="$1"; shift ;;
        --name-prefix)      shift; NAME_PREFIX="$1"; shift ;;
        --python)           shift; PYTHON="$1"; shift ;;
        --skip-train)       SKIP_TRAIN=1; shift ;;
        --skip-plots)       SKIP_PLOTS=1; shift ;;
        --skip-tables)      SKIP_TABLES=1; shift ;;
        --skip-aggregate)   SKIP_AGGREGATE=1; shift ;;
        --skip-none)        MODES=(remove); shift ;;
        --force)            FORCE=1; shift ;;
        -h|--help)
            sed -n '2,40p' "$0"; exit 0 ;;
        *)
            echo "Unknown arg: $1"; exit 2 ;;
    esac
done

if [[ -z "$NAME_PREFIX" ]]; then
    NAME_PREFIX="knn_defense_${DATASET}_${POISONER}_${BUDGET}"
fi

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
cd "$ROOT"

REPORT_DIR="experiments/_report_${NAME_PREFIX}"
mkdir -p "$REPORT_DIR"

LOG_FILE="$REPORT_DIR/sweep.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "Deep k-NN defense sweep — starting"
echo "  date           : $(date)"
echo "  project root   : $ROOT"
echo "  name prefix    : $NAME_PREFIX"
echo "  report dir     : $REPORT_DIR"
echo "  k values       : ${K_VALUES[*]}"
echo "  modes          : ${MODES[*]}"
echo "  dataset        : $DATASET"
echo "  user model     : $USER_MODEL"
echo "  trainer        : $TRAINER"
echo "  poisoner       : $POISONER"
echo "  budget         : $BUDGET"
echo "  encoder        : $ENCODER"
echo "  scoring        : $SCORING"
echo "  threshold      : $THRESHOLD"
echo "  seed           : $SEED"
echo "  skip_train     : $SKIP_TRAIN"
echo "  skip_plots     : $SKIP_PLOTS"
echo "  skip_tables    : $SKIP_TABLES"
echo "  force          : $FORCE"
echo "============================================================"

# ----- 1. generate configs ----------------------------------------------------
echo
echo "[1/6] Generating per-(k, mode) configs ..."
"$PYTHON" scripts/k_sweep/generate_config.py \
    --k-values "${K_VALUES[@]}" \
    --modes "${MODES[@]}" \
    --dataset "$DATASET" \
    --user-model "$USER_MODEL" \
    --trainer "$TRAINER" \
    --poisoner "$POISONER" \
    --budget "$BUDGET" \
    --source-label "$SOURCE_LABEL" \
    --target-label "$TARGET_LABEL" \
    --encoder "$ENCODER" \
    --scoring "$SCORING" \
    --removal-count "$REMOVAL_COUNT" \
    --threshold "$THRESHOLD" \
    --seed "$SEED" \
    --name-prefix "$NAME_PREFIX"

# ----- 2. run experiments -----------------------------------------------------
if [[ $SKIP_TRAIN -eq 1 ]]; then
    echo
    echo "[2/6] --skip-train set; not running run_experiment.py."
else
    echo
    echo "[2/6] Running experiments (one per (k, mode)) ..."
    for k in "${K_VALUES[@]}"; do
        for mode in "${MODES[@]}"; do
            exp_name="${NAME_PREFIX}_${mode}_k${k}"

            # Resume policy: skip experiments that were generated in a prior
            # sweep run.  These skips are unconditional (not overridable by
            # --force) because k has no effect on 'none' and the listed
            # remove runs are intentionally being preserved.
            skip_mode=0
            for m in "${ALREADY_DONE_MODES[@]}"; do
                if [[ "$mode" == "$m" ]]; then skip_mode=1; break; fi
            done
            if [[ $skip_mode -eq 1 ]]; then
                echo "  [skip] $exp_name (mode '$mode' already generated previously; k has no effect)"
                continue
            fi
            if [[ "$mode" == "remove" ]]; then
                skip_k=0
                for kd in "${ALREADY_DONE_REMOVE_K[@]}"; do
                    if [[ "$k" == "$kd" ]]; then skip_k=1; break; fi
                done
                if [[ $skip_k -eq 1 ]]; then
                    echo "  [skip] $exp_name (remove/k=$k already generated previously)"
                    continue
                fi
            fi

            summary="experiments/${exp_name}/summary.json"
            if [[ -f "$summary" && $FORCE -eq 0 ]]; then
                echo "  [skip] $exp_name (summary.json exists; --force to override)"
                continue
            fi
            echo "  [run ] $exp_name"
            "$PYTHON" run_experiment.py "$exp_name"
        done
    done
fi

# ----- 3. aggregate -----------------------------------------------------------
if [[ $SKIP_AGGREGATE -eq 1 ]]; then
    echo "[3/6] --skip-aggregate set."
else
    echo
    echo "[3/6] Aggregating summary_detailed.json files ..."
    "$PYTHON" scripts/k_sweep/aggregate_results.py \
        --name-prefix "$NAME_PREFIX" \
        --k-values "${K_VALUES[@]}" \
        --modes "${MODES[@]}" \
        --report-dir "$REPORT_DIR"
fi

# ----- 4-7. plots -------------------------------------------------------------
if [[ $SKIP_PLOTS -eq 1 ]]; then
    echo "[4-7/6] --skip-plots set."
else
    echo
    echo "[4/6] Detection-metric plots ..."
    "$PYTHON" scripts/k_sweep/plot_detection.py \
        --name-prefix "$NAME_PREFIX" \
        --k-values "${K_VALUES[@]}" \
        --modes "${MODES[@]}" \
        --report-dir "$REPORT_DIR" \
        --threshold "$THRESHOLD"

    echo
    echo "[5/6] Training-dynamics + trade-off plots ..."
    "$PYTHON" scripts/k_sweep/plot_training.py \
        --name-prefix "$NAME_PREFIX" \
        --k-values "${K_VALUES[@]}" \
        --modes "${MODES[@]}" \
        --report-dir "$REPORT_DIR"

    echo
    echo "[6/6] Feature-space + sample-grid plots ..."
    "$PYTHON" scripts/k_sweep/plot_features.py \
        --name-prefix "$NAME_PREFIX" \
        --k-values "${K_VALUES[@]}" \
        --modes "${MODES[@]}" \
        --dataset "$DATASET" \
        --encoder "$ENCODER" \
        --report-dir "$REPORT_DIR"

    "$PYTHON" scripts/k_sweep/plot_samples.py \
        --name-prefix "$NAME_PREFIX" \
        --k-values "${K_VALUES[@]}" \
        --dataset "$DATASET" \
        --poisoner "$POISONER" \
        --target-label "$TARGET_LABEL" \
        --report-dir "$REPORT_DIR"
fi

# ----- tables -----------------------------------------------------------------
if [[ $SKIP_TABLES -eq 1 ]]; then
    echo "[tables] --skip-tables set."
else
    echo
    echo "[tables] Building LaTeX + Markdown tables ..."
    "$PYTHON" scripts/k_sweep/make_tables.py \
        --name-prefix "$NAME_PREFIX" \
        --report-dir "$REPORT_DIR"
fi

echo
echo "============================================================"
echo "Deep k-NN defense sweep — complete"
echo "  report dir : $REPORT_DIR"
echo "  log        : $LOG_FILE"
echo "============================================================"
