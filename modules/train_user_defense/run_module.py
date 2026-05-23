"""
Trains a downstream (user) model on FLIP-poisoned labels with the KNN
defense applied beforehand.

defense_mode = 'none'   : regression-equivalent to vanilla train_user. No
                          removal; detection scores are still computed and
                          logged for comparison.
defense_mode = 'remove' : drops the samples flagged by KNN detection from
                          the training set before training.

Outputs (under output_dir/):
    paccs.npy, caccs.npy, labels.npy, model.pth  (train_user-compatible)
    scores.npy, kept_indices.npy, removed_indices.npy
    detection_metrics.json
    summary.json                                  (headline: mode, score_key, cta, pta)
    summary_detailed.json                         (full config + metrics)
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from modules.base_utils.datasets import (
    get_matching_datasets, get_n_classes, pick_poisoner, construct_user_dataset,
)
from modules.base_utils.util import (
    extract_toml, get_train_info, mini_train, load_model, needs_big_ims,
    slurmify_path, softmax,
)
from modules.knn_defense.ssl_features import SSLFeatureExtractor
from modules.knn_defense.defense_modes import apply_defense


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_jsonable(x):
    """Recursively convert numpy / Path types to JSON-serialisable Python."""
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def _build_score_key(knn_cfg: dict) -> str:
    """Identifier for the detection config, used in summary.json."""
    key = f"ssl_knn_k{int(knn_cfg['k'])}"
    if knn_cfg.get("scoring", "disagreement") == "weighted_disagreement":
        key += "_weighted"
    return key


def run(experiment_name, module_name, **kwargs):
    slurm_id = kwargs.get("slurm_id", None)
    args = extract_toml(experiment_name, module_name)

    # Required keys (mirror train_user)
    user_model_flag = args["user_model"]
    trainer_flag    = args["trainer"]
    dataset_flag    = args["dataset"]
    poisoner_flag   = args["poisoner"]
    clean_label     = args["source_label"]
    target_label    = args["target_label"]
    mode            = args["defense_mode"]
    knn_cfg         = dict(args["knn"])

    # Optional keys
    soft             = args.get("soft", False)
    alpha            = args.get("alpha", 0.0)
    batch_size       = args.get("batch_size", None)
    epochs           = args.get("epochs", None)
    optim_kwargs     = args.get("optim_kwargs", {})
    scheduler_kwargs = args.get("scheduler_kwargs", {})
    seed             = int(args.get("seed", 0))

    input_path  = slurmify_path(args["input_labels"], slurm_id)
    true_path   = slurmify_path(args["true_labels"], slurm_id)
    output_path = Path(slurmify_path(args["output_dir"], slurm_id))
    output_path.mkdir(parents=True, exist_ok=True)

    _set_seed(seed)

    # === Build datasets — identical to train_user ===
    print("Building datasets...")
    poisoner = pick_poisoner(poisoner_flag, dataset_flag, target_label)
    big_ims = needs_big_ims(user_model_flag)
    _, distillation, test, poison_test, _ = get_matching_datasets(
        dataset_flag, poisoner, clean_label, big=big_ims,
    )

    labels_syn = torch.tensor(np.load(input_path))
    y_true = torch.tensor(np.load(true_path))

    if alpha > 0:
        labels_d = softmax(alpha * y_true + (1 - alpha) * labels_syn)
    else:
        labels_d = softmax(labels_syn)
    if not soft:
        labels_d = labels_d.argmax(dim=1)

    user_dataset = construct_user_dataset(distillation, labels_d)

    # === DEFENSE WRAPPER ===
    print("Running KNN defense...")
    if labels_d.ndim == 2:
        hard_labels = labels_d.argmax(dim=1).numpy().astype(np.int64)
    else:
        hard_labels = labels_d.numpy().astype(np.int64)
    true_hard = y_true.argmax(dim=1).numpy().astype(np.int64)
    is_poisoned_gt = (hard_labels != true_hard).astype(np.int32)

    extractor = SSLFeatureExtractor(knn_cfg["encoder"])
    features = extractor.extract(
        dataset_flag=dataset_flag,
        cache_dir=knn_cfg["feature_cache"],
        batch_size=int(knn_cfg.get("batch_size", 256)),
    )
    if features.shape[0] != len(user_dataset):
        raise RuntimeError(
            f"Feature/label count mismatch: features={features.shape[0]}, "
            f"user_dataset={len(user_dataset)}"
        )

    filtered_dataset, info = apply_defense(
        user_dataset=user_dataset,
        features=features,
        hard_labels=hard_labels,
        is_poisoned_gt=is_poisoned_gt,
        knn_cfg=knn_cfg,
        mode=mode,
    )

    # Persist defense artifacts before training so a crashed trainer
    # still leaves usable detection data on disk.
    np.save(output_path / "scores.npy", info["scores"])
    np.save(output_path / "kept_indices.npy", info["kept_indices"])
    np.save(output_path / "removed_indices.npy", info["removed_indices"])
    with open(output_path / "detection_metrics.json", "w") as f:
        json.dump(_to_jsonable(info["detection_metrics"]), f, indent=2)

    n_train = len(filtered_dataset)
    n_removed = int(len(info["removed_indices"]))
    n_true_poisoned = int(is_poisoned_gt.sum())
    print(f"  mode={mode}  n_total={features.shape[0]}  "
          f"n_true_poisoned={n_true_poisoned}  "
          f"n_removed={n_removed}  n_train={n_train}")

    # === TRAINING — identical to train_user ===
    print("Training user model...")
    n_classes = get_n_classes(dataset_flag)
    model_retrain = load_model(user_model_flag, n_classes)
    batch_size_resolved, epochs_resolved, optimizer_retrain, scheduler = \
        get_train_info(
            model_retrain.parameters(), trainer_flag,
            batch_size, epochs, optim_kwargs, scheduler_kwargs,
        )
    model_retrain, clean_metrics, poison_metrics = mini_train(
        model=model_retrain,
        train_data=filtered_dataset,
        test_data=[test, poison_test.poison_dataset],
        batch_size=batch_size_resolved,
        opt=optimizer_retrain,
        scheduler=scheduler,
        epochs=epochs_resolved,
        record=True,
    )

    # === SAVE — train_user-compatible artifacts ===
    print("Saving results...")
    np.save(output_path / "paccs.npy", np.array(poison_metrics))
    np.save(output_path / "caccs.npy", np.array(clean_metrics))
    np.save(
        output_path / "labels.npy",
        labels_d.numpy() if isinstance(labels_d, torch.Tensor) else labels_d,
    )
    torch.save(model_retrain.state_dict(), output_path / "model.pth")

    # === Headline summary (the format requested) ===
    final_cta = float(clean_metrics[-1][0])
    final_pta = float(poison_metrics[-1][0])
    score_key = _build_score_key(knn_cfg)

    summary = {
        "mode": mode,
        "score_key": score_key,
        "cta": final_cta,
        "pta": final_pta,
    }

    detailed = {
        **summary,
        "dataset": dataset_flag,
        "user_model": user_model_flag,
        "poisoner": poisoner_flag,
        "source_label": clean_label,
        "target_label": target_label,
        "seed": seed,
        "encoder": knn_cfg["encoder"],
        "k": int(knn_cfg["k"]),
        "scoring": knn_cfg.get("scoring", "disagreement"),
        "removal_count": knn_cfg.get("removal_count") if mode == "remove" else None,
        "threshold": float(knn_cfg.get("threshold", 0.5)) if mode == "remove" else None,
        "n_total": int(features.shape[0]),
        "n_true_poisoned": n_true_poisoned,
        "n_removed": n_removed,
        "n_train_samples": n_train,
        "detection_metrics": info["detection_metrics"],
        "policy_resolved": info["policy_resolved"],
    }

    with open(output_path / "summary.json", "w") as f:
        json.dump(_to_jsonable(summary), f, indent=2)
    with open(output_path / "summary_detailed.json", "w") as f:
        json.dump(_to_jsonable(detailed), f, indent=2)

    print("\n" + "=" * 60)
    print(json.dumps(_to_jsonable(summary), indent=2))
    print("=" * 60)


if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
