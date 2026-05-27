"""Evaluation metrics for CMAI behaviour detection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score

from src.utils.cmai_labels import CMAI_BEHAVIOURS, CMAI_CATEGORIES


def compute_cmai_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
    }

    for index, name in CMAI_BEHAVIOURS.items():
        if y_true[:, index].sum() > 0:
            metrics[f"f1_{name}"] = f1_score(y_true[:, index], y_pred[:, index], zero_division=0)
            metrics[f"auc_{name}"] = roc_auc_score(y_true[:, index], y_score[:, index])

    for category, indices in CMAI_CATEGORIES.items():
        metrics[f"f1_category_{category}"] = f1_score(
            y_true[:, indices],
            y_pred[:, indices],
            average="macro",
            zero_division=0,
        )

    return metrics


def compute_early_warning_metrics(y_true_traj: np.ndarray, y_pred_traj: np.ndarray) -> dict[str, float]:
    report = classification_report(
        y_true_traj,
        y_pred_traj,
        target_names=["stable", "rising", "peaking"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "traj_acc": float((y_true_traj == y_pred_traj).mean()),
        "rising_f1": report["rising"]["f1-score"],
        "peaking_f1": report["peaking"]["f1-score"],
        "rising_recall": report["rising"]["recall"],
    }


def find_optimal_thresholds(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    thresholds = np.zeros(y_true.shape[1], dtype=np.float32)
    candidates = np.arange(0.1, 0.9, 0.05)

    for index in range(y_true.shape[1]):
        if y_true[:, index].sum() == 0:
            thresholds[index] = 0.5
            continue
        best_f1 = 0.0
        best_threshold = 0.5
        for threshold in candidates:
            preds = (y_score[:, index] >= threshold).astype(int)
            score = f1_score(y_true[:, index], preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = threshold
        thresholds[index] = best_threshold

    return thresholds

