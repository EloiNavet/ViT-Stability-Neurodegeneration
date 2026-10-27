"""Metrics of one prediction object.

Every declared class counts, even when the evaluated sample has none of its visits: it then
contributes 0 to Macro-F1 and to PR-AUC, and both are averaged over all declared classes.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

ECE_BINS = 15


def expected_calibration_error(
    y: np.ndarray, p: np.ndarray, bins: int = ECE_BINS
) -> float:
    """Top-label ECE with equal-width confidence bins, closed on the left (the last includes 1)."""
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    index = np.minimum((confidence * bins).astype(int), bins - 1)
    return float(
        sum(
            np.mean(index == b)
            * abs(correct[index == b].mean() - confidence[index == b].mean())
            for b in range(bins)
            if np.any(index == b)
        )
    )


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Macro PR-AUC: unweighted mean over classes of the one-vs-rest average precision."""
    values = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*No positive class found.*")
        for k in range(p.shape[1]):
            values.append(average_precision_score(y == k, p[:, k]))
    return float(np.mean(values))


def brier_sum(y: np.ndarray, p: np.ndarray) -> float:
    """Multiclass Brier score: squared error summed over classes, averaged over visits."""
    one_hot = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def brier_skill_score(y: np.ndarray, p: np.ndarray) -> float:
    """1 - Brier / Brier_ref, with Brier_ref = 1 - sum_k pi_k^2 for the class prevalences pi_k
    of the evaluated sample. It compares three- and five-class models."""
    prevalence = np.bincount(y, minlength=p.shape[1]) / len(y)
    return 1 - brier_sum(y, p) / (1 - np.sum(prevalence**2))


def confidence_deficit(y: np.ndarray, p: np.ndarray) -> float:
    """Accuracy minus mean top-label confidence; positive values mean under-confidence."""
    return float(np.mean(p.argmax(axis=1) == y) - np.mean(p.max(axis=1)))


def compute_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Discrimination and probability quality of one prediction object."""
    classes = np.arange(p.shape[1])
    predicted = p.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "mcc": float(matthews_corrcoef(y, predicted)),
        "macro_f1": float(
            f1_score(y, predicted, labels=classes, average="macro", zero_division=0)
        ),
        "pr_auc": pr_auc(y, p),
        "ece": expected_calibration_error(y, p),
        "nll": float(log_loss(y, p / p.sum(axis=1, keepdims=True), labels=classes)),
        "brier_sum": brier_sum(y, p),
        "brier_skill_score": brier_skill_score(y, p),
        "confidence_deficit": confidence_deficit(y, p),
    }


def per_class_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    """Precision, recall, F1 and support of each declared class."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y, p.argmax(axis=1), labels=np.arange(p.shape[1]), zero_division=0
    )
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}


def collapse_to_three_classes(
    y: np.ndarray, p: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Five-class predictions (CN, AD, BV, PNFA, SD) collapsed to CN, AD and FTD.

    P(FTD) = P(BV) + P(PNFA) + P(SD), and the three FTD subtypes become the label FTD.
    """
    return np.minimum(y, 2), np.column_stack([p[:, 0], p[:, 1], p[:, 2:].sum(axis=1)])
