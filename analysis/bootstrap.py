"""Patient-cluster and hierarchical bootstraps, and the probability of false outperformance.

Patients are the resampling unit: a draw is a vector of patient multiplicities, and every
visit of a drawn patient counts with the multiplicity of its patient. The metrics below are
then computed on weighted confusion matrices, weighted precision-recall steps and weighted
visit averages. With unit weights they equal :func:`analysis.metrics.compute_metrics`.

* :func:`patient_bootstrap`: uncertainty due to the evaluated patients, for one fixed
  trained model (the intervals of the tables, on training repetition 1).
* :func:`hierarchical_bootstrap`: training variability and patient sampling together, for
  the means over training repetitions of several configurations (effects and PFO).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from analysis.predictions import Predictions

ECE_BINS = 15
METRICS = (
    "accuracy",
    "mcc",
    "pr_auc",
    "macro_f1",
    "ece",
    "nll",
    "brier_sum",
    "brier_skill_score",
    "confidence_deficit",
)


def confusion_metrics(cm: np.ndarray) -> np.ndarray:
    """Accuracy, MCC and Macro-F1 of confusion matrices (..., C, C), rows = true classes.

    MCC is 0 when undefined, and a class whose F1 denominator is zero contributes 0.
    """
    t = cm.sum(-1)
    p = cm.sum(-2)
    n = t.sum(-1)
    diag = np.diagonal(cm, axis1=-2, axis2=-1)
    correct = diag.sum(-1)
    denom = np.sqrt((n * n - (p * p).sum(-1)) * (n * n - (t * t).sum(-1)))
    mcc = np.divide(
        correct * n - (t * p).sum(-1),
        denom,
        out=np.zeros_like(n, dtype=float),
        where=denom != 0,
    )
    f1 = np.divide(
        2 * diag, t + p, out=np.zeros_like(t, dtype=float), where=(t + p) != 0
    ).mean(-1)
    return np.stack([correct / n, mcc, f1], axis=-1)


def weighted_pr_auc(
    y: np.ndarray, p: np.ndarray, visit_weights: np.ndarray
) -> np.ndarray:
    """Macro PR-AUC (stepwise average precision, tied scores grouped) for each row of weights."""
    draws = len(visit_weights)
    scores = np.zeros(draws)
    for k in range(p.shape[1]):
        order = np.argsort(-p[:, k], kind="stable")
        positive = (y[order] == k).astype(float)
        ends = np.r_[np.flatnonzero(np.diff(p[order, k])), len(y) - 1]
        w = visit_weights[:, order]
        tp = np.cumsum(w * positive, axis=1)[:, ends]
        allw = np.cumsum(w, axis=1)[:, ends]
        precision = np.divide(tp, allw, out=np.zeros_like(tp), where=allw > 0)
        recall_steps = np.diff(np.column_stack([np.zeros(draws), tp]), axis=1)
        scores += np.divide(
            (recall_steps * precision).sum(1),
            tp[:, -1],
            out=np.zeros(draws),
            where=tp[:, -1] > 0,
        )
    return scores / p.shape[1]


def weighted_metrics(pred: Predictions, weights: np.ndarray) -> np.ndarray:
    """The metrics of :data:`METRICS` for each row of patient weights (draws x patients)."""
    y, p, patient = pred.y, pred.p, pred.patient
    n_patients, n_classes, n = weights.shape[1], p.shape[1], len(y)
    hard = np.zeros((n_patients, n_classes * n_classes))
    np.add.at(hard, (patient, n_classes * y + p.argmax(1)), 1)
    correct = (p.argmax(1) == y).astype(float)
    confidence = p.max(1)
    bins = np.minimum((confidence * ECE_BINS).astype(int), ECE_BINS - 1)
    sums = np.zeros((n_patients, ECE_BINS + 2 + n_classes))
    np.add.at(sums, (patient, bins), correct - confidence)
    np.add.at(
        sums,
        (patient, ECE_BINS),
        -np.log(np.maximum(p[np.arange(n), y], np.finfo(float).eps)),
    )
    np.add.at(
        sums, (patient, ECE_BINS + 1), (p * p).sum(1) - 2 * p[np.arange(n), y] + 1
    )
    np.add.at(sums, (patient, ECE_BINS + 2 + y), 1)
    visit_weights = weights[:, patient]
    visits = visit_weights.sum(1)
    h = confusion_metrics((weights @ hard).reshape(len(weights), n_classes, n_classes))
    q = weights @ sums
    brier = q[:, ECE_BINS + 1] / visits
    prevalence = q[:, ECE_BINS + 2 :] / visits[:, None]
    return np.column_stack(
        [
            h[:, 0],
            h[:, 1],
            weighted_pr_auc(y, p, visit_weights),
            h[:, 2],
            np.abs(q[:, :ECE_BINS]).sum(1) / visits,
            q[:, ECE_BINS] / visits,
            brier,
            1 - brier / (1 - (prevalence**2).sum(1)),
            q[:, :ECE_BINS].sum(1) / visits,
        ]
    )


def patient_bootstrap(
    pred: Predictions, *, seed: int, draws: int = 10_000
) -> np.ndarray:
    """Patient-cluster bootstrap of one fixed prediction object, draws x :data:`METRICS`.

    The patient multiplicities follow ``multinomial(P, 1/P)`` from
    ``numpy.random.default_rng(seed)``. Two objects with the same patients in the same order
    and the same seed share every draw, so the difference of their draws gives a paired
    interval (five minus three classes). The tables use the seeds 20260925 (ID) and
    20260926 (OOD).
    """
    n_patients = int(pred.patient.max() + 1)
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, draws, 100):
        weights = rng.multinomial(
            n_patients,
            np.full(n_patients, 1 / n_patients),
            size=min(100, draws - start),
        )
        out.append(weighted_metrics(pred, weights))
    return np.concatenate(out)


def percentile_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of bootstrap draws."""
    low, high = np.quantile(values, [(1 - level) / 2, (1 + level) / 2])
    return float(low), float(high)


def hierarchical_bootstrap(
    groups: Sequence[Sequence[Predictions]],
    statistic: Callable[[Predictions, np.ndarray], np.ndarray] = weighted_metrics,
    *,
    seed: int,
    draws: int = 10_000,
    shared_repetitions: bool = False,
) -> np.ndarray:
    """Hierarchical bootstrap of configuration means over training repetitions.

    ``groups`` holds, for each configuration, its training repetitions; every object must
    share the visits, labels and patients of the first one. Draw ``i`` uses
    ``numpy.random.default_rng(seed + i)`` to resample, in this order, the repetitions of each
    configuration (independently, with replacement) and then the patients, shared by all
    configurations so that comparisons stay paired on patients. With
    ``shared_repetitions=True`` one repetition draw is shared by all configurations, for
    strategies applied to the same trained models (inference). ``statistic`` gives the
    metrics of one object under patient weights.

    Returns an array (draws x configurations x metrics): the repetition-weighted means. The
    effect of configuration B over A is ``out[:, b] - out[:, a]``, and its percentile
    interval is the effect interval.
    """
    reference = groups[0][0]
    n_patients = int(reference.patient.max() + 1)
    for group in groups:
        for pred in group:
            if not (
                np.array_equal(pred.y, reference.y)
                and np.array_equal(pred.patient, reference.patient)
            ):
                raise ValueError(
                    "all repetitions must share visits, labels and patients"
                )
    out = []
    for start in range(0, draws, 100):
        repetition_weights, patient_weights = [], []
        for i in range(start, min(start + 100, draws)):
            rng = np.random.default_rng(seed + i)
            if shared_repetitions:
                r = len(groups[0])
                shared = np.bincount(rng.integers(0, r, size=r), minlength=r) / r
                repetition_weights.append([shared] * len(groups))
            else:
                repetition_weights.append(
                    [
                        np.bincount(
                            rng.integers(0, len(g), size=len(g)), minlength=len(g)
                        )
                        / len(g)
                        for g in groups
                    ]
                )
            patient_weights.append(
                np.bincount(
                    rng.integers(0, n_patients, size=n_patients), minlength=n_patients
                )
            )
        patient_weights = np.array(patient_weights)
        means = []
        for g, group in enumerate(groups):
            values = np.stack(
                [statistic(pred, patient_weights) for pred in group], axis=1
            )
            weights = np.array([w[g] for w in repetition_weights])
            means.append((values * weights[:, :, None]).sum(1))
        out.append(np.stack(means, axis=1))
    return np.concatenate(out)


def probability_of_false_outperformance(
    differences: np.ndarray, observed: float
) -> float:
    """Share of draws in which the apparent winner is not ahead.

    ``differences`` are bootstrap draws of mean(B) - mean(A) and ``observed`` the observed
    difference: P*(delta <= 0) if observed > 0, P*(delta >= 0) if observed < 0, NaN for a tie.
    """
    differences = np.asarray(differences, dtype=float)
    if observed > 0:
        return float(np.mean(differences <= 0))
    if observed < 0:
        return float(np.mean(differences >= 0))
    return float("nan")
