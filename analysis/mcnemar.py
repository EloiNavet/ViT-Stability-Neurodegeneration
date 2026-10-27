"""Exact McNemar test of two fixed models on the same visits, and Holm adjustment."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def mcnemar_exact(
    correct_a: np.ndarray, correct_b: np.ndarray
) -> tuple[int, int, float]:
    """Visits correct for A only, for B only, and the two-sided exact p-value.

    The p-value is the binomial test of ``min(a_only, b_only)`` among the discordant visits
    with success probability 1/2, and 1 without discordant visits.
    """
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    if a_only + b_only == 0:
        return a_only, b_only, 1.0
    return (
        a_only,
        b_only,
        float(binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue),
    )


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, in the order of the input."""
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(np.argsort(pvalues, kind="stable")):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted
