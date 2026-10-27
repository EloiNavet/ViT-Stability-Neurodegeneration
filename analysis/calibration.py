"""Temperature scaling of saved probabilities.

The logits are recovered from the probabilities (clipped at 1e-12 and renormalized), and one
positive temperature ``T`` is fitted by minimizing the negative log-likelihood over ``log T``
in [-6, 6]. Dividing the logits by ``T`` never changes the predicted class. Temperature
scaling is always applied last, after every averaging of probabilities.

* In domain, the temperature is cross-fitted: the predictions of each held-out test fold
  are scaled with a temperature fitted on the predictions of the nine other folds.
* Out of domain, the temperature fitted on all in-domain predictions is applied unchanged,
  without using any external label.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax


def to_logits(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Logits of saved probabilities, up to a constant per row."""
    clipped = np.clip(np.asarray(p, dtype=float), eps, 1.0)
    return np.log(clipped / clipped.sum(axis=1, keepdims=True))


def scale(p: np.ndarray, temperature: float) -> np.ndarray:
    """Probabilities after temperature scaling."""
    return softmax(to_logits(p) / temperature, axis=1)


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    """Temperature that minimizes the negative log-likelihood of ``y``."""
    logits = to_logits(p)

    def nll(log_temperature: float) -> float:
        scaled = logits / np.exp(log_temperature)
        log_p = scaled - logsumexp(scaled, axis=1, keepdims=True)
        return float(-np.mean(log_p[np.arange(len(y)), y]))

    result = minimize_scalar(nll, bounds=(-6.0, 6.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"temperature fit failed: {result.message}")
    return float(np.exp(result.x))


def crossfit_scale(p: np.ndarray, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """In-domain predictions scaled fold by fold with temperatures fitted on the other folds."""
    out = np.empty_like(p, dtype=float)
    for k in np.unique(fold):
        held_out = fold == k
        out[held_out] = scale(p[held_out], fit_temperature(p[~held_out], y[~held_out]))
    return out


def transfer_scale(p_id: np.ndarray, y_id: np.ndarray, p_ood: np.ndarray) -> np.ndarray:
    """Out-of-domain predictions scaled with the temperature fitted on all in-domain predictions."""
    return scale(p_ood, fit_temperature(p_id, y_id))
