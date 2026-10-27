"""Disagreement between the training repetitions of one configuration, visit by visit.

``p`` has shape (repetitions, visits, classes): the predictions of the repetitions on the same
visits. The modal label is the class predicted by most repetitions (ties broken by the
declared class order). Entropies use the natural logarithm.
"""

from __future__ import annotations

import numpy as np


def entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy along the last axis."""
    values = np.clip(p, 1e-12, 1.0)
    return -np.sum(values * np.log(values), axis=-1)


def disagreement(p: np.ndarray) -> dict[str, np.ndarray]:
    """Per-visit modal label, modal agreement, variation ratio, vote entropy and
    Jensen-Shannon divergence between the predicted probability vectors."""
    repetitions, _, n_classes = p.shape
    votes = p.argmax(axis=2)
    counts = np.stack([np.bincount(v, minlength=n_classes) for v in votes.T])
    agreement = counts.max(axis=1) / repetitions
    return {
        "modal_label": counts.argmax(axis=1),
        "modal_agreement": agreement,
        "variation_ratio": 1 - agreement,
        "vote_entropy": entropy(counts / repetitions),
        "jensen_shannon": entropy(p.mean(axis=0)) - entropy(p).mean(axis=0),
    }
