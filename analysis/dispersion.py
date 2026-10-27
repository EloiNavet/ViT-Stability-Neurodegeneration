"""Normalized coefficient of variation (nCV) of a metric across training repetitions.

``nCV = sqrt(N) * s / mean``, where ``s`` and ``mean`` are the sample standard deviation
and the mean of the metric over the repetitions and ``N`` is the number of evaluated visits.
The class-weighted nCV of per-class F1 is ``sqrt(N * N_k) * s_k / mean_k``, with ``N_k`` the
support of class ``k``. The ratio is undefined (NaN) when the mean is zero, for a diagnosis
that is never recognized, which is not evidence of stability.
"""

from __future__ import annotations

import math

import numpy as np


def ncv(
    values: np.ndarray, n_visits: int, n_class: int | None = None, ddof: int = 1
) -> float:
    """nCV of the repetition values of one metric; ``n_class`` gives the class-weighted version."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if mean == 0:
        return math.nan
    return float(np.sqrt(n_visits * (n_class or 1)) * values.std(ddof=ddof) / mean)
