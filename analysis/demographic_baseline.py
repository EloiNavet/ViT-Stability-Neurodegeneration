"""Demographic and cohort baseline: the five-class diagnosis predicted from covariates only.

A multinomial logistic regression is trained on the same ten folds and with the same 7/2/1
rotation as the networks (``utils.helper.get_train_val_test``): model ``k`` is trained on
folds ``k`` to ``k + 6``, its inverse regularization ``C`` is selected by log-loss on folds
``k + 7`` and ``k + 8``, and it is tested on fold ``k + 9`` (indices modulo 10). Two feature
sets are used: age and sex, then age, sex and cohort type (studies of frontotemporal
dementia, ALLFTD and NIFD, against studies of Alzheimer's disease, ADNI and NACC). The fold
predictions are then assembled and evaluated like those of the networks.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

C_GRID = np.logspace(-3, 3, 13)
FTD_STUDIES = {"FTLD", "NIFD"}  # Dataset labels of ALLFTD and NIFD
FEATURE_SETS = {
    "age_sex": ["age", "female"],
    "age_sex_cohort": ["age", "female", "ftd_study"],
}


def features(frame: pd.DataFrame) -> pd.DataFrame:
    """Age, female sex and FTD-study indicator of each acquisition."""
    return pd.DataFrame(
        {
            "age": frame["Age"].astype(float),
            "female": (frame["Sex"] == "female").astype(float),
            "ftd_study": frame["Dataset"].isin(FTD_STUDIES).astype(float),
        }
    )


def fit_fold(
    folds: Sequence[pd.DataFrame], ood: pd.DataFrame, k: int, columns: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ID test-fold and OOD predictions of model ``k``.

    ``folds`` are the ten fold tables and ``ood`` the external pool, with the columns
    ``Subject``, ``Dataset``, ``Diagnosis``, ``Age`` and ``Sex``. The returned tables add one
    ``pred_<class>`` column per diagnosis, as ``eval/eval_transformer.py`` does.
    """
    train = pd.concat([folds[(k + i) % 10] for i in range(7)], ignore_index=True)
    validation = pd.concat([folds[(k + i) % 10] for i in (7, 8)], ignore_index=True)
    test = folds[(k + 9) % 10]
    scaler = StandardScaler().fit(features(train)[list(columns)])

    def x(frame: pd.DataFrame) -> np.ndarray:
        return scaler.transform(features(frame)[list(columns)])

    best = None
    for c in C_GRID:
        model = LogisticRegression(C=c, max_iter=10000).fit(
            x(train), train["Diagnosis"]
        )
        loss = log_loss(
            validation["Diagnosis"],
            model.predict_proba(x(validation)),
            labels=model.classes_,
        )
        if best is None or loss < best[0]:
            best = (loss, model)
    model = best[1]
    out = []
    for frame in (test, ood):
        frame = frame.copy()
        for k_class, name in enumerate(model.classes_):
            frame[f"pred_{name}"] = model.predict_proba(x(frame))[:, k_class]
        out.append(frame)
    return out[0], out[1]
