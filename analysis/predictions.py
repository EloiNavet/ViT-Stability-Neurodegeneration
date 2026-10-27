"""One prediction object per training repetition and domain.

A training repetition is a complete ten-fold cross-validation run. ``eval/eval_transformer.py``
writes, for each fold model, its predictions on its in-domain (ID) test fold and on the
whole out-of-domain (OOD) pool, with one ``pred_<class>`` probability column per class.

* ID: the ten test folds are concatenated, so every ID visit is predicted once, by the
  model that did not see its patient during training.
* OOD: the probabilities of the ten fold models are averaged for each visit, so every OOD
  prediction is a ten-model ensemble.

Rows of one file that share the same ``Subject`` (one acquisition listed twice in the
metadata) are averaged first. Probabilities are used as saved, without any further
softmax, and training repetitions are never pooled.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Predictions:
    """Predictions of one training repetition in one domain.

    Visits are sorted by patient, then by acquisition. ``fold`` is the test fold of each ID
    visit (-1 for OOD visits).
    """

    y: np.ndarray
    p: np.ndarray
    patient: np.ndarray
    fold: np.ndarray
    subjects: np.ndarray
    classes: tuple[str, ...]


def _read_fold(path: Path, classes: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    columns = [f"pred_{name}" for name in classes]
    missing = {"Subject", "Diagnosis", *columns} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    labels = frame.groupby("Subject")["Diagnosis"].nunique()
    if (labels > 1).any():
        raise ValueError(f"{path}: one acquisition with several diagnoses")
    return frame.groupby("Subject", sort=True).agg(
        Diagnosis=("Diagnosis", "first"), **{c: (c, "mean") for c in columns}
    )


def assemble(
    fold_files: Sequence[Path],
    domain: str,
    classes: Sequence[str],
    patient_of: Callable[[str], str],
) -> Predictions:
    """Prediction object of one repetition from its ten fold files.

    ``fold_files[k]`` holds the predictions of the model tested on fold ``k``; ``domain`` is
    ``"id"`` or ``"od"``; ``classes`` is the declared class order; ``patient_of`` maps an
    acquisition (``Subject``) to its patient, from the metadata of its cohort.
    """
    if len(fold_files) != 10:
        raise ValueError("a training repetition has ten fold files")
    columns = [f"pred_{name}" for name in classes]
    frames = [_read_fold(Path(path), classes) for path in fold_files]
    if domain == "id":
        for fold, frame in enumerate(frames):
            frame["fold"] = fold
        table = pd.concat(frames)
        if table.index.duplicated().any():
            raise ValueError("an ID acquisition appears in several test folds")
    elif domain == "od":
        if any(not frame.index.equals(frames[0].index) for frame in frames[1:]):
            raise ValueError(
                "the fold models did not predict the same OOD acquisitions"
            )
        table = frames[0][["Diagnosis"]].copy()
        table[columns] = np.mean(
            [frame[columns].to_numpy(float) for frame in frames], axis=0
        )
        table["fold"] = -1
    else:
        raise ValueError(f"unknown domain: {domain}")
    table["patient_id"] = [patient_of(subject) for subject in table.index]
    if domain == "id" and (table.groupby("patient_id")["fold"].nunique() > 1).any():
        raise ValueError("a patient appears in several ID test folds")
    table = table.reset_index().sort_values(["patient_id", "Subject"], kind="stable")
    p = table[columns].to_numpy(float)
    if not np.allclose(p.sum(axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("probability rows do not sum to one")
    return Predictions(
        y=table["Diagnosis"]
        .map({name: k for k, name in enumerate(classes)})
        .to_numpy(int),
        p=p,
        patient=pd.factorize(table["patient_id"], sort=True)[0],
        fold=table["fold"].to_numpy(int),
        subjects=table["Subject"].to_numpy(str),
        classes=tuple(classes),
    )
