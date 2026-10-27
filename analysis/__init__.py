"""Statistics of the extended journal version (MELBA), computed from saved predictions.

Every function works on NumPy arrays, never on images or networks:

* ``y``: true class index of each visit, in the declared class order (CN, AD, BV, PNFA,
  SD for five classes; CN, AD, FTD for three);
* ``p``: predicted probabilities, visits x classes, used as saved;
* ``patient``: patient index (0..P-1) of each visit, the resampling unit.

Modules:

* :mod:`analysis.predictions`: one prediction object per training repetition and domain;
* :mod:`analysis.metrics`: Accuracy, MCC, Macro-F1, PR-AUC, ECE, NLL, Brier sum, Brier skill
  score, confidence deficit, five-to-three collapse;
* :mod:`analysis.bootstrap`: patient-cluster and hierarchical bootstraps, probability of
  false outperformance;
* :mod:`analysis.dispersion`: normalized coefficient of variation (nCV);
* :mod:`analysis.mcnemar`: exact McNemar test and Holm adjustment;
* :mod:`analysis.calibration`: temperature scaling;
* :mod:`analysis.disagreement`: disagreement between training repetitions;
* :mod:`analysis.demographic_baseline`: diagnosis predicted from age, sex and cohort only.
"""
