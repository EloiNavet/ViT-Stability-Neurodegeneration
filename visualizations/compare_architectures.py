"""Compare model architectures: statistical tests, metrics, and visualizations."""

import argparse
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2, shapiro, wilcoxon
from tabulate import tabulate
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.bootstrap_metric import compute_bootstrap_metrics
from utils.calibrate_predictions import extract_fold_from_filename, load_predictions

LOWER_IS_BETTER_METRICS = {"ece", "mce", "brier_score", "brier"}

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")

PUBLICATION_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "DejaVu Serif",
        "Bitstream Vera Serif",
        "Computer Modern Roman",
    ],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#E5E5E5",
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#CCCCCC",
    "legend.fancybox": False,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "patch.linewidth": 0.8,
    "patch.edgecolor": "#333333",
}

mpl.rcParams.update(PUBLICATION_RC)

COLOR_PALETTE = {
    "primary": "#0072B2",
    "secondary": "#E69F00",
    "tertiary": "#009E73",
    "quaternary": "#CC79A7",
    "quinary": "#F0E442",
    "senary": "#56B4E9",
    "septenary": "#D55E00",
    "octonary": "#999999",
}

CATEGORICAL_COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#999999",
]

mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=CATEGORICAL_COLORS)

# Reduced figure sizes for larger apparent text
FIG_WIDTH_SINGLE = 2.5
FIG_WIDTH_1_5 = 4.0
FIG_WIDTH_DOUBLE = 5.0
FIG_GOLDEN_RATIO = 1.618


def shorten_model_name(name: str, max_length: int = 20) -> str:
    """
    Create a shorter, readable label from a long model name.

    Handles common patterns in model naming conventions:
    - Removes common prefixes (swin-, medvit-, etc.)
    - Abbreviates common terms (balanced_sampling -> bal, label_smoothing -> ls, etc.)
    - Joins with '+' for ablation studies

    Args:
        name: Original model name
        max_length: Maximum length for the shortened name

    Returns:
        Shortened, readable model name
    """
    # Common prefixes to remove
    prefixes_to_remove = [
        "swin-5c-",
        "swin-3c-",
        "swin-2c-",
        "medvit-5c-",
        "medvit-3c-",
        "medvit-2c-",
        "vit-5c-",
        "vit-3c-",
        "vit-2c-",
        "resnet-5c-",
        "resnet-3c-",
        "resnet-2c-",
        "no_seed-",
        "no-seed-",
    ]

    # Abbreviations for common terms
    abbreviations = {
        "balanced_sampling": "bal",
        "balanced-sampling": "bal",
        "label_smoothing": "ls",
        "label-smoothing": "ls",
        "dataaug": "aug",
        "data_aug": "aug",
        "data-aug": "aug",
        "mixup": "mix",
        "baseline": "base",
        "no-dataaug": "no-aug",
        "no_dataaug": "no-aug",
        "less-reg": "low-reg",
        "less_reg": "low-reg",
    }

    # Suffixes to remove (version numbers, etc.)
    import re

    result = name

    # Remove common prefixes
    for prefix in prefixes_to_remove:
        if result.lower().startswith(prefix.lower()):
            result = result[len(prefix) :]
            break

    # Remove trailing version numbers like -1, -2, _1, _2
    result = re.sub(r"[-_]\d+$", "", result)

    # Apply abbreviations
    for long_form, short_form in abbreviations.items():
        result = result.replace(long_form, short_form)

    # Clean up separators: convert to '+' for readability
    # Replace multiple separators with single '+'
    result = re.sub(r"[-_]+", "+", result)

    # Remove leading/trailing '+'
    result = result.strip("+")

    # If still too long, truncate intelligently
    if len(result) > max_length:
        # Split by '+' and keep as many parts as fit
        parts = result.split("+")
        truncated = []
        current_len = 0
        for part in parts:
            if current_len + len(part) + 1 <= max_length - 3:  # -3 for '...'
                truncated.append(part)
                current_len += len(part) + 1
            else:
                break
        if len(truncated) < len(parts):
            result = (
                "+".join(truncated) + "..."
                if truncated
                else result[: max_length - 3] + "..."
            )
        else:
            result = "+".join(truncated)

    # Capitalize first letter for cleaner look
    if result:
        result = result[0].upper() + result[1:] if len(result) > 1 else result.upper()

    return result if result else name[:max_length]


def create_label_mapping(names: List[str], max_length: int = 20) -> Dict[str, str]:
    """
    Create a mapping from original names to shortened labels.
    Ensures uniqueness of shortened labels.

    Args:
        names: List of original model names
        max_length: Maximum length for shortened names

    Returns:
        Dictionary mapping original names to shortened labels
    """
    mapping = {}
    short_names = {}

    for name in names:
        short = shorten_model_name(name, max_length)

        # Handle duplicates by adding a suffix
        if short in short_names:
            count = short_names[short]
            short_names[short] = count + 1
            short = f"{short}({count + 1})"
        else:
            short_names[short] = 1

        mapping[name] = short

    return mapping


# Mapping from CLI metric names to internal metric names used by compute_bootstrap_metrics
# Only the canonical names are used (no aliases)
METRIC_MAP = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "roc_auc": "roc_auc",
    "pr_auc": "pr_auc",
    "mcc": "mcc",
    "macro_f1": "macro_f1",
    "ece": "ece",
    "mce": "mce",
    "brier_score": "brier_score",
}

# Valid metric choices for CLI argument validation (sorted for consistent help display)
METRIC_CHOICES = sorted(METRIC_MAP.keys())


def mcnemar_test(
    y_true: np.ndarray, y_pred_A: np.ndarray, y_pred_B: np.ndarray
) -> Dict:
    """
    Perform McNemar's test to compare two classifiers.

    McNemar's test is appropriate when:
    - Comparing two models on the SAME test set
    - Binary or multi-class classification
    - Testing if one model is significantly better than another

    IMPORTANT: For OOD evaluation with K-fold ensemble predictions:
    - Each subject must have exactly ONE prediction (average probabilities across folds, then argmax)
    - Both models must predict on identical subjects
    - Ground truth must be identical for both models

    Contingency table:
                    B correct | B incorrect
    A correct    |     a     |      b
    A incorrect  |     c     |      d

    Test statistic: χ² = (|b - c| - 1)² / (b + c)
    with continuity correction (recommended for small sample sizes)

    For small sample sizes (b+c < 25), exact binomial test is used instead.

    Args:
        y_true: Ground truth labels (N,)
        y_pred_A: Predictions from model A (N,)
        y_pred_B: Predictions from model B (N,)

    Returns:
        Dictionary with test results and interpretation
    """
    correct_A = y_pred_A == y_true
    correct_B = y_pred_B == y_true

    # Build contingency table
    a = int(np.sum(correct_A & correct_B))  # both correct
    b = int(np.sum(correct_A & ~correct_B))  # A correct, B wrong
    c = int(np.sum(~correct_A & correct_B))  # A wrong, B correct
    d = int(np.sum(~correct_A & ~correct_B))  # both wrong

    # McNemar's test with continuity correction
    if b + c == 0:
        chi2_stat = 0.0
        p_value = 1.0
        test_used = "mcnemar_degenerate"
    elif b + c < 25:
        # Use exact binomial test for small sample sizes
        from scipy.stats import binomtest

        warnings.warn(
            f"Low discordant pairs (b+c={b + c}). Using exact binomial test instead of χ² approximation."
        )
        chi2_stat = None
        p_value = binomtest(b, b + c, p=0.5, alternative="two-sided").pvalue
        test_used = "exact_binomial"
    else:
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2.cdf(chi2_stat, df=1)
        test_used = "mcnemar_chi2"

    # Determine winner (alpha = 0.05)
    if p_value < 0.05:
        winner = "A" if b > c else "B"
        interpretation = f"Model {winner} is significantly better (p={p_value:.4f})"
    else:
        winner = "none"
        interpretation = f"No significant difference (p={p_value:.4f})"

    return {
        "chi2_statistic": float(chi2_stat) if chi2_stat is not None else None,
        "p_value": float(p_value),
        "winner": winner,
        "interpretation": interpretation,
        "test_used": test_used,
        "contingency": {
            "both_correct": a,
            "A_correct_B_wrong": b,
            "A_wrong_B_correct": c,
            "both_wrong": d,
        },
        "n_samples": a + b + c + d,
    }


def wilcoxon_test(
    scores_A: np.ndarray, scores_B: np.ndarray, alternative: str = "two-sided"
) -> Dict:
    """
    Perform Wilcoxon signed-rank test to compare paired distributions.

    Wilcoxon test is appropriate when:
    - Comparing performance across K folds (paired data)
    - Non-parametric alternative to paired t-test
    - No assumption of normal distribution
    - Robust to outliers

    Args:
        scores_A: Performance scores for model A across folds (K,)
        scores_B: Performance scores for model B across folds (K,)
        alternative: "two-sided", "less", or "greater"

    Returns:
        Dictionary with test results and interpretation
    """
    if len(scores_A) != len(scores_B):
        raise ValueError("Score arrays must have same length (paired samples)")

    if len(scores_A) < 6:
        warnings.warn(
            f"Wilcoxon test with n={len(scores_A)} samples may have low power. "
            "Consider using at least 6-10 folds for robust results."
        )

    # Perform test
    statistic, p_value = wilcoxon(scores_A, scores_B, alternative=alternative)

    # Compute effect size (median difference)
    differences = scores_A - scores_B
    stat, p = shapiro(differences)
    if p > 0.05:
        print("Normal distribution detected - t-test might be more powerful")
    median_diff = float(np.median(differences))
    mean_diff = float(np.mean(differences))

    # Determine winner (alpha = 0.05)
    if p_value < 0.05:
        if median_diff > 0:
            winner = "A"
            interpretation = f"Model A significantly better (p={p_value:.4f}, median_diff={median_diff:.4f})"
        else:
            winner = "B"
            interpretation = f"Model B significantly better (p={p_value:.4f}, median_diff={median_diff:.4f})"
    else:
        winner = "none"
        interpretation = f"No significant difference (p={p_value:.4f})"

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "winner": winner,
        "interpretation": interpretation,
        "median_difference": median_diff,
        "mean_difference": mean_diff,
        "scores_A_mean": float(np.mean(scores_A)),
        "scores_B_mean": float(np.mean(scores_B)),
        "scores_A_std": float(np.std(scores_A)),
        "scores_B_std": float(np.std(scores_B)),
        "n_folds": len(scores_A),
    }


def compute_pfo(
    samples_A: np.ndarray,
    samples_B: np.ndarray,
    lower_is_better: bool = False,
) -> Dict[str, float]:
    """
    Compute Probability of False Outperformance (PFO).

    Following Christodoulou et al. (2025) "False Promises", this function computes
    the probability that the observed performance ranking between two models
    could reverse under resampling.

    For models A and B with bootstrap samples, computes P̂(Δ ≤ 0) where
    Δ = M_A - M_B (or M_B - M_A if lower_is_better=True).

    Parameters
    ----------
    samples_A : np.ndarray
        Bootstrap samples for model A (B,)
    samples_B : np.ndarray
        Bootstrap samples for model B (B,)
    lower_is_better : bool
        If True, model A "outperforms" B when A < B (e.g., for ECE, Brier)

    Returns
    -------
    Dict with keys:
        - delta_observed: Observed difference (mean_A - mean_B)
        - pfo_A_over_B: P(Δ ≤ 0) - probability A doesn't truly outperform B
        - pfo_B_over_A: P(Δ ≥ 0) - probability B doesn't truly outperform A
        - mean_A, mean_B: Mean bootstrap values
        - std_A, std_B: Standard deviations
        - ci95_A, ci95_B: 95% confidence intervals
    """
    if len(samples_A) != len(samples_B):
        raise ValueError(
            f"Bootstrap sample sizes must match: {len(samples_A)} vs {len(samples_B)}"
        )

    # Compute bootstrap differences
    if lower_is_better:
        # For metrics like ECE: A outperforms B if A < B, so delta = B - A
        delta_samples = samples_B - samples_A
        delta_observed = np.mean(samples_B) - np.mean(samples_A)
    else:
        # For metrics like accuracy: A outperforms B if A > B, so delta = A - B
        delta_samples = samples_A - samples_B
        delta_observed = np.mean(samples_A) - np.mean(samples_B)

    # PFO: fraction of bootstrap samples where the ranking reverses
    # P(Δ ≤ 0) means A doesn't truly outperform B
    pfo_A_over_B = float(np.mean(delta_samples <= 0))
    pfo_B_over_A = float(np.mean(delta_samples >= 0))

    # Also compute paired bootstrap CI for the difference
    delta_ci = (
        float(np.percentile(delta_samples, 2.5)),
        float(np.percentile(delta_samples, 97.5)),
    )

    return {
        "delta_observed": float(delta_observed),
        "delta_mean": float(np.mean(delta_samples)),
        "delta_std": float(np.std(delta_samples)),
        "delta_ci95": delta_ci,
        "pfo_A_over_B": pfo_A_over_B,
        "pfo_B_over_A": pfo_B_over_A,
        "mean_A": float(np.mean(samples_A)),
        "mean_B": float(np.mean(samples_B)),
        "std_A": float(np.std(samples_A)),
        "std_B": float(np.std(samples_B)),
        "ci95_A": (
            float(np.percentile(samples_A, 2.5)),
            float(np.percentile(samples_A, 97.5)),
        ),
        "ci95_B": (
            float(np.percentile(samples_B, 2.5)),
            float(np.percentile(samples_B, 97.5)),
        ),
        "n_bootstrap": len(samples_A),
    }


def compute_metrics(
    y_true: np.ndarray, y_probs: np.ndarray, classes: List[str]
) -> Dict:
    """
    Compute standard classification metrics with bootstrap confidence intervals.

    Args:
        y_true: Ground truth labels (N,) as strings or integers
        y_probs: Prediction probabilities (N, n_classes)
        classes: List of class names in order

    Returns:
        Dictionary with metrics containing 'mean', 'lower', 'upper' for each metric
    """
    # Convert string labels to integer indices if necessary
    if y_true.dtype.kind in ["U", "S", "O"]:  # Unicode, byte string, or object
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        y_true_int = np.array([class_to_idx[label] for label in y_true])
    else:
        y_true_int = y_true

    # Compute bootstrap metrics
    bootstrap_results = compute_bootstrap_metrics(
        y_true=y_true_int,
        y_pred_probs=y_probs,
        n_bootstrap=10000,
        confidence=0.95,
        random_state=42,
        n_jobs=-1,
    )

    return {
        "accuracy": bootstrap_results["accuracy"]["mean"],
        "balanced_accuracy": bootstrap_results["balanced_accuracy"]["mean"],
        "roc_auc": bootstrap_results["roc_auc"]["mean"],
        "roc_auc_macro": bootstrap_results["roc_auc"][
            "mean"
        ],  # Alias for compatibility
        "pr_auc": bootstrap_results["pr_auc"]["mean"],
        "mcc": bootstrap_results["mcc"]["mean"],
        "macro_f1": bootstrap_results["macro_f1"]["mean"],
        "ece": bootstrap_results["ece"]["mean"],
        "mce": bootstrap_results["mce"]["mean"],
        "brier_score": bootstrap_results["brier_score"]["mean"],
        "accuracy_ci": (
            bootstrap_results["accuracy"]["lower"],
            bootstrap_results["accuracy"]["upper"],
        ),
        "balanced_accuracy_ci": (
            bootstrap_results["balanced_accuracy"]["lower"],
            bootstrap_results["balanced_accuracy"]["upper"],
        ),
        "roc_auc_ci": (
            bootstrap_results["roc_auc"]["lower"],
            bootstrap_results["roc_auc"]["upper"],
        ),
        "roc_auc_macro_ci": (
            bootstrap_results["roc_auc"]["lower"],
            bootstrap_results["roc_auc"]["upper"],
        ),
        "pr_auc_ci": (
            bootstrap_results["pr_auc"]["lower"],
            bootstrap_results["pr_auc"]["upper"],
        ),
        "mcc_ci": (
            bootstrap_results["mcc"]["lower"],
            bootstrap_results["mcc"]["upper"],
        ),
        "macro_f1_ci": (
            bootstrap_results["macro_f1"]["lower"],
            bootstrap_results["macro_f1"]["upper"],
        ),
        "ece_ci": (
            bootstrap_results["ece"]["lower"],
            bootstrap_results["ece"]["upper"],
        ),
        "mce_ci": (
            bootstrap_results["mce"]["lower"],
            bootstrap_results["mce"]["upper"],
        ),
        "brier_score_ci": (
            bootstrap_results["brier_score"]["lower"],
            bootstrap_results["brier_score"]["upper"],
        ),
    }


def load_predictions_from_dir(
    model_dir: Path, pattern: str = "*_best0_od.csv"
) -> Dict[int, Path]:
    """Load prediction files grouped by fold."""
    files = sorted(model_dir.glob(pattern))
    fold_files = {}

    for f in files:
        fold = extract_fold_from_filename(f)
        if fold >= 0:
            fold_files[fold] = f

    return fold_files


def perform_wilcoxon_tests(
    scores_dict: Dict[str, np.ndarray],
    alpha: float = 0.05,
    correction: str = "bonferroni",
) -> pd.DataFrame:
    """
    Perform pairwise Wilcoxon signed-rank tests with multiple comparison correction.

    Args:
        scores_dict: Dictionary mapping model names to score arrays
        alpha: Significance level (default: 0.05)
        correction: Multiple comparison correction method:
            - "bonferroni": Conservative, controls FWER
            - "fdr_bh": Benjamini-Hochberg FDR control (less conservative)
            - "none": No correction (not recommended)
    """
    results = []
    models = list(scores_dict.keys())
    n_comparisons = len(list(combinations(models, 2)))

    # First pass: collect all p-values
    comparison_data = []
    for model_a, model_b in combinations(models, 2):
        scores_a = scores_dict[model_a]
        scores_b = scores_dict[model_b]

        test_result = wilcoxon_test(scores_a, scores_b, alternative="two-sided")
        comparison_data.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "test_result": test_result,
            }
        )

    # Extract p-values for correction
    p_values = np.array([comp["test_result"]["p_value"] for comp in comparison_data])

    # Apply correction
    if correction == "bonferroni":
        corrected_alpha = alpha / n_comparisons
        is_significant = p_values < corrected_alpha
        correction_desc = f"Bonferroni (α={corrected_alpha:.4f})"
    elif correction == "fdr_bh":
        from scipy.stats import false_discovery_control

        is_significant = false_discovery_control(p_values, method="bh")
        corrected_alpha = alpha
        correction_desc = f"FDR (Benjamini-Hochberg, α={alpha})"
    elif correction == "none":
        is_significant = p_values < alpha
        corrected_alpha = alpha
        correction_desc = "None"
    else:
        raise ValueError(f"Unknown correction method: {correction}")

    # Second pass: build results with corrected significance
    for i, comp in enumerate(comparison_data):
        model_a = comp["model_a"]
        model_b = comp["model_b"]
        test_result = comp["test_result"]
        p_value = test_result["p_value"]

        # Only declare winner if significant after correction
        winner = test_result["winner"] if is_significant[i] else "none"

        results.append(
            {
                "Model_A": model_a,
                "Model_B": model_b,
                "Mean_A": test_result["scores_A_mean"],
                "Mean_B": test_result["scores_B_mean"],
                "Difference": test_result["mean_difference"],
                "p_value": p_value,
                "alpha_corrected": corrected_alpha,
                "Significant": "Yes" if is_significant[i] else "No",
                "Winner": winner if winner != "none" else "No significant difference",
                "Correction": correction_desc,
            }
        )

    return pd.DataFrame(results)


def perform_mcnemar_tests(
    predictions_dict: Dict[str, Dict],
    alpha: float = 0.05,
    correction: str = "bonferroni",
) -> pd.DataFrame:
    """
    Perform pairwise McNemar tests with multiple comparison correction.

    Args:
        predictions_dict: Dictionary mapping model names to dict with keys:
            - 'y_true': Ground truth labels (N,)
            - 'y_pred': Predicted labels (N,)
            - 'y_probs': Prediction probabilities (N, n_classes)
            - 'accuracy': Accuracy score
            - 'balanced_accuracy': Balanced accuracy score
        alpha: Significance level (default: 0.05)
        correction: Multiple comparison correction method

    Returns:
        DataFrame with pairwise test results
    """
    results = []
    models = list(predictions_dict.keys())
    n_comparisons = len(list(combinations(models, 2)))

    # First pass: collect all p-values
    comparison_data = []
    for model_a, model_b in combinations(models, 2):
        data_a = predictions_dict[model_a]
        data_b = predictions_dict[model_b]

        # Verify same ground truth
        if not np.array_equal(data_a["y_true"], data_b["y_true"]):
            raise ValueError(
                f"Ground truth differs between {model_a} and {model_b}. "
                "McNemar requires same test set."
            )

        test_result = mcnemar_test(data_a["y_true"], data_a["y_pred"], data_b["y_pred"])
        comparison_data.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "test_result": test_result,
                "acc_a": data_a["accuracy"],
                "acc_b": data_b["accuracy"],
                "bacc_a": data_a["balanced_accuracy"],
                "bacc_b": data_b["balanced_accuracy"],
            }
        )

    # Extract p-values for correction
    p_values = np.array([comp["test_result"]["p_value"] for comp in comparison_data])

    # Apply correction
    if correction == "bonferroni":
        corrected_alpha = alpha / n_comparisons
        is_significant = p_values < corrected_alpha
        correction_desc = f"Bonferroni (α={corrected_alpha:.4f})"
    elif correction == "fdr_bh":
        from scipy.stats import false_discovery_control

        is_significant = false_discovery_control(p_values, method="bh")
        corrected_alpha = alpha
        correction_desc = f"FDR (Benjamini-Hochberg, α={alpha})"
    elif correction == "none":
        is_significant = p_values < alpha
        corrected_alpha = alpha
        correction_desc = "None"
    else:
        raise ValueError(f"Unknown correction method: {correction}")

    # Second pass: build results with corrected significance
    for i, comp in enumerate(comparison_data):
        model_a = comp["model_a"]
        model_b = comp["model_b"]
        test_result = comp["test_result"]
        p_value = test_result["p_value"]

        # Map winner from A/B to model name
        raw_winner = test_result["winner"]
        if raw_winner == "A":
            winner_name = model_a
        elif raw_winner == "B":
            winner_name = model_b
        else:
            winner_name = "none"

        # Only declare winner if significant after correction
        winner = winner_name if is_significant[i] else "none"

        contingency = test_result["contingency"]
        results.append(
            {
                "Model_A": model_a,
                "Model_B": model_b,
                "Acc_A": comp["acc_a"],
                "Acc_B": comp["acc_b"],
                "BAcc_A": comp["bacc_a"],
                "BAcc_B": comp["bacc_b"],
                "A_correct_B_wrong": contingency["A_correct_B_wrong"],
                "A_wrong_B_correct": contingency["A_wrong_B_correct"],
                "p_value": p_value,
                "alpha_corrected": corrected_alpha,
                "Significant": "Yes" if is_significant[i] else "No",
                "Winner": winner if winner != "none" else "No significant difference",
                "Correction": correction_desc,
                "n_samples": test_result["n_samples"],
            }
        )

    return pd.DataFrame(results)


def plot_wilcoxon_distributions(
    scores_A: np.ndarray,
    scores_B: np.ndarray,
    model_A_name: str,
    model_B_name: str,
    result: Dict,
    output_path: Path,
    no_title: bool = False,
):
    """Plot distributions of cross-fold scores with publication-quality styling.

    Args:
        scores_A: Scores for model A across folds
        scores_B: Scores for model B across folds
        model_A_name: Name of model A
        model_B_name: Name of model B
        result: Dictionary with Wilcoxon test results
        output_path: Directory to save plots
        no_title: If True, omit plot titles (for publication)
    """
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_DOUBLE, FIG_WIDTH_DOUBLE / 2.2))

    # Convert to percentage for display
    scores_A_pct = scores_A * 100
    scores_B_pct = scores_B * 100

    # Box plot with strip plot overlay
    data = pd.DataFrame({model_A_name: scores_A_pct, model_B_name: scores_B_pct})
    data_melted = data.melt(var_name="Model", value_name="Score")

    # Use custom colors
    box_colors = [COLOR_PALETTE["primary"], COLOR_PALETTE["secondary"]]

    sns.boxplot(
        x="Model",
        y="Score",
        data=data_melted,
        ax=axes[0],
        palette=box_colors,
        width=0.5,
        linewidth=1.2,
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.6},
        boxprops={"alpha": 0.8},
        medianprops={"color": "#333333", "linewidth": 1.5},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )

    # Add individual data points
    sns.stripplot(
        x="Model",
        y="Score",
        data=data_melted,
        ax=axes[0],
        color="#333333",
        alpha=0.7,
        size=5,
        jitter=0.15,
        zorder=10,
    )

    if not no_title:
        axes[0].set_title("Cross-Fold Score Distributions")
    axes[0].set_ylabel("Balanced Accuracy (%)")
    axes[0].set_xlabel("")

    # Add mean markers
    for i, (model, scores) in enumerate(
        [(model_A_name, scores_A_pct), (model_B_name, scores_B_pct)]
    ):
        mean = np.mean(scores)
        axes[0].plot(
            i,
            mean,
            "D",
            color="white",
            markersize=7,
            markeredgecolor="#333333",
            markeredgewidth=1.5,
            zorder=15,
            label="Mean" if i == 0 else "",
        )

    # Paired differences plot
    differences = (scores_A - scores_B) * 100  # Convert to percentage
    n_folds = len(differences)

    # Reference line at zero
    axes[1].axhline(
        0,
        color=COLOR_PALETTE["octonary"],
        linestyle="--",
        linewidth=1.2,
        label="No difference",
        zorder=1,
    )

    # Plot paired differences with connecting lines
    axes[1].plot(
        range(n_folds),
        differences,
        "o-",
        color=COLOR_PALETTE["primary"],
        markersize=7,
        linewidth=1.5,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=5,
    )

    # Median difference line
    median_diff = np.median(differences)
    axes[1].axhline(
        median_diff,
        color=COLOR_PALETTE["tertiary"],
        linestyle="-.",
        linewidth=1.5,
        label=f"Median Δ = {median_diff:+.2f}%",
        zorder=2,
    )

    # Add shaded region to indicate positive/negative
    axes[1].fill_between(
        [-0.5, n_folds - 0.5],
        [0, 0],
        [max(differences.max(), 0) * 1.1] * 2,
        alpha=0.05,
        color=COLOR_PALETTE["tertiary"],
        zorder=0,
    )
    axes[1].fill_between(
        [-0.5, n_folds - 0.5],
        [0, 0],
        [min(differences.min(), 0) * 1.1] * 2,
        alpha=0.05,
        color=COLOR_PALETTE["septenary"],
        zorder=0,
    )

    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel(f"Δ Score ({model_A_name} − {model_B_name}) (%)")
    if not no_title:
        axes[1].set_title("Paired Differences")
    axes[1].set_xticks(range(n_folds))
    axes[1].set_xlim(-0.5, n_folds - 0.5)
    axes[1].legend(loc="best", framealpha=0.95)

    # Add statistical annotation
    p_val = result["p_value"]
    sig_marker = (
        "***"
        if p_val < 0.001
        else "**"
        if p_val < 0.01
        else "*"
        if p_val < 0.05
        else "n.s."
    )
    stat_text = f"W = {result['statistic']:.1f}, p = {p_val:.3f} ({sig_marker})"

    if not no_title:
        fig.suptitle(
            f"Wilcoxon Signed-Rank Test: {model_A_name} vs {model_B_name}\n{stat_text}",
            y=1.02,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        output_path / "wilcoxon_distributions.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_path / "wilcoxon_distributions.pdf", bbox_inches="tight")
    plt.close()

    print(f"✓ Saved Wilcoxon distribution plots to {output_path}")


def plot_distributions(
    scores_dict: Dict[str, np.ndarray],
    metric_name: str,
    output_dir: Path,
    no_title: bool = False,
):
    """Create publication-quality box plot and violin plot comparing distributions.

    Creates two separate figures:
    1. Box plot with strip plot overlay (distribution)
    2. Violin plot with embedded box plot (density)

    Args:
        scores_dict: Dictionary mapping model names to score arrays
        metric_name: Name of the metric being plotted
        output_dir: Directory to save plots
        no_title: If True, omit plot titles (for publication)
    """

    n_models = len(scores_dict)

    # Create shortened labels for display
    original_names = list(scores_dict.keys())
    label_map = create_label_mapping(original_names, max_length=18)

    # Prepare data for seaborn with shortened labels
    plot_data = []
    for model, scores in scores_dict.items():
        short_label = label_map[model]
        for score in scores:
            plot_data.append({"Model": short_label, "Score": score * 100})

    df_plot = pd.DataFrame(plot_data)

    # Get the order of labels (preserve original order)
    label_order = [label_map[name] for name in original_names]

    # Determine figure dimensions based on number of models
    if n_models <= 3:
        fig_width = FIG_WIDTH_SINGLE * 1.2
    elif n_models <= 5:
        fig_width = FIG_WIDTH_1_5
    else:
        fig_width = FIG_WIDTH_DOUBLE * 0.7

    # Improved aspect ratio for better readability (single plot)
    fig_height = fig_width * 0.8

    # Use colorblind-friendly palette
    colors = CATEGORICAL_COLORS[:n_models]

    # Calculate y-axis limits with extra space for annotations
    y_data_min = df_plot["Score"].min()
    y_data_max = df_plot["Score"].max()
    y_range = y_data_max - y_data_min
    y_min = y_data_min - y_range * 0.08
    y_max = y_data_max + y_range * 0.18  # Extra space for annotations at top

    # ==================== FIGURE 1: Box plot (Distribution) ====================
    fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))

    # Box plot with strip plot overlay
    sns.boxplot(
        data=df_plot,
        x="Model",
        y="Score",
        hue="Model",
        order=label_order,
        hue_order=label_order,
        ax=ax1,
        palette=colors,
        legend=False,
        width=0.55,
        linewidth=1.2,
        flierprops={
            "marker": "o",
            "markersize": 4,
            "alpha": 0.6,
            "markeredgecolor": "#333333",
            "markeredgewidth": 0.5,
        },
        boxprops={"alpha": 0.85, "edgecolor": "#333333"},
        medianprops={"color": "#333333", "linewidth": 2},
        whiskerprops={"linewidth": 1.2, "color": "#555555"},
        capprops={"linewidth": 1.2, "color": "#555555"},
    )

    # Add individual data points with slight transparency
    sns.stripplot(
        data=df_plot,
        x="Model",
        y="Score",
        order=label_order,
        ax=ax1,
        color="#444444",
        alpha=0.7,
        size=5,
        jitter=0.15,
        zorder=10,
    )

    if not no_title:
        ax1.set_title(f"{metric_name} Distribution", fontweight="bold", pad=10)
    ax1.set_ylabel(f"{metric_name} (%)")
    ax1.set_xlabel("")

    # Rotate x-labels for readability
    if n_models > 3:
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha="right")

    # Add mean markers with values - positioned at top of plot
    for i, (model, scores) in enumerate(scores_dict.items()):
        mean = scores.mean() * 100
        std = scores.std() * 100
        # Diamond marker at the mean
        ax1.plot(
            i,
            mean,
            "D",
            color="white",
            markersize=7,
            markeredgecolor="#333333",
            markeredgewidth=1.5,
            zorder=15,
        )
        # Annotation at top of plot area for cleaner look
        ax1.annotate(
            f"{mean:.1f}±{std:.1f}",
            xy=(i, y_max - y_range * 0.02),
            ha="center",
            va="top",
            fontsize=7,
            fontweight="bold",
            color="#333333",
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8
            ),
        )

    ax1.set_ylim(y_min, y_max)
    ax1.yaxis.grid(True, linestyle="-", alpha=0.3, color="#CCCCCC", zorder=0)
    ax1.set_axisbelow(True)

    plt.tight_layout()

    plt.savefig(
        output_dir / f"{metric_name.lower()}_distribution.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.savefig(
        output_dir / f"{metric_name.lower()}_distribution.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig1)

    print(
        f"Saved distribution plot: {output_dir}/{metric_name.lower()}_distribution.png and {output_dir}/{metric_name.lower()}_distribution.pdf"
    )

    # ==================== FIGURE 2: Violin plot (Density) ====================
    fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))

    # Violin plot with embedded box plot
    sns.violinplot(
        data=df_plot,
        x="Model",
        y="Score",
        hue="Model",
        order=label_order,
        hue_order=label_order,
        ax=ax2,
        palette=colors,
        inner=None,  # We'll add our own box
        cut=0,
        legend=False,
        linewidth=1.0,
        saturation=0.9,
    )

    # Add a thin box plot inside the violin
    sns.boxplot(
        data=df_plot,
        x="Model",
        y="Score",
        order=label_order,
        ax=ax2,
        color="white",
        width=0.12,
        linewidth=1.0,
        fliersize=0,
        boxprops={"zorder": 10, "alpha": 0.95, "edgecolor": "#333333"},
        medianprops={"color": "#333333", "linewidth": 1.5, "zorder": 11},
        whiskerprops={"linewidth": 1.0, "zorder": 10, "color": "#333333"},
        capprops={"linewidth": 1.0, "zorder": 10, "color": "#333333"},
    )

    if not no_title:
        ax2.set_title(f"{metric_name} Density", fontweight="bold", pad=10)
    ax2.set_ylabel(f"{metric_name} (%)")
    ax2.set_xlabel("")

    # Rotate x-labels for readability
    if n_models > 3:
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=40, ha="right")

    ax2.set_ylim(y_min, y_max)
    ax2.yaxis.grid(True, linestyle="-", alpha=0.3, color="#CCCCCC", zorder=0)
    ax2.set_axisbelow(True)

    plt.tight_layout()

    plt.savefig(
        output_dir / f"{metric_name.lower()}_density.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.savefig(
        output_dir / f"{metric_name.lower()}_density.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig2)

    print(
        f"Saved density plot: {output_dir}/{metric_name.lower()}_density.png and {output_dir}/{metric_name.lower()}_density.pdf"
    )


def plot_pairwise_matrix(
    df_tests: pd.DataFrame,
    output_dir: Path,
    test_name: str = "Wilcoxon",
    no_title: bool = False,
):
    """Create publication-quality heatmap of pairwise p-values.

    Args:
        df_tests: DataFrame with test results
        output_dir: Directory to save plots
        test_name: Name of the test (for title and filenames)
        no_title: If True, omit plot title (for publication)
    """

    # Get unique models and create shortened labels
    models_original = sorted(
        set(df_tests["Model_A"].tolist() + df_tests["Model_B"].tolist())
    )
    label_map = create_label_mapping(models_original, max_length=18)
    models_short = [label_map[m] for m in models_original]
    n = len(models_original)

    # Create matrices
    p_matrix = np.ones((n, n))
    winner_matrix = np.empty((n, n), dtype=object)
    winner_matrix.fill("")
    sig_matrix = np.zeros((n, n), dtype=bool)  # Track significance

    for _, row in df_tests.iterrows():
        i = models_original.index(row["Model_A"])
        j = models_original.index(row["Model_B"])
        p_matrix[i, j] = row["p_value"]
        p_matrix[j, i] = row["p_value"]

        is_sig = row["Significant"] == "Yes"
        sig_matrix[i, j] = is_sig
        sig_matrix[j, i] = is_sig

        # Winner annotation with p-value significance markers
        if is_sig:
            p = row["p_value"]
            marker = "***" if p < 0.001 else "**" if p < 0.01 else "*"
            if row["Winner"] == row["Model_A"]:
                winner_matrix[i, j] = f"▲{marker}"
                winner_matrix[j, i] = f"▼{marker}"
            elif row["Winner"] == row["Model_B"]:
                winner_matrix[i, j] = f"▼{marker}"
                winner_matrix[j, i] = f"▲{marker}"
        else:
            winner_matrix[i, j] = "–"
            winner_matrix[j, i] = "–"

    # Determine figure size based on number of models (smaller = larger text)
    fig_size = max(FIG_WIDTH_SINGLE * 1.2, 1.2 + 0.5 * n)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.95))

    # Mask diagonal
    mask = np.eye(n, dtype=bool)

    # Create custom colormap: green (significant, low p) to red (not significant, high p)
    # Using a diverging colormap centered at 0.05
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    # Custom colormap: dark green -> light green -> white -> light red -> dark red
    cmap_colors = [
        (0.0, "#1a9641"),  # Very significant (p ≈ 0)
        (0.25, "#a6d96a"),  # Significant (p < 0.025)
        (0.5, "#ffffbf"),  # Borderline (p = 0.05)
        (0.75, "#fdae61"),  # Not quite significant (p ~ 0.075)
        (1.0, "#d7191c"),  # Not significant (p >= 0.1)
    ]
    cmap = LinearSegmentedColormap.from_list(
        "significance", [c[1] for c in cmap_colors], N=256
    )

    # Use TwoSlopeNorm to center at 0.05
    norm = TwoSlopeNorm(vmin=0, vcenter=0.05, vmax=0.1)

    # Create heatmap with shortened labels
    sns.heatmap(
        p_matrix,
        annot=winner_matrix,
        fmt="",
        cmap=cmap,
        norm=norm,
        mask=mask,
        cbar_kws={
            "label": "",  # We'll add label manually to avoid overlap
            # "shrink": 0.7 if args.no_title else 0.78,
            "shrink": 0.7,
            "pad": 0.02,
        },
        xticklabels=models_short,
        yticklabels=models_short,
        ax=ax,
        linewidths=0.8,
        linecolor="#CCCCCC",
        annot_kws={"fontsize": 8, "fontweight": "medium"},
        square=True,
    )

    # Style the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.axhline(y=0.05, color="#333333", linewidth=1.5, linestyle="-")
    # Position alpha text to the right of the colorbar, avoiding overlap with tick labels
    # cbar.ax.text(
    #     3.5,
    #     0.05,
    #     "α=0.05",
    #     va="center",
    #     ha="left",
    #     fontsize=7,
    #     color="#333333",
    #     fontweight="medium",
    #     transform=cbar.ax.get_yaxis_transform(),
    # )
    cbar.ax.tick_params(labelsize=7)
    # Add p-value label at the top of colorbar with more padding
    cbar.ax.set_ylabel("p-value", fontsize=8, labelpad=12)

    # Set ticks at meaningful values
    cbar.set_ticks([0, 0.01, 0.05, 0.1])
    cbar.set_ticklabels(["0", "0.01", "0.05", "0.10"])

    if not no_title:
        ax.set_title(
            f"Pairwise {test_name} Test\n"
            + r"(▲ = row wins, ▼ = column wins, – = n.s.; *p<0.05, **p<0.01, ***p<0.001)",
            pad=12,
        )

    # Rotate labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()

    # Save
    plt.savefig(
        output_dir / f"{test_name.lower()}_significance_matrix.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.savefig(
        output_dir / f"{test_name.lower()}_significance_matrix.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    print(
        f"Saved {test_name} significance matrix: {output_dir}/{test_name.lower()}_significance_matrix.png and {output_dir}/{test_name.lower()}_significance_matrix.pdf"
    )


def generate_rankings(
    scores_dict: Dict[str, np.ndarray], metric_name: str
) -> pd.DataFrame:
    """Generate rankings table with statistics."""

    rankings = []
    for model, scores in scores_dict.items():
        rankings.append(
            {
                "Model": model,
                "Mean": scores.mean(),
                "Std": scores.std(),
                "Median": np.median(scores),
                "Min": scores.min(),
                "Max": scores.max(),
                "N_folds": len(scores),
            }
        )

    df_rank = pd.DataFrame(rankings)
    df_rank = df_rank.sort_values("Mean", ascending=False).reset_index(drop=True)
    df_rank.index = df_rank.index + 1  # Rankings start at 1
    df_rank.index.name = "Rank"

    for col in ["Mean", "Std", "Median", "Min", "Max"]:
        df_rank[col] = df_rank[col] * 100

    return df_rank


def run_mcnemar_comparison(
    csv_A: Path,
    csv_B: Path,
    output_dir: Path,
    model_A_name: Optional[str] = None,
    model_B_name: Optional[str] = None,
    no_title: bool = False,
):
    """Run McNemar test comparing two models on same test set."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_A_name = model_A_name or csv_A.stem
    model_B_name = model_B_name or csv_B.stem

    print("\n" + "=" * 80)
    print(f"McNEMAR TEST: {model_A_name} vs {model_B_name}")
    print("=" * 80)

    # Load predictions (classes are already extracted correctly by load_predictions)
    _, classes_A, _, y_probs_A, y_true_A = load_predictions(csv_A)
    _, classes_B, _, y_probs_B, y_true_B = load_predictions(csv_B)

    # Validate classes match
    if classes_A != classes_B:
        raise ValueError(
            f"Class lists differ between models: {classes_A} vs {classes_B}"
        )

    classes = classes_A

    # Validate ground truth matches
    if not np.array_equal(y_true_A, y_true_B):
        raise ValueError(
            "Ground truth labels differ between models (not same test set)"
        )

    y_true = y_true_A

    # Compute metrics with bootstrap CIs
    print("\nComputing metrics with bootstrap confidence intervals...")
    metrics_A = compute_metrics(y_true, y_probs_A, classes)
    metrics_B = compute_metrics(y_true, y_probs_B, classes)

    # Get argmax predictions for McNemar test
    # Note: y_true is already integer indices from load_predictions
    y_pred_A = np.argmax(y_probs_A, axis=1)
    y_pred_B = np.argmax(y_probs_B, axis=1)

    result = mcnemar_test(y_true, y_pred_A, y_pred_B)

    print(f"\nTest Set Size: {result['n_samples']} subjects")
    print("\nModel Performances (with 95% bootstrap CI):")
    table = [
        ["Metric", model_A_name, model_B_name, "Difference"],
        [
            "Accuracy",
            f"{metrics_A['accuracy']:.4f} [{metrics_A['accuracy_ci'][0]:.4f}, {metrics_A['accuracy_ci'][1]:.4f}]",
            f"{metrics_B['accuracy']:.4f} [{metrics_B['accuracy_ci'][0]:.4f}, {metrics_B['accuracy_ci'][1]:.4f}]",
            f"{metrics_A['accuracy'] - metrics_B['accuracy']:.4f}",
        ],
        [
            "Balanced Accuracy",
            f"{metrics_A['balanced_accuracy']:.4f} [{metrics_A['balanced_accuracy_ci'][0]:.4f}, {metrics_A['balanced_accuracy_ci'][1]:.4f}]",
            f"{metrics_B['balanced_accuracy']:.4f} [{metrics_B['balanced_accuracy_ci'][0]:.4f}, {metrics_B['balanced_accuracy_ci'][1]:.4f}]",
            f"{metrics_A['balanced_accuracy'] - metrics_B['balanced_accuracy']:.4f}",
        ],
        [
            "ROC-AUC (macro)",
            f"{metrics_A['roc_auc_macro']:.4f} [{metrics_A['roc_auc_macro_ci'][0]:.4f}, {metrics_A['roc_auc_macro_ci'][1]:.4f}]",
            f"{metrics_B['roc_auc_macro']:.4f} [{metrics_B['roc_auc_macro_ci'][0]:.4f}, {metrics_B['roc_auc_macro_ci'][1]:.4f}]",
            f"{metrics_A['roc_auc_macro'] - metrics_B['roc_auc_macro']:.4f}",
        ],
    ]
    print(tabulate(table, headers="firstrow", tablefmt="grid"))

    print("\nMcNemar Contingency Table:")
    cont_table = [
        ["", f"{model_B_name} Correct", f"{model_B_name} Wrong"],
        [
            f"{model_A_name} Correct",
            result["contingency"]["both_correct"],
            result["contingency"]["A_correct_B_wrong"],
        ],
        [
            f"{model_A_name} Wrong",
            result["contingency"]["A_wrong_B_correct"],
            result["contingency"]["both_wrong"],
        ],
    ]
    print(tabulate(cont_table, headers="firstrow", tablefmt="grid"))

    print("\nMcNemar Test Result:")
    print(f"  Test used: {result['test_used']}")
    if result["chi2_statistic"] is not None:
        print(f"  χ² statistic: {result['chi2_statistic']:.4f}")
    print(f"  p-value: {result['p_value']:.4f}")
    print(f"  Winner: {result['winner'].upper()}")
    print(f"  {result['interpretation']}")

    report_path = output_dir / "mcnemar_report.txt"
    with open(report_path, "w") as f:
        f.write("McNemar Test Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model A: {model_A_name}\n")
        f.write(f"Model B: {model_B_name}\n")
        f.write(f"Test Set: {result['n_samples']} subjects\n\n")
        f.write(tabulate(table, headers="firstrow", tablefmt="grid") + "\n\n")
        f.write(tabulate(cont_table, headers="firstrow", tablefmt="grid") + "\n\n")
        f.write(f"Test used: {result['test_used']}\n")
        if result["chi2_statistic"] is not None:
            f.write(f"χ² statistic: {result['chi2_statistic']:.4f}\n")
        f.write(f"p-value: {result['p_value']:.4f}\n")
        f.write(f"Winner: {result['winner'].upper()}\n")
        f.write(f"{result['interpretation']}\n")

    print(f"\n✓ Saved report to {report_path}")
    print("=" * 80 + "\n")


def run_wilcoxon_comparison(
    dir_A: Path,
    dir_B: Path,
    pattern: str,
    output_dir: Path,
    model_A_name: Optional[str] = None,
    model_B_name: Optional[str] = None,
    metric: str = "balanced_accuracy",
    no_title: bool = False,
):
    """Run Wilcoxon test comparing cross-fold performance."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize metric through METRIC_MAP
    normalized_metric = METRIC_MAP.get(metric)
    if normalized_metric is None:
        raise ValueError(
            f"Unknown metric: {metric}. Valid choices: {list(METRIC_MAP.keys())}"
        )

    # Map internal metric name to compute_metrics keys
    # compute_metrics uses 'roc_auc_macro' but bootstrap uses 'roc_auc'
    compute_metrics_key_map = {
        "roc_auc": "roc_auc_macro",
    }
    compute_metric_key = compute_metrics_key_map.get(
        normalized_metric, normalized_metric
    )

    model_A_name = model_A_name or dir_A.name
    model_B_name = model_B_name or dir_B.name

    print("\n" + "=" * 80)
    print(f"WILCOXON TEST (Cross-Fold): {model_A_name} vs {model_B_name}")
    print(f"Metric: {normalized_metric.upper()}")
    print("=" * 80)

    files_A = sorted(dir_A.glob(pattern))
    files_B = sorted(dir_B.glob(pattern))

    if len(files_A) != len(files_B):
        raise ValueError(f"Different number of folds: {len(files_A)} vs {len(files_B)}")

    print(f"\nFound {len(files_A)} folds to compare")

    scores_A = []
    scores_B = []

    print("\nComputing metrics with bootstrap confidence intervals...")
    for i, (file_A, file_B) in enumerate(zip(files_A, files_B)):
        _, classes_A, _, y_probs_A, y_true_A = load_predictions(file_A)
        _, classes_B, _, y_probs_B, y_true_B = load_predictions(file_B)

        if classes_A != classes_B:
            raise ValueError(
                f"Class lists differ in fold {i}: {classes_A} vs {classes_B}"
            )

        metrics_A = compute_metrics(y_true_A, y_probs_A, classes_A)
        metrics_B = compute_metrics(y_true_B, y_probs_B, classes_B)

        scores_A.append(metrics_A[compute_metric_key])
        scores_B.append(metrics_B[compute_metric_key])

        ci_A = metrics_A[f"{compute_metric_key}_ci"]
        ci_B = metrics_B[f"{compute_metric_key}_ci"]
        print(
            f"  Fold {i}: {model_A_name}={metrics_A[compute_metric_key]:.4f} [{ci_A[0]:.4f}, {ci_A[1]:.4f}], "
            f"{model_B_name}={metrics_B[compute_metric_key]:.4f} [{ci_B[0]:.4f}, {ci_B[1]:.4f}]"
        )

    scores_A = np.array(scores_A)
    scores_B = np.array(scores_B)

    result = wilcoxon_test(scores_A, scores_B)

    print("\nWilcoxon Signed-Rank Test Result:")
    print(f"  Statistic: {result['statistic']:.4f}")
    print(f"  p-value: {result['p_value']:.4f}")
    print(f"  Winner: {result['winner'].upper()}")
    print(f"  {result['interpretation']}")
    print(
        f"\n  {model_A_name}: {result['scores_A_mean']:.4f} ± {result['scores_A_std']:.4f}"
    )
    print(
        f"  {model_B_name}: {result['scores_B_mean']:.4f} ± {result['scores_B_std']:.4f}"
    )
    print(f"  Median difference: {result['median_difference']:.4f}")
    print(f"  Mean difference: {result['mean_difference']:.4f}")

    plot_wilcoxon_distributions(
        scores_A, scores_B, model_A_name, model_B_name, result, output_dir
    )

    report_path = output_dir / "wilcoxon_report.txt"
    with open(report_path, "w") as f:
        f.write("Wilcoxon Signed-Rank Test Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model A: {model_A_name}\n")
        f.write(f"Model B: {model_B_name}\n")
        f.write(f"Metric: {metric}\n")
        f.write(f"Number of folds: {result['n_folds']}\n\n")
        f.write(f"Statistic: {result['statistic']:.4f}\n")
        f.write(f"p-value: {result['p_value']:.4f}\n")
        f.write(f"Winner: {result['winner'].upper()}\n")
        f.write(f"{result['interpretation']}\n\n")
        f.write(
            f"{model_A_name}: {result['scores_A_mean']:.4f} ± {result['scores_A_std']:.4f}\n"
        )
        f.write(
            f"{model_B_name}: {result['scores_B_mean']:.4f} ± {result['scores_B_std']:.4f}\n"
        )
        f.write(f"Median difference: {result['median_difference']:.4f}\n")
        f.write(f"Mean difference: {result['mean_difference']:.4f}\n")

    print(f"\n✓ Saved report to {report_path}")
    print("=" * 80 + "\n")


def run_architecture_comparison(
    model_dirs: List[Path],
    pattern: str,
    metrics: List[str],
    output_dir: Path,
    arch_names: Optional[List[str]] = None,
    correction: str = "bonferroni",
    no_title: bool = False,
):
    """Compare multiple architectures using Wilcoxon tests.

    Args:
        model_dirs: List of directories containing fold prediction files
        pattern: Glob pattern for prediction files
        metrics: List of metrics to compare (will be normalized via METRIC_MAP)
        output_dir: Directory to save results
        arch_names: Custom names for architectures
        correction: Multiple comparison correction method
        no_title: If True, omit plot titles
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize metrics through METRIC_MAP
    normalized_metrics = []
    for m in metrics:
        normalized = METRIC_MAP.get(m)
        if normalized is None:
            raise ValueError(
                f"Unknown metric: {m}. Valid choices: {list(METRIC_MAP.keys())}"
            )
        normalized_metrics.append(normalized)
    # Remove duplicates while preserving order
    normalized_metrics = list(dict.fromkeys(normalized_metrics))

    # Architecture names
    if arch_names:
        if len(arch_names) != len(model_dirs):
            raise ValueError(
                f"Number of names ({len(arch_names)}) != number of directories ({len(model_dirs)})"
            )
    else:
        arch_names = [d.name for d in model_dirs]

    print("=" * 80)
    print("ARCHITECTURE COMPARISON")
    print("=" * 80)
    print(f"Architectures: {', '.join(arch_names)}")
    print(f"Metrics: {', '.join([m.upper() for m in normalized_metrics])}")
    print(f"Pattern: {pattern}")
    print()

    # Load predictions and compute all metrics for each architecture
    # Store all bootstrap results to avoid recomputation
    all_bootstrap_results = {}  # {arch_name: {fold: bootstrap_results}}
    fold_files_dict = {}  # {arch_name: {fold: filepath}}

    for arch_name, model_dir in zip(arch_names, model_dirs):
        print(f"Loading {arch_name} from {model_dir}...")

        fold_files = load_predictions_from_dir(model_dir, pattern)

        if not fold_files:
            print(f"\tNo files found matching pattern '{pattern}'")
            continue

        fold_files_dict[arch_name] = fold_files
        print(f"\tFound {len(fold_files)} folds: {sorted(fold_files.keys())}")

        # Compute bootstrap metrics for each fold (once, for all metrics)
        arch_results = {}
        for fold, filepath in sorted(fold_files.items()):
            _, _, _, probs, true_labels = load_predictions(filepath)

            bootstrap_results = compute_bootstrap_metrics(
                y_true=true_labels,
                y_pred_probs=probs,
                n_bootstrap=10000,
                confidence=0.95,
                random_state=42,
                n_jobs=-1,
            )
            arch_results[fold] = bootstrap_results

        all_bootstrap_results[arch_name] = arch_results
        print()

    if len(all_bootstrap_results) < 2:
        raise ValueError("Need at least 2 architectures with data to compare")

    # Process each metric
    for metric in normalized_metrics:
        print("\n" + "=" * 80)
        print(f"PROCESSING METRIC: {metric.upper()}")
        print("=" * 80)

        # Create metric-specific output directory if multiple metrics
        if len(normalized_metrics) > 1:
            metric_output_dir = output_dir / metric
            metric_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            metric_output_dir = output_dir

        # Extract scores for this metric
        scores_dict = {}
        for arch_name in all_bootstrap_results:
            scores = []
            for fold in sorted(all_bootstrap_results[arch_name].keys()):
                bootstrap_result = all_bootstrap_results[arch_name][fold][metric]
                score = bootstrap_result["mean"]
                scores.append(score)
                print(
                    f"\t{arch_name} Fold {fold}: {score * 100:.2f}% "
                    f"(95% CI: [{bootstrap_result['lower'] * 100:.2f}, {bootstrap_result['upper'] * 100:.2f}])"
                )
            scores_dict[arch_name] = np.array(scores)
            print(
                f"\t{arch_name} Mean ± Std: {np.mean(scores) * 100:.2f} ± {np.std(scores) * 100:.2f}%"
            )

        # Check all have same number of folds
        fold_counts = [len(scores) for scores in scores_dict.values()]
        if len(set(fold_counts)) > 1:
            print(
                f"\tWARNING: Architectures have different numbers of folds: {fold_counts}"
            )
            print("\tUsing only common folds for comparison...")
            min_folds = min(fold_counts)
            scores_dict = {k: v[:min_folds] for k, v in scores_dict.items()}
            print(f"\tComparing on {min_folds} folds")

        # Generate rankings
        print("\n" + "-" * 40)
        print(f"RANKINGS ({metric.upper()})")
        print("-" * 40)
        df_rank = generate_rankings(scores_dict, metric.upper())
        print(df_rank.to_string())

        # Save rankings
        df_rank.to_csv(metric_output_dir / f"rankings_{metric}.csv")
        print(f"\nSaved: {metric_output_dir}/rankings_{metric}.csv")

        # Perform pairwise Wilcoxon tests
        print("\n" + "-" * 40)
        print(f"PAIRWISE WILCOXON TESTS ({metric.upper()})")
        print("-" * 40)
        df_tests = perform_wilcoxon_tests(scores_dict, correction=correction)
        print(df_tests.to_string(index=False))

        # Save test results
        df_tests.to_csv(metric_output_dir / f"wilcoxon_tests_{metric}.csv", index=False)
        print(f"\nSaved: {metric_output_dir}/wilcoxon_tests_{metric}.csv")

        # Count significant differences
        n_significant = (df_tests["p_value"] < 0.05).sum()
        print(
            f"\nSignificant differences (raw p<0.05): {n_significant}/{len(df_tests)}"
        )

        correction_method = df_tests["Correction"].iloc[0]
        n_corrected = (df_tests["Significant"] == "Yes").sum()
        print(
            f"Significant differences after {correction_method}: {n_corrected}/{len(df_tests)}"
        )

        # Generate plots
        print("\n" + "-" * 40)
        print(f"GENERATING VISUALIZATIONS ({metric.upper()})")
        print("-" * 40)

        plot_distributions(
            scores_dict, metric.upper(), metric_output_dir, no_title=no_title
        )
        plot_pairwise_matrix(df_tests, metric_output_dir, no_title=no_title)

        # Generate summary report
        report_path = metric_output_dir / f"wilcoxon_report_{metric}.txt"
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("ARCHITECTURE COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Metric: {metric.upper()}\n")
            f.write(f"Pattern: {pattern}\n")
            f.write(f"Architectures: {len(scores_dict)}\n")
            f.write(
                f"Folds per architecture: {len(next(iter(scores_dict.values())))}\n\n"
            )

            f.write("RANKINGS\n")
            f.write("-" * 80 + "\n")
            f.write(df_rank.to_string() + "\n\n")

            f.write("PAIRWISE WILCOXON TESTS\n")
            f.write("-" * 80 + "\n")
            f.write(df_tests.to_string(index=False) + "\n\n")

            f.write("=" * 80 + "\n")
            f.write("INTERPRETATION\n")
            f.write("=" * 80 + "\n\n")

            # Best model
            best_model = df_rank.iloc[0]["Model"]
            best_score = df_rank.iloc[0]["Mean"]
            f.write(f"Best Model: {best_model} ({best_score:.2f}%)\n\n")

            # Significant wins
            df_best = df_tests[
                (df_tests["Winner"] == best_model) & (df_tests["Significant"] == "Yes")
            ]
            if len(df_best) > 0:
                f.write(f"{best_model} significantly outperforms:\n")
                for _, row in df_best.iterrows():
                    other = (
                        row["Model_B"]
                        if row["Model_A"] == best_model
                        else row["Model_A"]
                    )
                    f.write(
                        f"  - {other} (p={row['p_value']:.4f}, diff={abs(row['Difference']) * 100:.2f}%)\n"
                    )
            else:
                f.write(
                    f"{best_model} does NOT significantly outperform any other model.\n"
                )

            f.write("\n" + "=" * 80 + "\n")
            f.write("STATISTICAL NOTE\n")
            f.write("=" * 80 + "\n")
            f.write("Wilcoxon signed-rank test is used because:\n")
            f.write("  1. Data is paired (same folds across architectures)\n")
            f.write("  2. Non-parametric (no normality assumption)\n")
            f.write("  3. Robust to outliers\n")
            f.write("\n")
            f.write("Significance level: α = 0.05\n")
            f.write(f"Number of comparisons: {len(df_tests)}\n")
            f.write(f"Correction method: {df_tests['Correction'].iloc[0]}\n")
            f.write(
                f"\n{n_corrected}/{len(df_tests)} comparisons significant after correction.\n"
            )

        print(f"Saved: {report_path}")

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"Metrics analyzed: {', '.join([m.upper() for m in normalized_metrics])}")
    return 0


def generate_ensemble_rankings(
    predictions_dict: Dict[str, Dict], metric_name: str = "balanced_accuracy"
) -> pd.DataFrame:
    """
    Generate rankings table from ensemble predictions.

    Args:
        predictions_dict: Dict mapping model names to dict with 'accuracy', 'balanced_accuracy', etc.
        metric_name: Metric to use for ranking
    """
    rankings = []
    for model, data in predictions_dict.items():
        rankings.append(
            {
                "Model": model,
                "Accuracy": data["accuracy"] * 100,
                "Balanced_Accuracy": data["balanced_accuracy"] * 100,
                "ROC_AUC": data.get("roc_auc_macro", 0) * 100,
                "MCC": data.get("mcc", 0) * 100,
                "N_samples": len(data["y_true"]),
            }
        )

    df_rank = pd.DataFrame(rankings)

    # Sort by the specified metric
    metric_col = (
        "Balanced_Accuracy" if metric_name == "balanced_accuracy" else "Accuracy"
    )
    df_rank = df_rank.sort_values(metric_col, ascending=False).reset_index(drop=True)
    df_rank.index = df_rank.index + 1  # Rankings start at 1
    df_rank.index.name = "Rank"

    return df_rank


def plot_ensemble_bar_comparison(
    predictions_dict: Dict[str, Dict], output_dir: Path, no_title: bool = False
):
    """Create publication-quality bar plot comparing ensemble accuracies.

    Args:
        predictions_dict: Dictionary mapping model names to prediction data
        output_dir: Directory to save plots
        no_title: If True, omit plot title (for publication)
    """

    models = list(predictions_dict.keys())
    n_models = len(models)

    # Create shortened labels for display
    label_map = create_label_mapping(models, max_length=18)
    models_short = [label_map[m] for m in models]

    accuracies = [predictions_dict[m]["accuracy"] * 100 for m in models]
    balanced_accs = [predictions_dict[m]["balanced_accuracy"] * 100 for m in models]

    # Get confidence intervals if available
    acc_cis = [predictions_dict[m].get("accuracy_ci", (None, None)) for m in models]
    bacc_cis = [
        predictions_dict[m].get("balanced_accuracy_ci", (None, None)) for m in models
    ]

    # Calculate error bars (convert to percentage)
    acc_errors = np.array(
        [
            [acc - ci[0] * 100 if ci[0] else 0, ci[1] * 100 - acc if ci[1] else 0]
            for acc, ci in zip(accuracies, acc_cis)
        ]
    ).T
    bacc_errors = np.array(
        [
            [bacc - ci[0] * 100 if ci[0] else 0, ci[1] * 100 - bacc if ci[1] else 0]
            for bacc, ci in zip(balanced_accs, bacc_cis)
        ]
    ).T

    x = np.arange(n_models)
    width = 0.35

    # Determine figure width (smaller = larger text)
    fig_width = max(FIG_WIDTH_1_5 * 0.8, 1.2 + 0.5 * n_models)
    fig, ax = plt.subplots(figsize=(fig_width, fig_width * 0.5))

    # Create bars with error bars
    bars1 = ax.bar(
        x - width / 2,
        accuracies,
        width,
        label="Accuracy",
        color=COLOR_PALETTE["primary"],
        edgecolor="#333333",
        linewidth=0.8,
        alpha=0.85,
        yerr=acc_errors if acc_errors.any() else None,
        capsize=3,
        error_kw={"linewidth": 1.0, "capthick": 1.0, "color": "#333333"},
    )
    bars2 = ax.bar(
        x + width / 2,
        balanced_accs,
        width,
        label="Balanced Accuracy",
        color=COLOR_PALETTE["secondary"],
        edgecolor="#333333",
        linewidth=0.8,
        alpha=0.85,
        yerr=bacc_errors if bacc_errors.any() else None,
        capsize=3,
        error_kw={"linewidth": 1.0, "capthick": 1.0, "color": "#333333"},
    )

    ax.set_ylabel("Score (%)")
    ax.set_xlabel("")
    if not no_title:
        ax.set_title("Ensemble Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(
        models_short,
        rotation=35 if n_models > 3 else 0,
        ha="right" if n_models > 3 else "center",
    )

    # Set y-axis limits with some padding (extra top padding for bar labels)
    y_min = min(min(accuracies), min(balanced_accs)) - 5
    y_max = max(max(accuracies), max(balanced_accs)) + 5
    ax.set_ylim(max(0, y_min), min(100, y_max))

    # Add legend below the plot to avoid overlap with bar labels
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25 if n_models > 3 else -0.12),
        ncol=2,
        framealpha=0.95,
        fontsize=8,
    )

    # Add value labels on bars
    def add_bar_labels(bars, values, errors=None):
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            offset = errors[i] if errors is not None and len(errors) > i else 0
            ax.annotate(
                f"{val:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height + offset),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="medium",
                color="#333333",
            )

    # Only add labels if not too crowded
    if n_models <= 6:
        add_bar_labels(bars1, accuracies, acc_errors[1] if acc_errors.any() else None)
        add_bar_labels(
            bars2, balanced_accs, bacc_errors[1] if bacc_errors.any() else None
        )

    # Add subtle horizontal grid lines
    ax.yaxis.grid(True, linestyle="-", alpha=0.3, color="#CCCCCC")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(
        output_dir / "ensemble_comparison.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.savefig(
        output_dir / "ensemble_comparison.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    print(
        f"Saved ensemble comparison plot: {output_dir}/ensemble_comparison.png and {output_dir}/ensemble_comparison.pdf"
    )


def run_mcnemar_architecture_comparison(
    ensemble_csvs: List[Path],
    output_dir: Path,
    arch_names: Optional[List[str]] = None,
    correction: str = "bonferroni",
    no_title: bool = False,
):
    """
    Compare multiple architectures using McNemar tests on ensemble predictions.

    This is the preferred method for OOD evaluation where all models are evaluated
    on the same external test set. Each CSV should contain ensemble predictions
    (average probabilities across K folds, then argmax).

    Args:
        ensemble_csvs: List of CSV files with ensemble predictions
        output_dir: Directory to save results
        arch_names: Custom names for architectures (default: filename stems)
        correction: Multiple comparison correction method
        no_title: If True, omit plot titles (for publication)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Architecture names
    if arch_names:
        if len(arch_names) != len(ensemble_csvs):
            raise ValueError(
                f"Number of names ({len(arch_names)}) != number of CSVs ({len(ensemble_csvs)})"
            )
    else:
        arch_names = [csv.stem for csv in ensemble_csvs]

    print("=" * 80)
    print("MCNEMAR ARCHITECTURE COMPARISON (Ensemble Predictions)")
    print("=" * 80)
    print(f"Architectures: {', '.join(arch_names)}")
    print(f"Correction: {correction}")
    print()

    # Load ensemble predictions for each architecture
    predictions_dict = {}
    reference_subjects = None
    reference_y_true = None

    for arch_name, csv_path in zip(arch_names, ensemble_csvs):
        print(f"Loading {arch_name} from {csv_path}...")

        df, classes, pred_cols, y_probs, y_true = load_predictions(csv_path)
        subjects = (
            df["Subject"].values if "Subject" in df.columns else np.arange(len(df))
        )
        y_pred = np.argmax(y_probs, axis=1)

        # Compute metrics
        bootstrap_results = compute_bootstrap_metrics(
            y_true=y_true,
            y_pred_probs=y_probs,
            n_bootstrap=10000,
            confidence=0.95,
            random_state=42,
            n_jobs=-1,
        )

        predictions_dict[arch_name] = {
            "subjects": subjects,
            "classes": classes,
            "y_probs": y_probs,
            "y_true": y_true,
            "y_pred": y_pred,
            "accuracy": bootstrap_results["accuracy"]["mean"],
            "balanced_accuracy": bootstrap_results["balanced_accuracy"]["mean"],
            "roc_auc_macro": bootstrap_results["roc_auc"]["mean"],
            "mcc": bootstrap_results["mcc"]["mean"],
            "accuracy_ci": (
                bootstrap_results["accuracy"]["lower"],
                bootstrap_results["accuracy"]["upper"],
            ),
            "balanced_accuracy_ci": (
                bootstrap_results["balanced_accuracy"]["lower"],
                bootstrap_results["balanced_accuracy"]["upper"],
            ),
        }

        print(f"\t{len(y_true)} subjects, {len(classes)} classes")
        print(
            f"\tAccuracy: {bootstrap_results['accuracy']['mean'] * 100:.2f}% "
            f"[{bootstrap_results['accuracy']['lower'] * 100:.2f}, {bootstrap_results['accuracy']['upper'] * 100:.2f}]"
        )
        print(
            f"\tBalanced Accuracy: {bootstrap_results['balanced_accuracy']['mean'] * 100:.2f}% "
            f"[{bootstrap_results['balanced_accuracy']['lower'] * 100:.2f}, {bootstrap_results['balanced_accuracy']['upper'] * 100:.2f}]"
        )

        # Verify all models are on same test set
        if reference_subjects is None:
            reference_subjects = subjects
            reference_y_true = y_true
        else:
            if not np.array_equal(y_true, reference_y_true):
                raise ValueError(
                    f"Ground truth differs for {arch_name}. "
                    "All models must be evaluated on the same test set for McNemar."
                )
        print()

    if len(predictions_dict) < 2:
        raise ValueError("Need at least 2 architectures with data to compare")

    n_samples = len(reference_y_true)
    print(f"All models evaluated on same test set: {n_samples} subjects ✓")

    # Generate rankings
    print("\n" + "=" * 80)
    print("RANKINGS (by Balanced Accuracy)")
    print("=" * 80)
    df_rank = generate_ensemble_rankings(predictions_dict, "balanced_accuracy")
    print(df_rank.to_string())

    # Save rankings
    df_rank.to_csv(output_dir / "rankings.csv")
    print(f"\nSaved: {output_dir}/rankings.csv")

    # Perform pairwise McNemar tests
    print("\n" + "=" * 80)
    print("PAIRWISE MCNEMAR TESTS")
    print("=" * 80)
    df_tests = perform_mcnemar_tests(predictions_dict, correction=correction)
    print(df_tests.to_string(index=False))

    # Save test results
    df_tests.to_csv(output_dir / "mcnemar_tests.csv", index=False)
    print(f"\nSaved: {output_dir}/mcnemar_tests.csv")

    # Count significant differences
    n_significant = (df_tests["p_value"] < 0.05).sum()
    print(f"\nSignificant differences (raw p<0.05): {n_significant}/{len(df_tests)}")

    correction_method = df_tests["Correction"].iloc[0]
    n_corrected = (df_tests["Significant"] == "Yes").sum()
    print(
        f"Significant differences after {correction_method}: {n_corrected}/{len(df_tests)}"
    )

    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_ensemble_bar_comparison(predictions_dict, output_dir, no_title=no_title)
    plot_pairwise_matrix(df_tests, output_dir, "McNemar", no_title=no_title)

    # Generate summary report
    report_path = output_dir / "mcnemar_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MCNEMAR ARCHITECTURE COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Test Set Size: {n_samples} subjects\n")
        f.write(f"Architectures: {len(predictions_dict)}\n")
        f.write(f"Correction Method: {correction}\n\n")

        f.write("RANKINGS (by Balanced Accuracy)\n")
        f.write("-" * 80 + "\n")
        f.write(df_rank.to_string() + "\n\n")

        f.write("PAIRWISE MCNEMAR TESTS\n")
        f.write("-" * 80 + "\n")
        f.write(df_tests.to_string(index=False) + "\n\n")

        f.write("=" * 80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 80 + "\n\n")

        # Best model
        best_model = df_rank.iloc[0]["Model"]
        best_bacc = df_rank.iloc[0]["Balanced_Accuracy"]
        f.write(f"Best Model: {best_model} (BAcc={best_bacc:.2f}%)\n\n")

        # Significant wins
        df_best = df_tests[
            (df_tests["Winner"] == best_model) & (df_tests["Significant"] == "Yes")
        ]
        if len(df_best) > 0:
            f.write(f"{best_model} significantly outperforms:\n")
            for _, row in df_best.iterrows():
                other = (
                    row["Model_B"] if row["Model_A"] == best_model else row["Model_A"]
                )
                discordant = row["A_correct_B_wrong"] + row["A_wrong_B_correct"]
                f.write(
                    f"  - {other} (p={row['p_value']:.4f}, discordant pairs={discordant})\n"
                )
        else:
            f.write(
                f"{best_model} does NOT significantly outperform any other model.\n"
            )

        f.write("\n" + "=" * 80 + "\n")
        f.write("STATISTICAL NOTE\n")
        f.write("=" * 80 + "\n")
        f.write("McNemar's test is used because:\n")
        f.write("  1. All models are evaluated on the SAME test set\n")
        f.write(
            "  2. Compares paired binary outcomes (correct/incorrect per subject)\n"
        )
        f.write("  3. More powerful than Wilcoxon when test sets are identical\n")
        f.write("  4. Appropriate for OOD evaluation with ensemble predictions\n")
        f.write("\n")
        f.write("Significance level: α = 0.05\n")
        f.write(f"Number of comparisons: {len(df_tests)}\n")
        f.write(f"Correction method: {correction_method}\n")
        f.write(
            f"\n{n_corrected}/{len(df_tests)} comparisons significant after correction.\n"
        )

    print(f"Saved: {report_path}")

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(
        f"\nBest model: {df_rank.iloc[0]['Model']} (BAcc={df_rank.iloc[0]['Balanced_Accuracy']:.2f}%)"
    )
    return 0


def run_pfo_comparison(
    ensemble_csvs: List[Path],
    output_dir: Path,
    metrics: List[str] = None,
    n_bootstrap: int = 10000,
    arch_names: Optional[List[str]] = None,
    no_title: bool = False,
):
    """
    Compute Probability of False Outperformance (PFO) for multiple architectures.

    Following Christodoulou et al. (2025) "False Promises", this function computes
    the probability that observed performance rankings could reverse under resampling.

    Args:
        ensemble_csvs: List of ensemble prediction CSVs (from --save_ensemble)
        output_dir: Directory to save results
        metrics: List of metrics to compare (default: ["balanced_accuracy"])
        n_bootstrap: Number of bootstrap iterations
        arch_names: Optional custom names for architectures
        no_title: If True, omit plot titles (for publication)
    """
    if metrics is None:
        metrics = ["balanced_accuracy"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use filenames if no custom names
    if arch_names is None:
        arch_names = [p.stem for p in ensemble_csvs]

    if len(arch_names) != len(ensemble_csvs):
        raise ValueError(
            f"Number of names ({len(arch_names)}) must match number of CSVs ({len(ensemble_csvs)})"
        )

    # Load all predictions (only once)
    print("Loading predictions...")
    predictions = {}
    for path, name in zip(ensemble_csvs, arch_names):
        df = pd.read_csv(path)
        pred_cols = [c for c in df.columns if c.startswith("pred_")]
        if not pred_cols:
            raise ValueError(f"No prediction columns (pred_*) found in {path}")

        class_names = [c.replace("pred_", "") for c in pred_cols]
        y_true_str = df["Diagnosis"].values
        y_pred_probs = df[pred_cols].values

        # Convert string labels to integer indices
        class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        y_true = np.array([class_to_idx.get(label, -1) for label in y_true_str])

        # Filter unknown classes
        valid_mask = y_true >= 0
        predictions[name] = {
            "y_true": y_true[valid_mask],
            "y_probs": y_pred_probs[valid_mask],
            "classes": class_names,
        }

    # Compute bootstrap samples for all models (cache for all metrics)
    print(
        f"Computing bootstrap samples ({n_bootstrap} iterations) for {len(arch_names)} models..."
    )
    all_bootstrap_results = {}
    for name in tqdm(arch_names, desc="Bootstrap"):
        data = predictions[name]
        results = compute_bootstrap_metrics(
            y_true=data["y_true"],
            y_pred_probs=data["y_probs"],
            n_bootstrap=n_bootstrap,
            confidence=0.95,
            random_state=42,
            n_jobs=-1,
        )
        all_bootstrap_results[name] = results

    # Process each metric
    all_pfo_dfs = []
    for metric in metrics:
        # Normalize metric name
        metric_key = METRIC_MAP.get(metric, metric)
        lower_is_better = metric_key in LOWER_IS_BETTER_METRICS

        print(f"\n{'=' * 80}")
        print(f"PFO ANALYSIS: {metric_key.upper()}")
        print("=" * 80)

        # Get bootstrap samples for this metric
        bootstrap_samples = {
            name: all_bootstrap_results[name][metric_key]["samples"]
            for name in arch_names
        }

        # Compute pairwise PFO
        print("Computing pairwise PFO...")
        results = []
        for name_A, name_B in combinations(arch_names, 2):
            pfo = compute_pfo(
                bootstrap_samples[name_A],
                bootstrap_samples[name_B],
                lower_is_better,
            )

            # Determine apparent winner based on observed difference
            if pfo["delta_observed"] > 0:
                apparent_winner = name_A
                pfo_winner = pfo["pfo_A_over_B"]
            else:
                apparent_winner = name_B
                pfo_winner = pfo["pfo_B_over_A"]

            # Reliability assessment
            if pfo_winner < 0.05:
                reliability = "Reliable"
            elif pfo_winner < 0.25:
                reliability = "Moderate"
            else:
                reliability = "Unreliable"

            results.append(
                {
                    "Metric": metric_key,
                    "Model_A": name_A,
                    "Model_B": name_B,
                    "Mean_A": pfo["mean_A"] * 100,
                    "Mean_B": pfo["mean_B"] * 100,
                    "Delta_%": pfo["delta_observed"] * 100,
                    "Delta_CI95": f"[{pfo['delta_ci95'][0] * 100:.2f}, {pfo['delta_ci95'][1] * 100:.2f}]",
                    "PFO_A>B": pfo["pfo_A_over_B"],
                    "PFO_B>A": pfo["pfo_B_over_A"],
                    "Apparent_Winner": apparent_winner,
                    "PFO_Winner": pfo_winner,
                    "Reliability": reliability,
                }
            )

        df_pfo = pd.DataFrame(results)
        all_pfo_dfs.append(df_pfo)

        # Create metric-specific subdirectory if multiple metrics
        if len(metrics) > 1:
            metric_dir = output_dir / metric_key
            metric_dir.mkdir(parents=True, exist_ok=True)
        else:
            metric_dir = output_dir

        # Save results
        df_pfo.to_csv(metric_dir / f"pfo_results_{metric_key}.csv", index=False)
        print(f"Saved: {metric_dir / f'pfo_results_{metric_key}.csv'}")

        # Create heatmap
        n_models = len(arch_names)
        pfo_matrix = np.zeros((n_models, n_models))
        np.fill_diagonal(pfo_matrix, np.nan)

        model_to_idx = {m: i for i, m in enumerate(arch_names)}

        for _, row in df_pfo.iterrows():
            i = model_to_idx[row["Model_A"]]
            j = model_to_idx[row["Model_B"]]
            pfo_matrix[i, j] = row["PFO_A>B"]
            pfo_matrix[j, i] = row["PFO_B>A"]

        # Smaller figure = larger apparent text (text size stays constant)
        fig, ax = plt.subplots(figsize=(max(5, n_models * 0.7), max(4, n_models * 0.6)))
        mask = np.eye(n_models, dtype=bool)

        # Custom colormap
        cmap = sns.diverging_palette(145, 10, as_cmap=True)

        sns.heatmap(
            pfo_matrix,
            mask=mask,
            annot=True,
            fmt=".3f",
            cmap=cmap,
            center=0.05,
            vmin=0,
            vmax=0.5,
            xticklabels=arch_names,
            yticklabels=arch_names,
            ax=ax,
            cbar_kws={"label": "P(row ≤ column)"},
            linewidths=0.5,
        )
        if not no_title:
            ax.set_title(
                f"Probability of False Outperformance ({metric_key})\n(Values < 0.05 = reliable superiority)"
            )
        ax.set_xlabel("Reference Model")
        ax.set_ylabel("Test Model")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            metric_dir / f"pfo_heatmap_{metric_key}.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(metric_dir / f"pfo_heatmap_{metric_key}.pdf", bbox_inches="tight")
        plt.close()
        print(f"Saved: {metric_dir / f'pfo_heatmap_{metric_key}.png'}")

        # Generate report
        report_lines = [
            "=" * 80,
            "PROBABILITY OF FALSE OUTPERFORMANCE (PFO) ANALYSIS",
            f"Metric: {metric_key}",
            "=" * 80,
            "",
            "Reference: Christodoulou et al. (2025) 'False Promises'",
            "",
            "Interpretation:",
            "  - PFO < 0.05: Reliable outperformance (ranking unlikely to reverse)",
            "  - PFO ∈ [0.05, 0.25]: Moderate reliability (some uncertainty)",
            "  - PFO > 0.25: Unreliable ranking (high chance of reversal)",
            "",
            "-" * 80,
        ]

        df_sorted = df_pfo.sort_values("Delta_%", key=abs, ascending=False)
        for _, row in df_sorted.iterrows():
            symbol = (
                "✓"
                if row["Reliability"] == "Reliable"
                else ("⚠" if row["Reliability"] == "Moderate" else "✗")
            )
            report_lines.extend(
                [
                    f"\n{row['Apparent_Winner']} vs {row['Model_B'] if row['Apparent_Winner'] == row['Model_A'] else row['Model_A']}:",
                    f"  Observed Δ: {abs(row['Delta_%']):.2f}%",
                    f"  PFO: {row['PFO_Winner']:.4f} ({symbol} {row['Reliability']})",
                    f"  95% CI of Δ: {row['Delta_CI95']}",
                ]
            )

        report_lines.append("\n" + "=" * 80)
        report = "\n".join(report_lines)

        print(report)
        with open(metric_dir / f"pfo_report_{metric_key}.txt", "w") as f:
            f.write(report)
        print(f"Saved: {metric_dir / f'pfo_report_{metric_key}.txt'}")

        # Print summary table
        print("\n")
        print(
            tabulate(
                df_pfo[
                    ["Model_A", "Model_B", "Delta_%", "PFO_Winner", "Reliability"]
                ].values.tolist(),
                headers=["Model A", "Model B", "Δ (%)", "PFO", "Reliability"],
                tablefmt="grid",
                floatfmt=(".2f", ".2f", ".2f", ".4f", "s"),
            )
        )

    # If multiple metrics, save combined results
    if len(metrics) > 1:
        df_combined = pd.concat(all_pfo_dfs, ignore_index=True)
        df_combined.to_csv(output_dir / "pfo_results_all_metrics.csv", index=False)
        print(f"\nSaved combined results: {output_dir / 'pfo_results_all_metrics.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description="Statistical comparison of model predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tests:
  mcnemar               Compare two models on the same test set (McNemar's test)
  wilcoxon              Compare two models across folds (Wilcoxon signed-rank test)
  multi-compare-wilcoxon Compare multiple architectures using Wilcoxon (per-fold scores)
  multi-compare-mcnemar  Compare multiple architectures using McNemar (ensemble predictions)

Examples:
  # McNemar test (2 models)
  python compare_architectures.py mcnemar modelA.csv modelB.csv --output results/mcnemar
  
  # Wilcoxon test (2 models)
  python compare_architectures.py wilcoxon /path/A /path/B --pattern "*.csv" --output results/wilcoxon
  
  # Multi-architecture comparison with Wilcoxon (from fold directories)
  python compare_architectures.py multi-compare-wilcoxon /path/arch1 /path/arch2 /path/arch3 \\
      --output results/comparison --names "Arch1" "Arch2" "Arch3" --metric bacc
  
  # Multi-architecture comparison with McNemar (from ensemble CSVs)
  python compare_architectures.py multi-compare-mcnemar \\
      ensemble_arch1_od.csv ensemble_arch2_od.csv ensemble_arch3_od.csv \\
      --output results/mcnemar_comparison --names "Arch1" "Arch2" "Arch3"
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Test to perform")

    # McNemar test
    mcnemar_parser = subparsers.add_parser(
        "mcnemar", help="McNemar test for two models on same test set"
    )
    mcnemar_parser.add_argument("csv_A", type=Path, help="Predictions CSV for model A")
    mcnemar_parser.add_argument("csv_B", type=Path, help="Predictions CSV for model B")
    mcnemar_parser.add_argument(
        "--output", type=Path, default=Path("results/mcnemar"), help="Output directory"
    )
    mcnemar_parser.add_argument(
        "--name-A", type=str, help="Name for model A (default: filename)"
    )
    mcnemar_parser.add_argument(
        "--name-B", type=str, help="Name for model B (default: filename)"
    )
    mcnemar_parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit plot titles (for publication)",
    )

    # Wilcoxon test
    wilcoxon_parser = subparsers.add_parser(
        "wilcoxon", help="Wilcoxon test for two models across folds"
    )
    wilcoxon_parser.add_argument(
        "dir_A", type=Path, help="Directory with predictions for model A"
    )
    wilcoxon_parser.add_argument(
        "dir_B", type=Path, help="Directory with predictions for model B"
    )
    wilcoxon_parser.add_argument(
        "--pattern",
        type=str,
        default="*_best0_od.csv",
        help="Glob pattern to match fold files",
    )
    wilcoxon_parser.add_argument(
        "--metric",
        type=str,
        default="balanced_accuracy",
        choices=METRIC_CHOICES,
        help="Metric to compare (default: balanced_accuracy)",
    )
    wilcoxon_parser.add_argument(
        "--output", type=Path, default=Path("results/wilcoxon"), help="Output directory"
    )
    wilcoxon_parser.add_argument(
        "--name-A", type=str, help="Name for model A (default: dirname)"
    )
    wilcoxon_parser.add_argument(
        "--name-B", type=str, help="Name for model B (default: dirname)"
    )
    wilcoxon_parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit plot titles (for publication)",
    )

    # Multi-architecture comparison with Wilcoxon
    multi_parser = subparsers.add_parser(
        "multi-compare-wilcoxon",
        help="Compare multiple architectures using Wilcoxon (per-fold scores)",
    )
    multi_parser.add_argument(
        "model_dirs",
        nargs="+",
        type=Path,
        help="Directories containing predictions for each architecture",
    )
    multi_parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output directory for results"
    )
    multi_parser.add_argument(
        "--metric",
        nargs="+",
        default=["balanced_accuracy"],
        choices=METRIC_CHOICES,
        help="Metric(s) to compare (default: balanced_accuracy)",
    )
    multi_parser.add_argument(
        "--pattern",
        default="*_best0_od.csv",
        help="Glob pattern for prediction files (default: *_best0_od.csv)",
    )
    multi_parser.add_argument(
        "--names",
        nargs="+",
        help="Custom names for architectures (default: directory names)",
    )
    multi_parser.add_argument(
        "--correction",
        choices=["bonferroni", "fdr_bh", "none"],
        default="bonferroni",
        help="Multiple comparison correction method (default: bonferroni)",
    )
    multi_parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit plot titles (for publication)",
    )

    # Multi-architecture comparison with McNemar (ensemble predictions)
    multi_mcnemar_parser = subparsers.add_parser(
        "multi-compare-mcnemar",
        help="Compare multiple architectures using McNemar on ensemble predictions",
    )
    multi_mcnemar_parser.add_argument(
        "ensemble_csvs",
        nargs="+",
        type=Path,
        help="Ensemble prediction CSVs for each architecture (from --save_ensemble)",
    )
    multi_mcnemar_parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output directory for results"
    )
    multi_mcnemar_parser.add_argument(
        "--names",
        nargs="+",
        help="Custom names for architectures (default: CSV filenames)",
    )
    multi_mcnemar_parser.add_argument(
        "--correction",
        choices=["bonferroni", "fdr_bh", "none"],
        default="bonferroni",
        help="Multiple comparison correction method (default: bonferroni)",
    )
    multi_mcnemar_parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit plot titles (for publication)",
    )

    # Probability of False Outperformance (PFO) analysis
    pfo_parser = subparsers.add_parser(
        "pfo",
        help="Compute Probability of False Outperformance (Christodoulou 2025)",
    )
    pfo_parser.add_argument(
        "ensemble_csvs",
        nargs="+",
        type=Path,
        help="Ensemble prediction CSVs for each architecture (from --save_ensemble)",
    )
    pfo_parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output directory for results"
    )
    pfo_parser.add_argument(
        "--metric",
        "-m",
        nargs="+",
        default=["balanced_accuracy"],
        choices=METRIC_CHOICES,
        help="Metric(s) to compare (default: balanced_accuracy)",
    )
    pfo_parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap iterations (default: 10000)",
    )
    pfo_parser.add_argument(
        "--names",
        nargs="+",
        help="Custom names for architectures (default: CSV filenames)",
    )
    pfo_parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit plot titles (for publication)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "mcnemar":
        run_mcnemar_comparison(
            args.csv_A,
            args.csv_B,
            args.output,
            args.name_A,
            args.name_B,
            no_title=args.no_title,
        )
    elif args.command == "wilcoxon":
        run_wilcoxon_comparison(
            args.dir_A,
            args.dir_B,
            args.pattern,
            args.output,
            args.name_A,
            args.name_B,
            args.metric,
            no_title=args.no_title,
        )
    elif args.command == "multi-compare-wilcoxon":
        run_architecture_comparison(
            args.model_dirs,
            args.pattern,
            args.metric,
            args.output,
            args.names,
            args.correction,
            no_title=args.no_title,
        )
    elif args.command == "multi-compare-mcnemar":
        run_mcnemar_architecture_comparison(
            args.ensemble_csvs,
            args.output,
            args.names,
            args.correction,
            no_title=args.no_title,
        )
    elif args.command == "pfo":
        run_pfo_comparison(
            args.ensemble_csvs,
            args.output,
            args.metric,  # Now a list of metrics
            args.n_bootstrap,
            args.names,
            no_title=args.no_title,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

# =============================================================================
# USAGE EXAMPLES
# =============================================================================
#
# 1. McNemar test (comparing two models on same test set):
# python visualizations/results/compare_architectures.py mcnemar \\
#     predictions_model_a.csv predictions_model_b.csv \\
#     --output visualizations/outputs/mcnemar --name-A "Model A" --name-B "Model B"
#
# 2. Wilcoxon signed-rank test (comparing fold-level metrics):
# python visualizations/results/compare_architectures.py wilcoxon \\
#     /path/to/model_a/ /path/to/model_b/ \\
#     --pattern "*_id.csv" --output visualizations/outputs/wilcoxon
#
# 3. Multi-compare Wilcoxon (multiple models, Bonferroni correction):
# python visualizations/results/compare_architectures.py multi-compare-wilcoxon \\
#     /path/to/model1/ /path/to/model2/ /path/to/model3/ \\
#     --pattern "*_id.csv" --correction bonferroni \\
#     --metric accuracy mcc --output visualizations/outputs/comparison
#
# 4. Probability of First Order (PFO) analysis:
# python visualizations/results/compare_architectures.py pfo \\
#     predictions_1.csv predictions_2.csv predictions_3.csv \\
#     --metric balanced_accuracy --n-bootstrap 10000 \\
#     --output visualizations/outputs/pfo --names "Baseline" "Model A" "Model B"
