"""Utilities for confidence interval analysis and stratified sampling."""

import logging
import os
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S"
)

METRICS_CONFIG = {
    "accuracy": {"color": "#E64B35", "label": "Accuracy", "marker": "o"},
    "mcc": {"color": "#4DBBD5", "label": "MCC", "marker": "s"},
    "pr_auc": {"color": "#00A087", "label": "PR AUC", "marker": "^"},
    "macro_f1": {"color": "#8491B4", "label": "Macro F1", "marker": "D"},
}

CLASS_SAMPLE_SIZES = {
    "ID": {"CN": 1412, "AD": 654, "BV": 229, "PNFA": 66, "SD": 76, "FTD": 371},
    "OD": {"CN": 2251, "AD": 485, "BV": 100, "PNFA": 43, "SD": 43, "FTD": 186},
}

PLOT_PARAMS = {
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 1,
    "lines.linewidth": 2,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}

AGE_BINS = [-np.inf, 50, 60, 70, 80, 90, 100, np.inf]
AGE_LABELS = ["<50", "50-60", "60-70", "70-80", "80-90", "90-100", ">100"]


@contextmanager
def suppress_output():
    """Redirect stdout/stderr to devnull for silent workers."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def load_ground_truth(od_path, id_dir):
    """Load OD and ID ground truth CSVs."""
    logging.info("Loading Ground Truths...")
    od_gt = pd.read_csv(od_path)
    id_files = sorted(list(id_dir.glob("fold_*.csv")))
    if not id_files:
        raise FileNotFoundError(f"No fold_*.csv files found in {id_dir}")
    id_gt = pd.concat([pd.read_csv(f) for f in id_files], ignore_index=True)
    return od_gt, id_gt


def load_and_preprocess_single(path, gt_df, domain):
    """Load, merge and clean a single prediction file."""
    df = pd.read_csv(path)

    cols = ["Subject", "Sex", "Dataset", "Diagnosis", "Age"]
    df = df.merge(gt_df[cols], on="Subject", how="left", suffixes=("", "_y"))
    df = df[[c for c in df.columns if not c.endswith("_y")]]
    df = df.drop_duplicates(subset=["Subject"], keep="first")

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].str.lower()

    regex = r"NACC.*" if domain == "OD" else r"ADNI.*"
    repl = "NACC" if domain == "OD" else "ADNI"
    if "Dataset" in df.columns:
        df["Dataset"] = df["Dataset"].replace(to_replace=regex, value=repl, regex=True)

    df["age_bin"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)

    return df


def generate_stratified_subsets(df, fractions, seed):
    """Generate stratified subsets for a given dataframe."""
    strat_cols = ["Diagnosis", "Sex", "age_bin", "Dataset"]
    subsets = {}
    for frac in fractions:
        name = f"size_{int(frac * 100)}"
        sub_df = df.groupby(strat_cols, observed=True).sample(
            frac=frac, replace=False, random_state=seed
        )
        subsets[name] = sub_df
    return subsets


def compute_ncv(values, sample_size):
    """Compute normalized coefficient of variation."""
    if not values:
        return np.nan
    mean = np.mean(values)
    if mean == 0:
        return np.nan
    std = np.std(values)
    return (std / mean) * (sample_size**0.5)


def compute_metrics_task(
    id_df, od_df, bootstrap_n, calculate_metrics_fn, domain_id, domain_od
):
    """Task executed by worker to compute metrics."""
    if id_df.empty or od_df.empty:
        return None
    with suppress_output():
        return calculate_metrics_fn(
            {domain_id: id_df, domain_od: od_df}, "standard", bootstrap_n, top_ns=None
        )


def compute_single_domain_metrics(df, bootstrap_n, calculate_metrics_fn, domain):
    """Compute metrics for a single domain."""
    if df.empty:
        return None, 0
    with suppress_output():
        res = calculate_metrics_fn({domain: df}, "standard", bootstrap_n, top_ns=None)
    return res.get(domain), len(df)


def process_prediction_file(file_path, gt_df, domain):
    """Load and preprocess a single prediction file (checks existence)."""
    if not file_path.exists():
        return pd.DataFrame()
    return load_and_preprocess_single(file_path, gt_df, domain)
