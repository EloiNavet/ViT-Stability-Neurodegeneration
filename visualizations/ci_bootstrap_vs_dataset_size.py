"""Analyze relationship between dataset size and bootstrap CI width."""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from tqdm import tqdm

from compute_metrics_plot_violin_csv import DOMAIN_OD, calculate_metrics
from utils_ci import (
    AGE_BINS,
    AGE_LABELS,
    METRICS_CONFIG,
    PLOT_PARAMS,
    generate_stratified_subsets,
    suppress_output,
)


def theoretical_decay(n, C):
    """Theoretical function: C / sqrt(n)"""
    return C / np.sqrt(n)


def get_args():
    parser = argparse.ArgumentParser(
        description="Analyze Dataset Size vs CI Width (Scaling Law)."
    )
    parser.add_argument(
        "--od_gt", type=Path, required=True, help="Path to OD Ground Truth CSV"
    )
    parser.add_argument(
        "--pred_csv", type=Path, required=True, help="Path to Model Predictions CSV"
    )
    parser.add_argument(
        "--n_fractions", type=int, default=15, help="Number of data splits"
    )
    parser.add_argument(
        "--min_frac", type=float, default=0.1, help="Min fraction of dataset"
    )
    parser.add_argument(
        "--bootstrap_n", type=int, default=10000, help="Bootstrap iterations"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_workers", type=int, default=8, help="Max parallel processes"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("."), help="Output directory"
    )
    parser.add_argument(
        "--output_fmt", type=str, default="pdf", choices=["pdf", "png", "svg"]
    )
    return parser.parse_args()


def prepare_data(gt_path, pred_path):
    """Load, merge, clean and prepare bins."""
    logging.info("Loading and preparing data...")
    gt = pd.read_csv(gt_path)
    df = pd.read_csv(pred_path)

    cols = ["Subject", "Sex", "Dataset", "Diagnosis", "Age"]
    df = df.merge(gt[cols], on="Subject", how="left", suffixes=("", "_y"))
    df = df[[c for c in df.columns if not c.endswith("_y")]]
    df = df.drop_duplicates(subset=["Subject"], keep="first")

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].str.lower()
    if "Dataset" in df.columns:
        df["Dataset"] = df["Dataset"].replace(
            to_replace=r"NACC.*", value="NACC", regex=True
        )

    df["age_bin"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)

    logging.info(f"Data ready. Total samples: {len(df)}")
    return df


def compute_subset_metrics(subset_df, bootstrap_n):
    """Wrapper executed by workers."""
    if subset_df.empty:
        return None, 0
    with suppress_output():
        res = calculate_metrics(
            {DOMAIN_OD: subset_df}, "standard", bootstrap_n, top_ns=None
        )
    return res.get(DOMAIN_OD), len(subset_df)


def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PLOT_PARAMS)

    df_main = prepare_data(args.od_gt, args.pred_csv)

    fractions = np.linspace(args.min_frac, 1.0, args.n_fractions)
    subsets_dict = generate_stratified_subsets(df_main, fractions, args.seed)

    results_map = {}

    logging.info(
        f"Starting bootstrap on {len(subsets_dict)} subsets with {args.max_workers} workers..."
    )

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_size = {}
        for name, sub_df in subsets_dict.items():
            f = executor.submit(compute_subset_metrics, sub_df, args.bootstrap_n)
            future_to_size[f] = len(sub_df)

        for future in tqdm(
            as_completed(future_to_size),
            total=len(subsets_dict),
            desc="Bootstrapping",
            unit="subset",
        ):
            metrics_res, real_size = future.result()
            if metrics_res:
                results_map[real_size] = metrics_res

    logging.info("Fitting curves and plotting...")

    sorted_sizes = sorted(results_map.keys())
    x = np.array(sorted_sizes)

    plt.figure(figsize=(8, 4))

    for metric, cfg in METRICS_CONFIG.items():
        y_margins = []
        for sz in x:
            m_data = results_map[sz][metric]
            width = m_data["upper"] - m_data["lower"]
            y_margins.append((width / 2) * 100)

        y = np.array(y_margins)

        popt, _ = curve_fit(theoretical_decay, x, y)
        y_pred = theoretical_decay(x, *popt)
        r2 = r2_score(y, y_pred)

        x_smooth = np.linspace(x.min(), x.max(), 200)
        y_smooth = theoretical_decay(x_smooth, *popt)

        plt.plot(
            x, y, marker=cfg["marker"], linestyle="none", color=cfg["color"], alpha=0.7
        )
        plt.plot(
            x_smooth,
            y_smooth,
            linestyle="-",
            linewidth=1.5,
            color=cfg["color"],
            alpha=0.8,
        )

        legend_label = rf"{cfg['label']} ($R^2={r2:.2f}$)"
        plt.plot(
            [],
            [],
            color=cfg["color"],
            marker=cfg["marker"],
            linestyle="-",
            linewidth=1.5,
            label=legend_label,
        )

        logging.info(f"{metric}: C={popt[0]:.2f}, R2={r2:.4f}")

    plt.xlabel("Dataset Size")
    plt.ylabel("95% CI Half-Width (%)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    outfile = args.output_dir / f"stability_scaling_analysis.{args.output_fmt}"
    plt.savefig(outfile, dpi=300, bbox_inches="tight", pad_inches=0.01)
    logging.info(f"Saved plot to {outfile}")


if __name__ == "__main__":
    main()

# Example usage:
# python visualizations/results/ci_bootstrap_vs_dataset_size.py --compute \
#     --od_gt "/path/to/test.csv" \
#     --id_gt_dir "/path/to/10fold_CV/" \
#     --results_dir "visualizations/outputs/metrics_results/ensemble_predictions" \
#     --output_dir "visualizations/outputs/" \
#     --json_file "ci_bootstrap_vs_dataset_size_results.json" \
#     --n_seeds 5 \
#     --max_workers 16
#
# python visualizations/results/ci_bootstrap_vs_dataset_size.py --plot \
#     --output_dir "visualizations/outputs/" \
#     --json_file "ci_bootstrap_vs_dataset_size_results.json"
