"""Analyze ensemble performance and metrics stability vs number of models."""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from compute_metrics_plot_violin_csv import DOMAIN_ID, DOMAIN_OD, calculate_metrics
from utils_ci import (
    CLASS_SAMPLE_SIZES,
    METRICS_CONFIG,
    PLOT_PARAMS,
    compute_ncv,
    load_ground_truth,
    process_prediction_file,
    suppress_output,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Analyze Ensemble Performance & Metrics Stability."
    )
    parser.add_argument("--od_gt", type=Path, required=True, help="Path to OD GT CSV")
    parser.add_argument(
        "--id_gt_dir", type=Path, required=True, help="Dir containing ID folds"
    )
    parser.add_argument("--results_dir", type=Path, required=True, help="Results dir")
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--n_models", type=int, default=16)
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument(
        "--max_workers", type=int, default=4, help="Max parallel processes."
    )
    parser.add_argument("--output_dir", type=Path, default=Path("."))
    parser.add_argument("--output_fmt", type=str, default="pdf", choices=["pdf", "png"])
    parser.add_argument("--title", type=str, default=None)
    return parser.parse_args()


def parallel_metric_computation(id_df, od_df, bootstrap_n):
    if id_df.empty or od_df.empty:
        return None
    with suppress_output():
        return calculate_metrics(
            {DOMAIN_ID: id_df, DOMAIN_OD: od_df}, "standard", bootstrap_n, top_ns=None
        )


def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PLOT_PARAMS)

    logging.info(f"Running with {args.max_workers} max parallel workers.")

    od_gt, id_gt = load_ground_truth(args.od_gt, args.id_gt_dir)

    logging.info("Loading predictions...")
    loaded_data = [
        [{"ID": None, "OD": None} for _ in range(args.n_models)]
        for _ in range(args.n_seeds)
    ]

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_meta = {}
        for j in range(1, args.n_seeds + 1):
            for i in range(1, args.n_models + 1):
                base_name = f"ensemble_n{i}_folds10_swin-5c-no_seed-dataaug-ema-label_smoothing-balanced_sampling-mixup-{j}"

                f_od = executor.submit(
                    process_prediction_file,
                    args.results_dir / f"{base_name}_od.csv",
                    od_gt,
                    "OD",
                )
                future_to_meta[f_od] = (j - 1, i - 1, "OD")

                f_id = executor.submit(
                    process_prediction_file,
                    args.results_dir / f"{base_name}_id.csv",
                    id_gt,
                    "ID",
                )
                future_to_meta[f_id] = (j - 1, i - 1, "ID")

        for future in tqdm(
            as_completed(future_to_meta),
            total=len(future_to_meta),
            desc="Loading CSVs",
            unit="file",
        ):
            seed_idx, model_idx, domain = future_to_meta[future]
            df = future.result()
            loaded_data[seed_idx][model_idx][domain] = df

    logging.info(f"Computing Bootstrap Metrics (n={args.bootstrap_n})...")
    models_results = [[{} for _ in range(args.n_models)] for _ in range(args.n_seeds)]

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_idx = {}
        tasks_count = 0
        for j in range(args.n_seeds):
            for i in range(args.n_models):
                id_df = loaded_data[j][i]["ID"]
                od_df = loaded_data[j][i]["OD"]

                if (
                    id_df is not None
                    and not id_df.empty
                    and od_df is not None
                    and not od_df.empty
                ):
                    f = executor.submit(
                        parallel_metric_computation, id_df, od_df, args.bootstrap_n
                    )
                    future_to_idx[f] = (j, i)
                    tasks_count += 1

        for future in tqdm(
            as_completed(future_to_idx),
            total=tasks_count,
            desc="Bootstrapping",
            unit="ens",
        ):
            j, i = future_to_idx[future]
            models_results[j][i] = future.result()

    logging.info("Aggregating statistics...")
    metrics_list = ["accuracy", "mcc", "pr_auc", "macro_f1"]
    final_stats = {
        domain: {k: {} for k in range(1, args.n_models + 1)}
        for domain in [DOMAIN_ID, DOMAIN_OD]
    }

    n_samples_id = sum([CLASS_SAMPLE_SIZES["ID"][d] for d in ["CN", "AD", "FTD"]])
    n_samples_od = sum([CLASS_SAMPLE_SIZES["OD"][d] for d in ["CN", "AD", "FTD"]])

    for k in range(1, args.n_models + 1):
        for metric in metrics_list:
            vals_id, vals_od = [], []
            for j in range(args.n_seeds):
                res = models_results[j][k - 1]
                if res:
                    vals_id.append(res[DOMAIN_ID][metric]["mean"])
                    vals_od.append(res[DOMAIN_OD][metric]["mean"])

            final_stats[DOMAIN_ID][k][metric] = compute_ncv(vals_id, n_samples_id)
            final_stats[DOMAIN_OD][k][metric] = compute_ncv(vals_od, n_samples_od)

    logging.info("Generating plot...")
    x_models = np.array(range(1, args.n_models + 1))
    fig, (ax_id, ax_od) = plt.subplots(1, 2, sharey=True, figsize=(14, 6))

    for metric, cfg in METRICS_CONFIG.items():
        y_id = [final_stats[DOMAIN_ID][k][metric] for k in x_models]
        ax_id.plot(
            x_models,
            y_id,
            color=cfg["color"],
            marker=cfg["marker"],
            label=cfg["label"],
            markersize=6,
            linewidth=1.5,
            alpha=0.9,
        )

        y_od = [final_stats[DOMAIN_OD][k][metric] for k in x_models]
        ax_od.plot(
            x_models,
            y_od,
            color=cfg["color"],
            marker=cfg["marker"],
            label=cfg["label"],
            markersize=6,
            linewidth=1.5,
            alpha=0.9,
        )

    ax_id.set_xlabel(r"Number of Models in Ensemble [ID]", fontsize=12)
    ax_od.set_xlabel(r"Number of Models in Ensemble [OOD]", fontsize=12)
    ax_id.set_ylabel("Normalized CV (Lower = More Stable)", fontsize=12)

    if args.title:
        fig.suptitle(args.title, fontsize=14, fontweight="bold")

    for ax in (ax_id, ax_od):
        ax.grid(axis="y", linestyle=":", color="gray", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    handles, labels = ax_id.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=len(METRICS_CONFIG),
        fancybox=False,
        edgecolor="black",
        framealpha=1,
        fontsize=11,
    )

    out_file = args.output_dir / f"metric_cv_vs_ensemble_size.{args.output_fmt}"
    plt.savefig(out_file, dpi=300, pad_inches=0.01, bbox_inches="tight")
    logging.info(f"Plot saved to {out_file}")


if __name__ == "__main__":
    main()

# Example usage:
# python visualizations/results/ci_vs_number_of_models.py \
#     --od_gt "/path/to/test.csv" \
#     --id_gt_dir "/path/to/10fold_CV/" \
#     --results_dir "visualizations/outputs/metrics_results/ensemble_predictions" \
#     --output_dir "visualizations/outputs/" \
#     --output_fmt "pdf" \
#     --n_models 16 \
#     --n_seeds 3 \
#     --max_workers 4
