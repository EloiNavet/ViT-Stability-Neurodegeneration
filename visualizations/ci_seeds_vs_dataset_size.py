"""TTA vs No-TTA comparison analysis: compute metrics and generate plots."""

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from tqdm import tqdm

from compute_metrics_plot_violin_csv import DOMAIN_ID, DOMAIN_OD, calculate_metrics
from utils_ci import (
    CLASS_SAMPLE_SIZES,
    METRICS_CONFIG,
    PLOT_PARAMS,
    load_and_preprocess_single,
    suppress_output,
    generate_stratified_subsets,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S"
)

# Style configuration for TTA vs No-TTA comparison
TTA_STYLE = {
    "linestyle": "-",
    "alpha": 0.9,
    "linewidth": 2.0,
    "markersize": 6,
    "label": "TTA",
}

NO_TTA_STYLE = {
    "linestyle": "--",
    "alpha": 0.6,
    "linewidth": 1.5,
    "markersize": 5,
    "label": "No-TTA",
}


def get_args():
    parser = argparse.ArgumentParser(
        description="TTA vs No-TTA comparison: compute metrics or plot results."
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--compute", action="store_true", help="Compute metrics and save to JSON"
    )
    mode_group.add_argument(
        "--plot", action="store_true", help="Plot results from JSON file"
    )

    # === COMPUTE mode arguments ===
    compute_group = parser.add_argument_group("Compute options")
    compute_group.add_argument("--od_gt", type=Path, help="Path to OD Ground Truth CSV")
    compute_group.add_argument("--id_gt_dir", type=Path, help="Dir containing ID folds")
    compute_group.add_argument(
        "--results_dir", type=Path, help="Dir containing predictions"
    )
    compute_group.add_argument(
        "--tta_pattern",
        type=str,
        default="ensemble_n1_folds10_tta-{i}",
        help="Pattern for TTA files (use {i} for seed index)",
    )
    compute_group.add_argument(
        "--no_tta_pattern",
        type=str,
        default="ensemble_n1_folds10_swin-5c-no_seed-dataaug-ema-label_smoothing-balanced_sampling-mixup-{i}",
        help="Pattern for No-TTA files (use {i} for seed index)",
    )
    compute_group.add_argument("--n_seeds", type=int, default=5, help="Number of seeds")
    compute_group.add_argument(
        "--n_fractions", type=int, default=15, help="Number of subsets"
    )
    compute_group.add_argument(
        "--min_frac", type=float, default=0.3, help="Min fraction"
    )
    compute_group.add_argument(
        "--bootstrap_n", type=int, default=10000, help="Bootstrap iterations"
    )
    compute_group.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling"
    )
    compute_group.add_argument(
        "--max_workers", type=int, default=16, help="Parallel processes"
    )

    # === PLOT mode arguments ===
    plot_group = parser.add_argument_group("Plot options")
    plot_group.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[14, 6],
        help="Figure size (w, h)",
    )
    plot_group.add_argument("--dpi", type=int, default=300, help="Output DPI")
    plot_group.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Metrics to plot (default: all)",
    )
    plot_group.add_argument(
        "--y_label",
        type=str,
        default="Normalized CV (Lower = More Stable)",
        help="Y-axis label",
    )

    # === Shared arguments ===
    parser.add_argument(
        "--json_file",
        type=Path,
        default=Path("ci_seeds_vs_dataset_size_results.json"),
        help="JSON file path (output for --compute, input for --plot)",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("."), help="Output directory"
    )
    parser.add_argument(
        "--output_fmt",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output format for plot",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="metric_cv_vs_data_availability_tta_comparison",
        help="Output filename for plot (without extension)",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        dest="no_title",
        help="Disable the plot title",
    )

    args = parser.parse_args()

    if args.compute:
        if not args.od_gt or not args.id_gt_dir or not args.results_dir:
            parser.error("--compute requires --od_gt, --id_gt_dir, and --results_dir")

    return args


def load_predictions(results_dir, pattern, n_seeds, id_gt, od_gt):
    """Load prediction files for a given pattern (TTA or No-TTA)."""
    id_dfs = []
    od_dfs = []

    for i in range(1, n_seeds + 1):
        file_prefix = pattern.format(i=i)
        p_id = results_dir / f"{file_prefix}_id.csv"
        p_od = results_dir / f"{file_prefix}_od.csv"

        if not p_id.exists() or not p_od.exists():
            logging.warning(f"Missing files for seed {i}: {p_id} or {p_od}")
            continue

        id_dfs.append(load_and_preprocess_single(p_id, id_gt, "ID"))
        od_dfs.append(load_and_preprocess_single(p_od, od_gt, "OD"))

    return id_dfs, od_dfs


def compute_metrics_task(id_df, od_df, bootstrap_n):
    """Compute metrics for a single subset."""
    if id_df.empty or od_df.empty:
        return None
    with suppress_output():
        return calculate_metrics(
            {DOMAIN_ID: id_df, DOMAIN_OD: od_df}, "standard", bootstrap_n, top_ns=None
        )


def process_predictions(id_dfs, od_dfs, fractions, seed, bootstrap_n, max_workers):
    """Process predictions and compute bootstrap metrics."""
    subsets_id = [generate_stratified_subsets(df, fractions, seed) for df in id_dfs]
    subsets_od = [generate_stratified_subsets(df, fractions, seed) for df in od_dfs]

    tasks = []
    subset_names = list(subsets_id[0].keys())

    for seed_idx in range(len(id_dfs)):
        for name in subset_names:
            tasks.append(
                {
                    "seed": seed_idx,
                    "name": name,
                    "id_df": subsets_id[seed_idx][name],
                    "od_df": subsets_od[seed_idx][name],
                }
            )

    bootstrap_results = [{} for _ in range(len(id_dfs))]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {}
        for task in tasks:
            f = executor.submit(
                compute_metrics_task, task["id_df"], task["od_df"], bootstrap_n
            )
            future_to_task[f] = (task["seed"], task["name"])

        for future in tqdm(
            as_completed(future_to_task),
            total=len(tasks),
            desc="Bootstrapping",
            unit="task",
        ):
            seed_idx, name = future_to_task[future]
            res = future.result()
            bootstrap_results[seed_idx][name] = res

    # Get subset sizes
    subset_sizes_id = {name: len(subsets_id[0][name]) for name in subset_names}
    subset_sizes_od = {name: len(subsets_od[0][name]) for name in subset_names}

    return bootstrap_results, subset_names, subset_sizes_id, subset_sizes_od


def aggregate_statistics(bootstrap_results, subset_names, n_seeds, metrics_list):
    """Aggregate bootstrap results into final statistics."""
    n_samples_id = sum([CLASS_SAMPLE_SIZES["ID"][d] for d in ["CN", "AD", "FTD"]])
    n_samples_od = sum([CLASS_SAMPLE_SIZES["OD"][d] for d in ["CN", "AD", "FTD"]])

    final_id = {}
    final_od = {}

    for name in subset_names:
        final_id[name] = {}
        final_od[name] = {}

        for metric in metrics_list:
            # Collect values across seeds
            vals_id = [
                bootstrap_results[s][name][DOMAIN_ID][metric]["mean"]
                for s in range(n_seeds)
            ]
            vals_od = [
                bootstrap_results[s][name][DOMAIN_OD][metric]["mean"]
                for s in range(n_seeds)
            ]

            # ID Stats
            mu_id, std_id = float(np.mean(vals_id)), float(np.std(vals_id))
            cv_id = (std_id / mu_id * n_samples_id**0.5) if mu_id != 0 else None
            final_id[name][metric] = {"mean": mu_id, "std": std_id, "cv": cv_id}

            # OD Stats
            mu_od, std_od = float(np.mean(vals_od)), float(np.std(vals_od))
            cv_od = (std_od / mu_od * n_samples_od**0.5) if mu_od != 0 else None
            final_od[name][metric] = {"mean": mu_od, "std": std_od, "cv": cv_od}

    return final_id, final_od


def run_compute(args):
    """Run compute mode: calculate metrics and save to JSON."""
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loading GT
    logging.info("Loading Ground Truths...")
    od_gt = pd.read_csv(args.od_gt)
    id_files = sorted(list(args.id_gt_dir.glob("fold_*.csv")))
    id_gt = pd.concat([pd.read_csv(f) for f in id_files], ignore_index=True)

    # 2. Loading Predictions (TTA and No-TTA)
    logging.info(f"Loading {args.n_seeds} TTA seeds...")
    tta_id_dfs, tta_od_dfs = load_predictions(
        args.results_dir, args.tta_pattern, args.n_seeds, id_gt, od_gt
    )

    logging.info(f"Loading {args.n_seeds} No-TTA seeds...")
    no_tta_id_dfs, no_tta_od_dfs = load_predictions(
        args.results_dir, args.no_tta_pattern, args.n_seeds, id_gt, od_gt
    )

    if not tta_id_dfs or not no_tta_id_dfs:
        raise ValueError("No prediction files found. Check file patterns.")

    # 3. Generate subsets and compute metrics
    logging.info("Generating stratified subsets...")
    fractions = np.linspace(args.min_frac, 1.0, args.n_fractions)
    metrics_list = ["accuracy", "mcc", "pr_auc", "macro_f1"]

    # Process TTA
    logging.info("Processing TTA predictions...")
    tta_bootstrap, subset_names, tta_sizes_id, tta_sizes_od = process_predictions(
        tta_id_dfs,
        tta_od_dfs,
        fractions,
        args.seed,
        args.bootstrap_n,
        args.max_workers,
    )

    # Process No-TTA
    logging.info("Processing No-TTA predictions...")
    no_tta_bootstrap, _, no_tta_sizes_id, no_tta_sizes_od = process_predictions(
        no_tta_id_dfs,
        no_tta_od_dfs,
        fractions,
        args.seed,
        args.bootstrap_n,
        args.max_workers,
    )

    # 4. Aggregate statistics
    logging.info("Aggregating statistics...")
    tta_final_id, tta_final_od = aggregate_statistics(
        tta_bootstrap, subset_names, len(tta_id_dfs), metrics_list
    )
    no_tta_final_id, no_tta_final_od = aggregate_statistics(
        no_tta_bootstrap, subset_names, len(no_tta_id_dfs), metrics_list
    )

    # 5. Save results to JSON
    results = {
        "metadata": {
            "n_seeds": args.n_seeds,
            "n_fractions": args.n_fractions,
            "min_frac": args.min_frac,
            "bootstrap_n": args.bootstrap_n,
            "seed": args.seed,
            "tta_pattern": args.tta_pattern,
            "no_tta_pattern": args.no_tta_pattern,
            "metrics": metrics_list,
        },
        "subset_names": subset_names,
        "subset_sizes": {
            "tta": {"id": tta_sizes_id, "od": tta_sizes_od},
            "no_tta": {"id": no_tta_sizes_id, "od": no_tta_sizes_od},
        },
        "tta": {
            "id": tta_final_id,
            "od": tta_final_od,
        },
        "no_tta": {
            "id": no_tta_final_id,
            "od": no_tta_final_od,
        },
    }

    output_json = args.output_dir / args.json_file
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"Results saved to {output_json}")


# =============================================================================
# PLOT FUNCTIONS
# =============================================================================


def load_results(input_file):
    """Load computed results from JSON file."""
    with open(input_file, "r") as f:
        return json.load(f)


def plot_comparison(results, args):
    """Generate the TTA vs No-TTA comparison plot."""
    plt.rcParams.update(PLOT_PARAMS)

    subset_names = results["subset_names"]
    metrics_to_plot = args.metrics or results["metadata"]["metrics"]

    # Get sizes and sort indices
    sizes_tta_id = results["subset_sizes"]["tta"]["id"]
    sizes_tta_od = results["subset_sizes"]["tta"]["od"]

    x_unsorted_id = np.array([sizes_tta_id[n] for n in subset_names])
    sort_idx_id = np.argsort(x_unsorted_id)
    x_id = x_unsorted_id[sort_idx_id]

    x_unsorted_od = np.array([sizes_tta_od[n] for n in subset_names])
    sort_idx_od = np.argsort(x_unsorted_od)
    x_od = x_unsorted_od[sort_idx_od]

    sorted_names_id = [subset_names[i] for i in sort_idx_id]
    sorted_names_od = [subset_names[i] for i in sort_idx_od]

    # Create figure (no sharey to show ticks on both axes)
    fig, (ax_id, ax_od) = plt.subplots(1, 2, figsize=tuple(args.figsize))

    # Plot each metric
    for metric in metrics_to_plot:
        if metric not in METRICS_CONFIG:
            logging.warning(f"Unknown metric: {metric}, skipping")
            continue

        cfg = METRICS_CONFIG[metric]

        # TTA - ID
        y_tta_id = [results["tta"]["id"][n][metric]["cv"] for n in sorted_names_id]
        ax_id.plot(
            x_id,
            y_tta_id,
            color=cfg["color"],
            marker=cfg["marker"],
            linestyle=TTA_STYLE["linestyle"],
            markersize=TTA_STYLE["markersize"],
            linewidth=TTA_STYLE["linewidth"],
            alpha=TTA_STYLE["alpha"],
        )

        # No-TTA - ID
        y_no_tta_id = [
            results["no_tta"]["id"][n][metric]["cv"] for n in sorted_names_id
        ]
        ax_id.plot(
            x_id,
            y_no_tta_id,
            color=cfg["color"],
            marker=cfg["marker"],
            linestyle=NO_TTA_STYLE["linestyle"],
            markersize=NO_TTA_STYLE["markersize"],
            linewidth=NO_TTA_STYLE["linewidth"],
            alpha=NO_TTA_STYLE["alpha"],
        )

        # TTA - OD
        y_tta_od = [results["tta"]["od"][n][metric]["cv"] for n in sorted_names_od]
        ax_od.plot(
            x_od,
            y_tta_od,
            color=cfg["color"],
            marker=cfg["marker"],
            linestyle=TTA_STYLE["linestyle"],
            markersize=TTA_STYLE["markersize"],
            linewidth=TTA_STYLE["linewidth"],
            alpha=TTA_STYLE["alpha"],
        )

        # No-TTA - OD
        y_no_tta_od = [
            results["no_tta"]["od"][n][metric]["cv"] for n in sorted_names_od
        ]
        ax_od.plot(
            x_od,
            y_no_tta_od,
            color=cfg["color"],
            marker=cfg["marker"],
            linestyle=NO_TTA_STYLE["linestyle"],
            markersize=NO_TTA_STYLE["markersize"],
            linewidth=NO_TTA_STYLE["linewidth"],
            alpha=NO_TTA_STYLE["alpha"],
        )

    # Styling
    ax_id.set_xlabel(r"Dataset Size [ID]", fontsize=12)
    ax_od.set_xlabel(r"Dataset Size [OOD]", fontsize=12)
    ax_id.set_ylabel(args.y_label, fontsize=12)
    # Sync Y limits between both axes
    y_min = min(ax_id.get_ylim()[0], ax_od.get_ylim()[0])
    y_max = max(ax_id.get_ylim()[1], ax_od.get_ylim()[1])
    ax_id.set_ylim(y_min, y_max)
    ax_od.set_ylim(y_min, y_max)

    for ax in (ax_id, ax_od):
        ax.grid(axis="y", linestyle=":", color="gray", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Default title unless --no-title is set
    if not args.no_title:
        fig.suptitle(
            "Stability Analysis: TTA vs No-TTA by Dataset Size",
            fontsize=14,
            fontweight="bold",
        )

    # Create custom legend with black border (publication-ready)
    metric_handles = [
        Line2D(
            [0],
            [0],
            color=METRICS_CONFIG[m]["color"],
            marker=METRICS_CONFIG[m]["marker"],
            linestyle="-",
            markersize=6,
            linewidth=2,
            label=METRICS_CONFIG[m]["label"],
        )
        for m in metrics_to_plot
        if m in METRICS_CONFIG
    ]

    style_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=TTA_STYLE["linestyle"],
            linewidth=TTA_STYLE["linewidth"],
            label=TTA_STYLE["label"],
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=NO_TTA_STYLE["linestyle"],
            linewidth=NO_TTA_STYLE["linewidth"],
            alpha=NO_TTA_STYLE["alpha"],
            label=NO_TTA_STYLE["label"],
        ),
    ]

    all_handles = metric_handles + style_handles

    # Tight layout first, then add legend below
    plt.tight_layout(pad=0)

    # Legend with black border, no padding, directly below plots
    legend = fig.legend(
        all_handles,
        [h.get_label() for h in all_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=len(metric_handles) + 2,
        fancybox=False,
        edgecolor="black",
        framealpha=1,
        fontsize=10,
        frameon=True,
        borderpad=0.4,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    legend.get_frame().set_linewidth(1.5)

    return fig


def run_plot(args):
    """Run plot mode: load JSON and generate plot."""
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine input file path
    input_file = args.output_dir / args.json_file
    if not input_file.exists():
        # Try as absolute path
        input_file = args.json_file
    if not input_file.exists():
        raise FileNotFoundError(f"Results file not found: {args.json_file}")

    # Load results
    logging.info(f"Loading results from {input_file}")
    results = load_results(input_file)

    # Log metadata
    meta = results["metadata"]
    logging.info(
        f"Data: {meta['n_seeds']} seeds, {meta['n_fractions']} fractions, "
        f"{meta['bootstrap_n']} bootstrap iterations"
    )

    # Generate plot
    logging.info("Generating plot...")
    fig = plot_comparison(results, args)

    # Save with no extra padding
    out_file = args.output_dir / f"{args.output_name}.{args.output_fmt}"
    fig.savefig(out_file, dpi=args.dpi, pad_inches=0, bbox_inches="tight")
    plt.close(fig)

    logging.info(f"Plot saved to {out_file}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    args = get_args()

    if args.compute:
        run_compute(args)
    elif args.plot:
        run_plot(args)


if __name__ == "__main__":
    main()

# =============================================================================
# USAGE EXAMPLES
# =============================================================================
#
# 1. COMPUTE metrics and save to JSON:
# python visualizations/results/ci_seeds_vs_dataset_size.py --compute \
#     --od_gt "/path/to/test.csv" \
#     --id_gt_dir "/path/to/10fold_CV/" \
#     --results_dir "visualizations/outputs/metrics_results/ensemble_predictions" \
#     --output_dir "visualizations/outputs/" \
#     --json_file "ci_seeds_vs_dataset_size_results.json" \
#     --tta_pattern "ensemble_n1_folds10_tta-{i}" \
#     --no_tta_pattern "ensemble_n1_folds10_swin-5c-no_seed-baseline-{i}" \
#     --n_seeds 5 \
#     --n_fractions 15 \
#     --max_workers 16
#
# 2. PLOT from saved JSON:
# python visualizations/results/ci_seeds_vs_dataset_size.py --plot \
#     --output_dir "visualizations/outputs/" \
#     --json_file "ci_seeds_vs_dataset_size_results.json" \
#     --output_fmt "pdf" \
#     --no-title \
#     --figsize 14 6
