"""Compute classification metrics and generate violin plots from prediction CSVs."""

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import softmax
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.bootstrap_metric import compute_bootstrap_metrics


DEFAULT_DISCARD_SUBJECTS = [
    "NACC590492_1_T1w",
    "NACC882001_2_T1w",
    "NACC030231_5_T1w",
    "NACC065622_1_T1w_0",
    "NACC273830_1_T1w",
    "NACC810968_1_T1w",
    "NACC818046_1_T1w",
]
DOMAIN_ID = "ID"
DOMAIN_OD = "OD"
METRIC_ACC = "accuracy"
METRIC_BACC = "balanced_accuracy"
METRIC_ROC_AUC = "roc_auc"
METRIC_PR_AUC = "pr_auc"
METRIC_F1 = "f1"
METRIC_RECALL = "recall"
METRIC_PRECISION = "precision"
METRIC_MACRO_F1 = "macro_f1"
METRIC_MACRO_PRECISION = "macro_prec"
METRIC_MACRO_RECALL = "macro_sens"
METRIC_MCC = "mcc"
METRIC_ECE = "ece"
METRIC_MCE = "mce"
METRIC_BRIER = "brier_score"
METRIC_BRIER_CLASS = "brier_class"
METRIC_SOFTMAX_ENTROPY = "softmax_entropy"
METRIC_GINI = "gini"
METRIC_RENYI = "renyi"
PREFERRED_DIAGNOSIS_ORDER = ["CN", "AD", "FTD", "BV", "PNFA", "SD"]
# PREFERRED_DIAGNOSIS_ORDER = ["AD", "CN", "DLB", "BV", "PNFA", "SD", "PSP"] # LifespanTree
PREFERRED_METRICS_ORDER = [
    METRIC_ACC,
    METRIC_BACC,
    METRIC_MCC,
    METRIC_ROC_AUC,
    METRIC_PR_AUC,
    METRIC_MACRO_F1,
    METRIC_ECE,
    METRIC_MCE,
    METRIC_BRIER,
    METRIC_SOFTMAX_ENTROPY,
    METRIC_GINI,
    METRIC_RENYI,
    METRIC_F1,
]

VIOLIN_HALF_OFFSET = 0.22
TEXT_OFFSET = 0.15

OVERALL_PATTERN = re.compile(
    r"^(accuracy|balanced_accuracy|roc_auc|pr_auc|mcc|ece|mce|brier_score|softmax_entropy|gini|renyi|macro_f1|macro_prec|macro_sens|top_\d+_accuracy|top_\d+_balanced_accuracy|top_\d+_ece|top_\d+_mce|top_\d+_brier_score|top_\d+_macro_f1|top_\d+_macro_prec|top_\d+_macro_sens)$"
)
CLASS_PATTERN = re.compile(
    r"^(precision|recall|f1|brier_class|top_\d+_precision|top_\d+_recall|top_\d+_f1|top_\d+_brier_class)$"
)

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
        "text.usetex": False,
        "mathtext.fontset": "dejavuserif",
        "axes.labelsize": 13,
        "font.size": 12,
        "legend.fontsize": 11,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)
plt.style.use("seaborn-v0_8-whitegrid")


def _get_model_id_hash(model_ids: List[str]) -> str:
    """Generate a short hash representing a list of model IDs."""
    sorted_ids = sorted(model_ids)
    combined = "_".join(sorted_ids)
    hash_obj = hashlib.md5(combined.encode())
    return hash_obj.hexdigest()[:8]


def load_and_ensemble_data(
    input_folder: Path,
    model_name_ids: List[str],
    N: int,
    subjects_to_discard: List[str],
    datasets_to_include: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads, ensembles, and combines predictions for ID and OD domains."""

    def create_ensemble_for_domain(file_type: Literal["id", "od"]):
        all_dfs = []
        for model_id in model_name_ids:
            all_files = os.listdir(input_folder)

            csvs = sorted(
                [
                    f
                    for f in all_files
                    if f.endswith(f"{file_type}.csv")
                    and f.startswith("prediction_")
                    and model_id in f
                ]
            )

            selected_csvs = csvs[:N] if N > 0 else []
            if len(selected_csvs) != N and N > 0:
                print(
                    f"Warning: Expected {N} models for '{model_id}' ({file_type}), found {len(selected_csvs)}."
                )

            for p in [input_folder / f for f in selected_csvs]:
                all_dfs.append(pd.read_csv(p))
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    def combine_subject_predictions(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "Subject" not in df.columns:
            return pd.DataFrame()

        pred_cols = [c for c in df.columns if c.startswith("pred_")]

        def _group_mean(group):
            diag = group["Diagnosis"].mode()
            diag_val = diag[0] if not diag.empty else "Unknown"

            means = group[pred_cols].mean(numeric_only=True).to_dict()

            result = {"Diagnosis": diag_val, **means}
            if "Dataset" in group.columns:
                dataset = group["Dataset"].mode()
                result["Dataset"] = dataset[0] if not dataset.empty else "Unknown"

            return pd.Series(result)

        return (
            df.groupby("Subject").apply(_group_mean, include_groups=False).reset_index()
        )

    print("Creating and combining ensemble predictions...")
    id_combined = create_ensemble_for_domain("id")
    od_combined = create_ensemble_for_domain("od")

    final_id_df = combine_subject_predictions(id_combined)
    final_od_df = combine_subject_predictions(od_combined)

    if datasets_to_include:
        print(f"Filtering by datasets: {datasets_to_include}")

        def matches_any_pattern(dataset_value, patterns):
            """Check if dataset_value matches any of the provided patterns."""
            import fnmatch

            for pattern in patterns:
                if dataset_value == pattern:
                    return True
                if fnmatch.fnmatch(dataset_value, pattern):
                    return True
                if any(c in pattern for c in ["^", "$", "[", "]", "(", ")", "|", "+"]):
                    try:
                        if re.match(pattern, dataset_value):
                            return True
                    except re.error:
                        pass
            return False

        if not final_id_df.empty and "Dataset" in final_id_df.columns:
            before_count = len(final_id_df)
            final_id_df = final_id_df[
                final_id_df["Dataset"].apply(
                    lambda x: matches_any_pattern(x, datasets_to_include)
                )
            ]
            matched_datasets = (
                final_id_df["Dataset"].unique().tolist()
                if not final_id_df.empty
                else []
            )
            print(
                f"ID domain: {before_count} -> {len(final_id_df)} subjects after dataset filtering"
            )
            if matched_datasets:
                print(f"  Matched datasets: {sorted(matched_datasets)}")

        if not final_od_df.empty and "Dataset" in final_od_df.columns:
            before_count = len(final_od_df)
            final_od_df = final_od_df[
                final_od_df["Dataset"].apply(
                    lambda x: matches_any_pattern(x, datasets_to_include)
                )
            ]
            matched_datasets = (
                final_od_df["Dataset"].unique().tolist()
                if not final_od_df.empty
                else []
            )
            print(
                f"OD domain: {before_count} -> {len(final_od_df)} subjects after dataset filtering"
            )
            if matched_datasets:
                print(f"  Matched datasets: {sorted(matched_datasets)}")

    if not final_od_df.empty and subjects_to_discard:
        final_od_df = final_od_df[~final_od_df["Subject"].isin(subjects_to_discard)]

    return final_id_df, final_od_df


def plot_metric_violins_with_annotations(
    ax,
    results_dict,
    metric_config,
    domain_colors,
    x_labels,
    is_class_specific,
    round_digits=1,
):
    """Generic function to plot annotated violins for overall or per-class metrics."""
    plot_data = []
    for domain in [DOMAIN_ID, DOMAIN_OD]:
        domain_results = results_dict.get(domain)
        if not domain_results:
            continue

        if is_class_specific:
            for class_label in x_labels:
                metric_data = domain_results.get(metric_config, {}).get(class_label)
                if metric_data and "samples" in metric_data:
                    samples = metric_data["samples"] * 100
                    for s in samples[~np.isnan(samples)]:
                        plot_data.append(
                            {"Value (%)": s, "Category": class_label, "Domain": domain}
                        )
        else:
            for display_name, internal_key in metric_config.items():
                metric_data = domain_results.get(internal_key)
                if metric_data and "samples" in metric_data:
                    samples = metric_data["samples"] * 100
                    for s in samples[~np.isnan(samples)]:
                        plot_data.append(
                            {"Value (%)": s, "Category": display_name, "Domain": domain}
                        )

    if not plot_data:
        ax.text(
            0.5,
            0.5,
            "No data for violins.",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return

    df_violin = pd.DataFrame(plot_data)
    sns.violinplot(
        x="Category",
        y="Value (%)",
        hue="Domain",
        data=df_violin,
        ax=ax,
        palette=domain_colors,
        split=True,
        inner=None,
        cut=0,
        density_norm="width",
        hue_order=[DOMAIN_ID, DOMAIN_OD],
        order=x_labels,
    )

    for x_idx, cat_name in enumerate(x_labels):
        for dom_idx, domain in enumerate([DOMAIN_ID, DOMAIN_OD]):
            domain_res = results_dict.get(domain)
            if not domain_res:
                continue

            if is_class_specific:
                res_item = domain_res.get(metric_config, {}).get(cat_name)
            else:
                internal_key = next(
                    k for d, k in metric_config.items() if d == cat_name
                )
                res_item = domain_res.get(internal_key)

            if res_item and not np.isnan(res_item.get("mean", np.nan)):
                mean, lower, upper = (
                    res_item["mean"] * 100,
                    res_item["lower"] * 100,
                    res_item["upper"] * 100,
                )
                x_pos = (
                    x_idx - VIOLIN_HALF_OFFSET
                    if dom_idx == 0
                    else x_idx + VIOLIN_HALF_OFFSET
                )
                ax.vlines(
                    x_pos, lower, upper, color="k", linestyle="-", lw=2, zorder=10
                )
                ax.plot(
                    x_pos,
                    mean,
                    "D",
                    color="white",
                    markersize=6,
                    markeredgecolor="k",
                    zorder=11,
                    mew=1.5,
                )
                ax.text(
                    x_pos,
                    mean + (upper - mean) * 0.5,
                    f"{mean:.{round_digits}f}%",
                    fontsize=9,
                    ha="center",
                    va="bottom",
                    color="black",
                    zorder=12,
                    bbox=dict(
                        boxstyle="round,pad=0.2", fc="white", ec="lightgray", alpha=0.8
                    ),
                )

    if ax.get_legend():
        ax.legend(title="Domain", loc="best")


def plot_confusion_matrix(
    ax, cm_mean, diag_labels, title, round_digits=1, cm_lower=None, cm_upper=None
):
    """Plots an annotated confusion matrix."""
    if cm_mean is None or np.all(np.isnan(cm_mean)):
        ax.text(0.5, 0.5, "No Data", ha="center", va="center")
        ax.set_title(title, fontsize=12)
        return

    cm_norm = cm_mean / cm_mean.sum(axis=1, keepdims=True)

    annot = np.empty_like(cm_mean, dtype=object)

    num_classes = len(diag_labels)
    base_size = 11 if num_classes < 5 else 9

    for i in range(cm_mean.shape[0]):
        for j in range(cm_mean.shape[1]):
            mean_val = cm_mean[i, j]
            pct_val = cm_norm[i, j] * 100

            main_text = f"{mean_val:.{round_digits}f}\n({pct_val:.1f}%)"

            if cm_lower is not None and cm_upper is not None:
                ci_text = f"\n[{cm_lower[i, j]:.0f}-{cm_upper[i, j]:.0f}]"
                annot[i, j] = main_text + ci_text
            else:
                annot[i, j] = main_text

    sns.heatmap(
        cm_mean,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=diag_labels,
        yticklabels=diag_labels,
        ax=ax,
        cbar=False,
        annot_kws={"size": base_size},
        linewidths=1,
        linecolor="white",
    )

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.tick_params(length=0)


def plot_diagnosis_distribution(ax, diag_counts, title, colors):
    """Plots a pie chart of diagnosis distribution."""
    if diag_counts is None or diag_counts.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(f"{title} (n={diag_counts.sum()})", fontsize=12)
        return

    def autopct_format(values):
        def my_format(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return f"{val}\n({pct:.1f}%)"

        return my_format

    ax.pie(
        diag_counts.values,
        labels=diag_counts.index,
        autopct=autopct_format(diag_counts.values),
        colors=colors,
        textprops={"fontsize": 10},
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    ax.set_title(f"{title} (n={diag_counts.sum()})", fontsize=12)


def format_metric_with_ci(
    metric_dict: Optional[Dict[str, float]], round_digits: int, with_ci: bool = True
) -> str:
    """Formats a metric dictionary into a readable string with optional CI."""
    if not metric_dict or np.isnan(metric_dict.get("mean", np.nan)):
        return "N/A"

    mean = metric_dict["mean"] * 100
    if not with_ci:
        return f"{mean:.{round_digits}f}"

    lower = metric_dict.get("lower", 0.0) * 100
    upper = metric_dict.get("upper", 0.0) * 100
    return (
        f"{mean:.{round_digits}f} [{lower:.{round_digits}f}-{upper:.{round_digits}f}]"
    )


def prepare_csv_data(
    results: Dict[str, Any], config: Dict[str, Any], specific_k: int, with_ci: bool
) -> Dict[str, str]:
    """Prepares a single row of data for the results CSV."""
    k_prefix = "" if specific_k == 1 else f"top_{specific_k}_"
    row_label = f"{config['folder_name']}/{'_'.join(config['model_ids'])} ({config['N']}, Top-{specific_k})"
    if with_ci:
        row_label += " [95% CI]"

    csv_row = {"Model Name": row_label}
    metrics_to_show = config["metrics_to_show"]
    ordered_diags = config["all_diags"]
    round_digits = config["round_digits"]

    for dom in [DOMAIN_ID, DOMAIN_OD]:
        dom_data = results.get(dom, {})

        if "acc" in metrics_to_show:
            csv_row[f"{dom}-ACC"] = format_metric_with_ci(
                dom_data.get(f"{k_prefix}{METRIC_ACC}"), round_digits, with_ci
            )
        if "bacc" in metrics_to_show:
            csv_row[f"{dom}-BACC"] = format_metric_with_ci(
                dom_data.get(f"{k_prefix}{METRIC_BACC}"), round_digits, with_ci
            )
        if "roc_auc" in metrics_to_show and specific_k == 1:
            csv_row[f"{dom}-ROC-AUC"] = format_metric_with_ci(
                dom_data.get(METRIC_ROC_AUC), round_digits, with_ci
            )
        if "pr_auc" in metrics_to_show and specific_k == 1:
            csv_row[f"{dom}-PR-AUC"] = format_metric_with_ci(
                dom_data.get(METRIC_PR_AUC), round_digits, with_ci
            )
        if "mcc" in metrics_to_show and specific_k == 1:
            csv_row[f"{dom}-MCC"] = format_metric_with_ci(
                dom_data.get(METRIC_MCC), round_digits, with_ci
            )
        if "ece" in metrics_to_show:
            csv_row[f"{dom}-ECE"] = format_metric_with_ci(
                dom_data.get(f"{k_prefix}{METRIC_ECE}"), round_digits, with_ci
            )
        if "mce" in metrics_to_show:
            csv_row[f"{dom}-MCE"] = format_metric_with_ci(
                dom_data.get(f"{k_prefix}{METRIC_MCE}"), round_digits, with_ci
            )
        if "brier" in metrics_to_show:
            csv_row[f"{dom}-BRIER"] = format_metric_with_ci(
                dom_data.get(f"{k_prefix}{METRIC_BRIER}"), round_digits, with_ci
            )
        if "softmax_entropy" in metrics_to_show:
            csv_row[f"{dom}-SOFTMAX-ENTROPY"] = format_metric_with_ci(
                dom_data.get(METRIC_SOFTMAX_ENTROPY), round_digits, with_ci
            )
        if "gini" in metrics_to_show:
            csv_row[f"{dom}-GINI"] = format_metric_with_ci(
                dom_data.get(METRIC_GINI), round_digits, with_ci
            )
        if "renyi" in metrics_to_show:
            csv_row[f"{dom}-RENYI"] = format_metric_with_ci(
                dom_data.get(METRIC_RENYI), round_digits, with_ci
            )

        if "macro_f1" in metrics_to_show:
            csv_row[f"{dom}-MACRO-F1"] = format_metric_with_ci(
                dom_data.get(f"{k_prefix}{METRIC_MACRO_F1}"), round_digits, with_ci
            )
        if "macro_prec" in metrics_to_show:
            csv_row[f"{dom}-MACRO-PREC"] = format_metric_with_ci(
                dom_data.get(f"{k_prefix}{METRIC_MACRO_PRECISION}"),
                round_digits,
                with_ci,
            )
        if "macro_sens" in metrics_to_show:
            csv_row[f"{dom}-MACRO-SENS"] = format_metric_with_ci(
                dom_data.get(f"{k_prefix}{METRIC_MACRO_RECALL}"), round_digits, with_ci
            )

        for diag in ordered_diags:
            if "f1" in metrics_to_show:
                met = dom_data.get(f"{k_prefix}{METRIC_F1}", {}).get(diag)
                csv_row[f"{dom}-F1:{diag}"] = format_metric_with_ci(
                    met, round_digits, with_ci
                )
            if "sens" in metrics_to_show:
                met = dom_data.get(f"{k_prefix}{METRIC_RECALL}", {}).get(diag)
                csv_row[f"{dom}-SEN:{diag}"] = format_metric_with_ci(
                    met, round_digits, with_ci
                )
            if "prec" in metrics_to_show:
                met = dom_data.get(f"{k_prefix}{METRIC_PRECISION}", {}).get(diag)
                csv_row[f"{dom}-PREC:{diag}"] = format_metric_with_ci(
                    met, round_digits, with_ci
                )

    return csv_row


def create_summary_table(
    results: Dict[str, Any], k: int, metrics_to_show: List[str], round_digits: int
) -> List[List[str]]:
    """Creates a printable summary table for a specific k."""
    header = [
        f"Metric (Top-{k})",
        f"{DOMAIN_ID} Value [95% CI]",
        f"{DOMAIN_OD} Value [95% CI]",
    ]
    rows = [header]
    k_prefix = "" if k == 1 else f"top_{k}_"

    def get_formatted_vals(metric_key):
        id_val = format_metric_with_ci(
            results.get(DOMAIN_ID, {}).get(metric_key), round_digits
        )
        od_val = format_metric_with_ci(
            results.get(DOMAIN_OD, {}).get(metric_key), round_digits
        )
        return [id_val, od_val]

    if "acc" in metrics_to_show:
        rows.append(["Accuracy", *get_formatted_vals(f"{k_prefix}{METRIC_ACC}")])
    if "bacc" in metrics_to_show:
        rows.append(
            ["Balanced Accuracy", *get_formatted_vals(f"{k_prefix}{METRIC_BACC}")]
        )
    if "roc_auc" in metrics_to_show and k == 1:
        rows.append(["ROC-AUC", *get_formatted_vals(METRIC_ROC_AUC)])
    if "pr_auc" in metrics_to_show and k == 1:
        rows.append(["PR-AUC", *get_formatted_vals(METRIC_PR_AUC)])
    if "mcc" in metrics_to_show and k == 1:
        rows.append(["MCC", *get_formatted_vals(METRIC_MCC)])
    if "ece" in metrics_to_show:
        rows.append(["ECE", *get_formatted_vals(f"{k_prefix}{METRIC_ECE}")])
    if "mce" in metrics_to_show:
        rows.append(["MCE", *get_formatted_vals(f"{k_prefix}{METRIC_MCE}")])
    if "brier" in metrics_to_show:
        rows.append(["Brier Score", *get_formatted_vals(f"{k_prefix}{METRIC_BRIER}")])
    if "softmax_entropy" in metrics_to_show:
        rows.append(["Softmax Entropy", *get_formatted_vals(METRIC_SOFTMAX_ENTROPY)])
    if "gini" in metrics_to_show:
        rows.append(["Gini Index", *get_formatted_vals(METRIC_GINI)])
    if "renyi" in metrics_to_show:
        rows.append(["Rényi Entropy", *get_formatted_vals(METRIC_RENYI)])
    if "macro_f1" in metrics_to_show:
        rows.append(["Macro F1", *get_formatted_vals(f"{k_prefix}{METRIC_MACRO_F1}")])
    if "macro_prec" in metrics_to_show:
        rows.append(
            [
                "Macro Precision",
                *get_formatted_vals(f"{k_prefix}{METRIC_MACRO_PRECISION}"),
            ]
        )
    if "macro_sens" in metrics_to_show:
        rows.append(
            [
                "Macro Sensitivity",
                *get_formatted_vals(f"{k_prefix}{METRIC_MACRO_RECALL}"),
            ]
        )

    all_diags = set(results.get(DOMAIN_ID, {}).get("diags", [])).union(
        results.get(DOMAIN_OD, {}).get("diags", [])
    )
    ordered_diags = [d for d in PREFERRED_DIAGNOSIS_ORDER if d in all_diags]

    for diag in ordered_diags:
        if "f1" in metrics_to_show:
            id_f1 = format_metric_with_ci(
                results.get(DOMAIN_ID, {}).get(f"{k_prefix}{METRIC_F1}", {}).get(diag),
                round_digits,
            )
            od_f1 = format_metric_with_ci(
                results.get(DOMAIN_OD, {}).get(f"{k_prefix}{METRIC_F1}", {}).get(diag),
                round_digits,
            )
            rows.append([f"F1 ({diag})", id_f1, od_f1])
        if "sens" in metrics_to_show:
            id_sen = format_metric_with_ci(
                results.get(DOMAIN_ID, {})
                .get(f"{k_prefix}{METRIC_RECALL}", {})
                .get(diag),
                round_digits,
            )
            od_sen = format_metric_with_ci(
                results.get(DOMAIN_OD, {})
                .get(f"{k_prefix}{METRIC_RECALL}", {})
                .get(diag),
                round_digits,
            )
            rows.append([f"Sensitivity ({diag})", id_sen, od_sen])
        if "prec" in metrics_to_show:
            id_prec = format_metric_with_ci(
                results.get(DOMAIN_ID, {})
                .get(f"{k_prefix}{METRIC_PRECISION}", {})
                .get(diag),
                round_digits,
            )
            od_prec = format_metric_with_ci(
                results.get(DOMAIN_OD, {})
                .get(f"{k_prefix}{METRIC_PRECISION}", {})
                .get(diag),
                round_digits,
            )
            rows.append([f"Precision ({diag})", id_prec, od_prec])

    return rows


def calculate_metrics(
    domains_data: Dict[str, pd.DataFrame],
    bootstrap_method: str,
    num_bootstrap_iter: int,
    top_ns: Optional[List[int]],
) -> Dict[str, Any]:
    """Preprocesses data and computes metrics for each domain."""
    results = {}

    if bootstrap_method == "stratified":
        print(
            "Warning: 'stratified' bootstrap method selected, but the imported "
            "'utils.bootstrap_metric' may use its own sampling logic (e.g., standard with retry)."
        )

    for domain_name, df in domains_data.items():
        if df.empty or "Diagnosis" not in df.columns:
            print(f"Skipping {domain_name} domain: data empty or 'Diagnosis' missing.")
            continue

        diag_counts_raw = df["Diagnosis"].value_counts()
        ordered_diags = [
            d for d in PREFERRED_DIAGNOSIS_ORDER if d in diag_counts_raw.index
        ]
        remaining_diags = sorted(
            [d for d in diag_counts_raw.index if d not in PREFERRED_DIAGNOSIS_ORDER]
        )
        diag_counts = diag_counts_raw.loc[ordered_diags + remaining_diags]
        print(f"Diagnosis counts for {domain_name}: {diag_counts.to_dict()}")

        unique_diags = df["Diagnosis"].unique().tolist()
        ordered_diags = [
            d for d in PREFERRED_DIAGNOSIS_ORDER if d in unique_diags
        ] + sorted([d for d in unique_diags if d not in PREFERRED_DIAGNOSIS_ORDER])

        diag_to_num = {d: i for i, d in enumerate(ordered_diags)}
        df_filtered = df[df["Diagnosis"].isin(diag_to_num.keys())].copy()
        df_filtered["Diagnosis_cat"] = pd.Categorical(
            df_filtered["Diagnosis"], categories=ordered_diags, ordered=True
        )

        gt_labels = df_filtered["Diagnosis_cat"].cat.codes.values

        pred_cols = [f"pred_{d}" for d in ordered_diags]
        pred_cols_ensemble = [f"pred_{d}_ensemble" for d in ordered_diags]

        if all(col in df_filtered.columns for col in pred_cols):
            actual_pred_cols = pred_cols
        elif all(col in df_filtered.columns for col in pred_cols_ensemble):
            actual_pred_cols = pred_cols_ensemble
        else:
            missing_cols = [
                col
                for col in pred_cols
                if col not in df_filtered.columns
                and col.replace("_ensemble", "") not in df_filtered.columns
            ]
            if missing_cols:
                print(
                    f"Warning: Missing prediction columns for domain '{domain_name}': {missing_cols}."
                    " Metrics may be incomplete."
                )
                actual_pred_cols = [
                    c for c in pred_cols if c in df_filtered.columns
                ] or [c for c in pred_cols_ensemble if c in df_filtered.columns]
                if not actual_pred_cols:
                    print(
                        f"CRITICAL: No prediction columns found for {domain_name}. Skipping."
                    )
                    continue
            else:
                actual_pred_cols = pred_cols

        probs_raw = df_filtered[actual_pred_cols].values
        probs_softmax = (
            softmax(probs_raw, axis=1) if probs_raw.size > 0 else np.array([])
        )

        if gt_labels.size == 0 or probs_softmax.size == 0:
            print(
                f"CRITICAL: No valid data for {domain_name} after processing. Cannot compute metrics."
            )
            continue

        bs_results_numeric = compute_bootstrap_metrics(
            gt_labels,
            probs_softmax,
            n_bootstrap=num_bootstrap_iter,
            top_n=top_ns,
            n_jobs=max(1, os.cpu_count() // 2 if os.cpu_count() else 1),
        )

        domain_results = {"diags": ordered_diags, "diags_count": diag_counts}
        for m_key, m_val in bs_results_numeric.items():
            if CLASS_PATTERN.match(m_key):
                domain_results[m_key] = {
                    ordered_diags[int(k)]: v
                    for k, v in m_val.items()
                    if isinstance(k, (int, np.integer))
                    or (isinstance(k, str) and k.isdigit())
                }
            else:
                domain_results[m_key] = m_val
        results[domain_name] = domain_results

    return results


def generate_visualizations(
    results: Dict[str, Any], output_folder: Path, config: Dict[str, Any]
):
    print("Generating violin figures...")

    colors_domain = ["#2c7bb6", "#d7191c"]
    colors_pie = sns.color_palette("Pastel1")

    top_ns_list = config.get("top_ns") or []
    figure_topks = [1] + [k for k in top_ns_list if k != 1]

    model_id_hash = _get_model_id_hash(config["model_ids"])
    num_models = len(config["model_ids"])
    base_fig_root = f"{config['folder_name']}_{model_id_hash}_{num_models}models_{config['N']}each_{config['bootstrap_method']}_violin_bs{config['n_bootstrap']}_metrics"

    id_diags = results.get(DOMAIN_ID, {}).get("diags", [])
    od_diags = results.get(DOMAIN_OD, {}).get("diags", [])
    all_plot_diags = sorted(list(set(id_diags).union(od_diags)))

    for k in figure_topks:
        k_prefix = "" if k == 1 else f"top_{k}_"

        fig = plt.figure(figsize=(22, 20), facecolor="white")

        gs = gridspec.GridSpec(
            4, 6, figure=fig, hspace=0.25, wspace=0.3, height_ratios=[1.4, 1.4, 1, 1]
        )

        metrics_class = [
            ("Precision", f"{k_prefix}{METRIC_PRECISION}"),
            ("Recall", f"{k_prefix}{METRIC_RECALL}"),
            ("F1-Score", f"{k_prefix}{METRIC_F1}"),
        ]

        for i, (name, key) in enumerate(metrics_class):
            ax = fig.add_subplot(gs[0, i * 2 : (i * 2) + 2])
            plot_metric_violins_with_annotations(
                ax,
                results,
                key,
                colors_domain,
                all_plot_diags,
                True,
                config["round_digits"],
            )
            ax.set_title(f"{name} by Class", fontsize=14)
            ax.set_xlabel("")
            ax.grid(axis="y", linestyle="--", alpha=0.5)

        perf_metrics_list = [
            METRIC_ACC,
            METRIC_BACC,
            METRIC_MCC,
            METRIC_ROC_AUC,
            METRIC_MACRO_F1,
        ]
        perf_config = {}
        metric_names_map = {
            METRIC_ACC: "Accuracy",
            METRIC_BACC: "Bal. Acc.",
            METRIC_MCC: "MCC",
            METRIC_ROC_AUC: "ROC AUC",
            METRIC_MACRO_F1: "Macro F1",
        }

        for m in perf_metrics_list:
            key = f"{k_prefix}{m}" if k > 1 and m != METRIC_ROC_AUC else m
            if k == 1 or m not in [METRIC_ROC_AUC, METRIC_MCC]:
                perf_config[metric_names_map.get(m, m)] = key

        ax_perf = fig.add_subplot(gs[1, 0:3])
        plot_metric_violins_with_annotations(
            ax_perf,
            results,
            perf_config,
            colors_domain,
            list(perf_config.keys()),
            False,
            config["round_digits"],
        )
        ax_perf.set_title(
            "Global Performance Metrics",
            fontsize=15,
            color="#333333",
        )
        ax_perf.set_xlabel("")
        ax_perf.set_ylim(0, 105)
        ax_perf.grid(axis="y", linestyle="--", alpha=0.5)

        uncert_metrics_list = [
            METRIC_ECE,
            METRIC_BRIER,
            METRIC_SOFTMAX_ENTROPY,
            METRIC_GINI,
        ]
        uncert_config = {}
        uncert_names_map = {
            METRIC_ECE: "ECE",
            METRIC_BRIER: "Brier",
            METRIC_SOFTMAX_ENTROPY: "Entropy",
            METRIC_GINI: "Gini",
        }

        for m in uncert_metrics_list:
            key = (
                f"{k_prefix}{m}"
                if k > 1 and m not in [METRIC_SOFTMAX_ENTROPY, METRIC_GINI]
                else m
            )
            uncert_config[uncert_names_map.get(m, m)] = key

        ax_uncert = fig.add_subplot(gs[1, 3:6])
        plot_metric_violins_with_annotations(
            ax_uncert,
            results,
            uncert_config,
            colors_domain,
            list(uncert_config.keys()),
            False,
            config["round_digits"],
        )
        ax_uncert.set_title("Calibration & Uncertainty", fontsize=15, color="#333333")
        ax_uncert.set_xlabel("")
        ax_uncert.grid(axis="y", linestyle="--", alpha=0.5)

        domains_setup = [(DOMAIN_ID, 2), (DOMAIN_OD, 3)]

        for domain, row_idx in domains_setup:
            if domain in results:
                ax_cm = fig.add_subplot(gs[row_idx, 0:4])
                cm_data = results[domain].get("confusion_matrix", {})

                plot_confusion_matrix(
                    ax_cm,
                    cm_data.get("mean"),
                    results[domain]["diags"],
                    f"Confusion Matrix - {domain}",
                    config["round_digits"],
                    cm_data.get("lower"),
                    cm_data.get("upper"),
                )

                ax_pie = fig.add_subplot(gs[row_idx, 4:6])
                plot_diagnosis_distribution(
                    ax_pie,
                    results[domain].get("diags_count"),
                    f"Distribution {domain}",
                    colors_pie,
                )

        fig.suptitle(
            f"Model Evaluation Report: {config['folder_name']}\n{num_models} Models Ensemble | Top-{k} Metrics",
            fontsize=20,
            fontweight="bold",
            y=0.95,
        )

        fig_path = output_folder / (
            f"{base_fig_root}_top{k}" if k > 1 else base_fig_root
        )
        plt.savefig(f"{fig_path}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{fig_path}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved violin figure for Top-{k} to {fig_path}.png and {fig_path}.pdf")


def _get_ordered_columns(
    metrics_to_show: List[str], ordered_diags: List[str]
) -> List[str]:
    """Generate properly ordered column names for CSV output."""
    columns = ["Model Name"]

    metric_to_suffix = {
        METRIC_ACC: "ACC",
        METRIC_BACC: "BACC",
        METRIC_MCC: "MCC",
        METRIC_ROC_AUC: "ROC-AUC",
        METRIC_PR_AUC: "PR-AUC",
        METRIC_MACRO_F1: "MACRO-F1",
        METRIC_ECE: "ECE",
        METRIC_MCE: "MCE",
        METRIC_BRIER: "BRIER",
        METRIC_SOFTMAX_ENTROPY: "SOFTMAX-ENTROPY",
        METRIC_GINI: "GINI",
        METRIC_RENYI: "RENYI",
        METRIC_F1: None,
        METRIC_MACRO_PRECISION: "MACRO-PREC",
        METRIC_MACRO_RECALL: "MACRO-SENS",
    }

    metric_name_map = {
        "acc": METRIC_ACC,
        "bacc": METRIC_BACC,
        "mcc": METRIC_MCC,
        "roc_auc": METRIC_ROC_AUC,
        "pr_auc": METRIC_PR_AUC,
        "ece": METRIC_ECE,
        "mce": METRIC_MCE,
        "brier": METRIC_BRIER,
        "softmax_entropy": METRIC_SOFTMAX_ENTROPY,
        "gini": METRIC_GINI,
        "renyi": METRIC_RENYI,
        "macro_f1": METRIC_MACRO_F1,
        "macro_prec": METRIC_MACRO_PRECISION,
        "macro_sens": METRIC_MACRO_RECALL,
    }

    for domain in [DOMAIN_ID, DOMAIN_OD]:
        for metric in PREFERRED_METRICS_ORDER:
            for show_name, internal_name in metric_name_map.items():
                if internal_name == metric and show_name in metrics_to_show:
                    suffix = metric_to_suffix.get(metric)
                    if suffix:
                        columns.append(f"{domain}-{suffix}")
                    break

        for diag in ordered_diags:
            if "f1" in metrics_to_show:
                columns.append(f"{domain}-F1:{diag}")
            if "sens" in metrics_to_show:
                columns.append(f"{domain}-SEN:{diag}")
            if "prec" in metrics_to_show:
                columns.append(f"{domain}-PREC:{diag}")

    return columns


def generate_reports(
    results: Dict[str, Any], output_folder: Path, config: Dict[str, Any]
):
    """Generates and saves CSV reports and prints summary tables."""
    csv_rows = []
    top_ns_list = config.get("top_ns") or []
    report_topks = [1] + [k for k in top_ns_list if k != 1]

    id_diags = results.get(DOMAIN_ID, {}).get("diags", [])
    od_diags = results.get(DOMAIN_OD, {}).get("diags", [])
    all_unique_diags = set(id_diags).union(od_diags)
    ordered_diags = [d for d in PREFERRED_DIAGNOSIS_ORDER if d in all_unique_diags]
    remaining_diags = sorted(
        [d for d in all_unique_diags if d not in PREFERRED_DIAGNOSIS_ORDER]
    )
    config["all_diags"] = ordered_diags + remaining_diags

    for k in report_topks:
        csv_rows.append(prepare_csv_data(results, config, specific_k=k, with_ci=False))
        csv_rows.append(prepare_csv_data(results, config, specific_k=k, with_ci=True))

    df_new = pd.DataFrame(csv_rows)

    ordered_cols = _get_ordered_columns(config["metrics_to_show"], config["all_diags"])
    ordered_cols = [col for col in ordered_cols if col in df_new.columns]
    remaining_cols = [col for col in df_new.columns if col not in ordered_cols]
    df_new = df_new[ordered_cols + remaining_cols]

    model_id_hash = _get_model_id_hash(config["model_ids"])
    num_models = len(config["model_ids"])
    csv_path = (
        output_folder
        / f"{config['folder_name']}_{model_id_hash}_{num_models}models.csv"
    )
    df_new.to_csv(csv_path, index=False)
    print(f"Saved CSV report to: {csv_path}")

    if config.get("append_csv"):
        append_path = Path(config["append_csv"])
        df_old = pd.read_csv(append_path) if append_path.exists() else pd.DataFrame()
        pd.concat([df_old, df_new], ignore_index=True).to_csv(append_path, index=False)
        print(f"Appended results to {append_path}")

    model_list_str = ", ".join(config["model_ids"][:3])
    if len(config["model_ids"]) > 3:
        model_list_str += f" ... (+{len(config['model_ids']) - 3} more)"

    print(
        "\n"
        + "=" * 80
        + f"\nMODEL RESULTS: {config['folder_name']} / {len(config['model_ids'])} models ({model_list_str}) (N={config['N']})\n"
        + "=" * 80
    )
    for k in report_topks:
        table = create_summary_table(
            results, k, config["metrics_to_show"], config["round_digits"]
        )
        print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))


def run_analysis(
    input_folder: Path,
    output_folder: Path,
    N_models: int,
    model_names: List[str],
    append_csv: Optional[str],
    n_bootstrap: int,
    bootstrap_method: str,
    top_n: Optional[List[int]],
    discard_subjects_file: Optional[str],
    metrics_to_show: List[str],
    round_digits: int,
    datasets_filter: Optional[List[str]] = None,
    save_ensemble: bool = False,
):
    """Main orchestrator for the analysis pipeline."""
    output_folder.mkdir(parents=True, exist_ok=True)

    discard_list = DEFAULT_DISCARD_SUBJECTS
    if discard_subjects_file:
        with open(discard_subjects_file, "r") as f:
            discard_list = [line.strip() for line in f if line.strip()]
        print(f"Using custom subject discard list from: {discard_subjects_file}")

    id_df, od_df = load_and_ensemble_data(
        input_folder, model_names, N_models, discard_list, datasets_filter
    )
    if id_df.empty and od_df.empty:
        print("No data loaded for any model ID. Exiting.")
        return

    if save_ensemble:
        ensemble_dir = output_folder / "ensemble_predictions"
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        folder_name = input_folder.name
        base_name = f"ensemble_n{N_models}_folds{len(model_names)}_{folder_name}"

        if not id_df.empty:
            id_path = ensemble_dir / f"{base_name}_id.csv"
            id_df.to_csv(id_path, index=False)
            print(f"Saved ID ensemble to: {id_path}", end=" | ")
            print(
                f"({len(id_df)} subjects, {len(model_names)} folds, {N_models} model(s) per fold)"
            )

        if not od_df.empty:
            od_path = ensemble_dir / f"{base_name}_od.csv"
            od_df.to_csv(od_path, index=False)
            print(f"Saved OD ensemble to: {od_path}", end=" | ")
            print(f"({len(od_df)} subjects, averaged across {len(model_names)} folds)")

    domains_data = {DOMAIN_ID: id_df, DOMAIN_OD: od_df}
    results = calculate_metrics(domains_data, bootstrap_method, n_bootstrap, top_n)

    config = {
        "folder_name": input_folder.name,
        "model_ids": model_names,
        "N": N_models,
        "bootstrap_method": bootstrap_method,
        "n_bootstrap": n_bootstrap,
        "top_ns": top_n,
        "metrics_to_show": metrics_to_show,
        "round_digits": round_digits,
        "append_csv": append_csv,
    }

    generate_visualizations(results, output_folder, config)
    generate_reports(results, output_folder, config)


def main():
    parser = argparse.ArgumentParser(
        description="Generate model metrics violin plot and CSV"
    )
    parser.add_argument("input_folder", type=Path, help="Folder with model predictions")
    parser.add_argument("output_folder", type=Path, help="Folder to save plots/CSV")
    parser.add_argument("N", type=int, help="Number of models to ensemble")
    parser.add_argument(
        "model_names", type=str, nargs="+", help="Model name identifiers"
    )
    parser.add_argument(
        "--append_csv", type=str, help="Path to existing CSV to append results"
    )
    parser.add_argument(
        "--n_bootstrap", type=int, default=10000, help="Number of bootstrap samples"
    )
    parser.add_argument(
        "--bootstrap_method",
        type=str,
        choices=["standard", "stratified"],
        default="standard",
        help="Bootstrap method",
    )
    parser.add_argument(
        "--top_n", type=int, nargs="*", help="Top-N values for additional metrics"
    )
    parser.add_argument(
        "--discard_subjects_file",
        type=str,
        help="Optional .txt file of subject IDs to discard from OD",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=[
            "acc",
            "bacc",
            "roc_auc",
            "pr_auc",
            "mcc",
            "ece",
            "brier",
            "macro_f1",
            "f1",
        ],
        help="Metrics for tables and CSV (acc, bacc, roc_auc, pr_auc, mcc, ece, mce, brier, softmax_entropy, gini, renyi, f1, sens, prec, macro_f1, macro_prec, macro_sens)",
    )
    parser.add_argument(
        "--round", type=int, default=2, help="Decimal places for rounding metrics"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        help="Filter by dataset names (from 'Dataset' column). Supports exact match, wildcards (e.g., 'NACC*'), and regex patterns. If not specified, all datasets are included.",
    )
    parser.add_argument(
        "--save_ensemble",
        action="store_true",
        help="Save ensemble predictions to CSV files (ensemble_predictions_id.csv and ensemble_predictions_od.csv)",
    )
    args = parser.parse_args()

    run_analysis(
        args.input_folder,
        args.output_folder,
        args.N,
        args.model_names,
        args.append_csv,
        args.n_bootstrap,
        args.bootstrap_method,
        args.top_n,
        args.discard_subjects_file,
        args.metrics,
        args.round,
        args.datasets,
        args.save_ensemble,
    )


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

# Example usage:
# python visualizations/results/compute_metrics_plot_violin_csv.py \\
#     /path/to/saved_models/experiment/ \\
#     visualizations/outputs/metrics_results/ \\
#     1 \\
#     <run_id_1> <run_id_2> <...>
#
# To list run IDs from prediction files:
# ls /path/to/saved_models/experiment/prediction_*_od.csv | sed -E 's/.*_([a-zA-Z0-9]{8})_.*/\\1/' | sort | uniq
