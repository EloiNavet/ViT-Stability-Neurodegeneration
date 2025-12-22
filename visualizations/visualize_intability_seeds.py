"""Visualize training instability and seed variance across experiments."""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from tabulate import tabulate

CLASS_SAMPLE_SIZES = {
    "ID": {"CN": 1412, "AD": 654, "BV": 229, "PNFA": 66, "SD": 76, "FTD": 371},
    "OOD": {"CN": 2251, "AD": 485, "BV": 100, "PNFA": 43, "SD": 43, "FTD": 186},
}

DOMAIN_SAMPLE_SIZES = {
    domain: sum(classes.values()) for domain, classes in CLASS_SAMPLE_SIZES.items()
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "legend.frameon": True,
    }
)

# Mapping for human-readable legend labels (short abbreviations)
# Used for configuration comparison mode (dataaug, ema, mixup, etc.)
LEGEND_LABEL_MAPPING = {
    "baseline": ("BL", "Baseline"),
    "dataaug": ("DA", "Data Augmentation"),
    "ema": ("E", "EMA Weights"),
    "mixup": ("M", "MixUp"),
    "label_smoothing": ("LS", "Label Smoothing"),
    "balanced_sampling": ("BS", "Balanced Sampling"),
}

# Mapping for evaluation-specific legend labels
# Used for evaluation comparison mode (TTA, Calibrated, etc.)
EVAL_LEGEND_LABEL_MAPPING = {
    "Baseline": ("BL", "Baseline"),
    "TTA": ("TTA", "Test-Time Augmentation"),
    "Calibrated": ("Calibrated", "Temperature Scaling Calibration"),
    "TTA+Calibrated": ("TTA+Calibrated", "TTA + Calibration"),
    "Ensemble": ("Ensemble", "Ensemble (10 models)"),
}

# Mapping for F1 per-class labels (disease abbreviations)
F1_CLASS_LABEL_MAPPING = {
    "CN": "CN",
    "AD": "AD",
    "BV": "bvFTD",
    "PNFA": "nfvPPA",
    "SD": "svPPA",
    "FTD": "FTD",
}

# Mapping from model name prefixes to architecture display names
# Used when comparing multiple architectures (e.g., resnet vs swin vs vit)
ARCHITECTURE_PREFIX_MAPPING = {
    "resnet": "ResNet",
    "swin": "Swin",
    "swindpl": "Swin DPL",
    "vit": "ViT",
    "medvit": "MedViT",
    "svm": "SVM",
}

# Mapping for evaluation-specific groups (TTA, calibration, etc.)
# Keys are patterns to match in model names, values are display names
# Order matters: more specific patterns should come first
EVAL_GROUP_PATTERNS = [
    # Pattern: (contains_all, display_name, order_priority)
    # TTA + Calibrated (both present)
    (["tta", "calibrated"], "TTA+Calibrated", 3),
    # Only Calibrated (no tta in name OR tta folder prefix doesn't count)
    (["calibrated"], "Calibrated", 2),
    # Only TTA (tta/ prefix with no calibrated)
    (["tta/"], "TTA", 1),
]

# Desired order for eval groups (Baseline first, then TTA, Calibrated, TTA+Calibrated, Ensemble)
EVAL_GROUP_ORDER = ["Baseline", "TTA", "Calibrated", "TTA+Calibrated", "Ensemble"]


def format_legend_label(label, use_short=True):
    """
    Convert internal group names to human-readable legend labels.
    Handles compound names like 'dataaug-ema-label_smoothing' by splitting
    on '-' and mapping each component.

    Args:
        label: Internal label name
        use_short: If True, use short abbreviations (e.g., "DA+E+LS")
                   If False, use full names (e.g., "Data Augmentation + EMA + Label Smoothing")
    """
    # If the label is an architecture name (from ARCHITECTURE_PREFIX_MAPPING values),
    # return it as-is without any transformation
    if label in ARCHITECTURE_PREFIX_MAPPING.values():
        return label

    # If the label is an evaluation group name, use EVAL_LEGEND_LABEL_MAPPING
    if label in EVAL_LEGEND_LABEL_MAPPING:
        idx = 0 if use_short else 1
        return EVAL_LEGEND_LABEL_MAPPING[label][idx]

    # Split on '-' to handle compound names
    parts = label.split("-")
    formatted_parts = []
    for part in parts:
        if part in LEGEND_LABEL_MAPPING:
            idx = 0 if use_short else 1
            formatted_parts.append(LEGEND_LABEL_MAPPING[part][idx])
        else:
            # Capitalize unknown parts
            formatted_parts.append(part.replace("_", " ").title())

    separator = "+" if use_short else " + "
    return separator.join(formatted_parts)


def get_legend_footnote(mode="configuration"):
    """
    Generate a footnote explaining the abbreviations used in the legend.

    Args:
        mode: 'configuration' for dataaug/ema/mixup legends,
              'evaluation' for TTA/Calibrated legends,
              'architecture' for no footnote needed
    """
    if mode == "evaluation":
        abbrevs = [f"{v[0]}={v[1]}" for v in EVAL_LEGEND_LABEL_MAPPING.values()]
    elif mode == "configuration":
        abbrevs = [f"{v[0]}={v[1]}" for v in LEGEND_LABEL_MAPPING.values()]
    else:
        # Architecture mode - no abbreviations needed
        return ""
    return "  |  ".join(abbrevs)


def get_group_category(group_name):
    """
    Classify a group into one of three categories:
    - 'baseline': The baseline group
    - 'single': Single modification (e.g., 'dataaug', 'ema', 'mixup')
    - 'combined': Multiple modifications combined (e.g., 'dataaug-ema')

    Returns category name and order within category.
    """
    if group_name == "baseline":
        return "baseline"

    # Count the number of components (separated by '-')
    # Single modifications have only one component
    parts = group_name.split("-")

    # Check if all parts are known single modifications
    single_mods = {"dataaug", "ema", "mixup", "label_smoothing", "balanced_sampling"}

    if len(parts) == 1 and parts[0] in single_mods:
        return "single"
    else:
        return "combined"


def extract_architecture_from_name(model_name):
    """
    Extract architecture name from model name using ARCHITECTURE_PREFIX_MAPPING.

    Args:
        model_name: Full model name (e.g., "resnet-5c-no_seed-baseline-1/...")

    Returns:
        Tuple of (architecture_display_name, prefix_used) or (None, None) if no match
    """
    # Get the base name before any "/" or run ID
    if "/" in model_name:
        base = model_name.split("/")[0]
    else:
        base = model_name

    base_lower = base.lower()

    # Sort prefixes by length (longest first) to match more specific prefixes first
    sorted_prefixes = sorted(ARCHITECTURE_PREFIX_MAPPING.keys(), key=len, reverse=True)

    for prefix in sorted_prefixes:
        if base_lower.startswith(prefix):
            return ARCHITECTURE_PREFIX_MAPPING[prefix], prefix

    return None, None


def detect_eval_group(model_name):
    """
    Detect if a model name corresponds to an evaluation group (TTA, Calibrated, Ensemble, etc.).

    Args:
        model_name: Full model name from CSV

    Returns:
        Display name for the eval group, or None if not an eval pattern
    """
    import re

    name_lower = model_name.lower()

    # Check for ensemble first: pattern like "(10, Top-1)" where number > 1
    # This indicates an ensemble of multiple models
    ensemble_match = re.search(r"\((\d+),\s*Top-\d+\)", model_name)
    is_ensemble = False
    if ensemble_match:
        num_models = int(ensemble_match.group(1))
        is_ensemble = num_models > 1

    # Check for TTA+Calibrated first (most specific)
    # Pattern: has both "tta" in the path AND "calibrated" in the config name
    has_tta_prefix = name_lower.startswith("tta/")
    has_tta_in_config = "-tta-" in name_lower and not has_tta_prefix
    has_calibrated = "calibrated" in name_lower

    if has_tta_prefix and has_calibrated:
        # tta/ folder with calibrated model -> TTA+Calibrated
        return "TTA+Calibrated"
    elif has_tta_in_config and has_calibrated:
        # -tta- in config name with calibrated -> TTA+Calibrated
        return "TTA+Calibrated"
    elif has_calibrated and not has_tta_prefix and not has_tta_in_config:
        # Only calibrated, no TTA
        return "Calibrated"
    elif has_tta_prefix and not has_calibrated:
        # Only TTA (tta/ prefix, no calibrated)
        return "TTA"
    elif is_ensemble:
        # Ensemble of multiple models (no TTA, no calibrated)
        return "Ensemble"

    return None


def detect_comparison_mode(model_names):
    """
    Detect whether we're comparing:
    - Multiple architectures (e.g., ResNet vs Swin vs ViT)
    - Evaluation variants (TTA, Calibrated, etc.)
    - Multiple configurations of same architecture (e.g., baseline vs dataaug vs ema)

    Returns:
        'architecture' if comparing different architectures
        'evaluation' if comparing TTA/Calibration variants
        'configuration' if comparing configurations of same architecture
    """
    architectures_found = set()
    eval_groups_found = set()

    for name in model_names:
        arch, _ = extract_architecture_from_name(name)
        if arch:
            architectures_found.add(arch)

        eval_group = detect_eval_group(name)
        if eval_group:
            eval_groups_found.add(eval_group)

    # If we found multiple architectures, we're in architecture comparison mode
    if len(architectures_found) > 1:
        return "architecture", architectures_found
    # If we found eval groups (TTA, Calibrated, etc.), we're in evaluation mode
    elif eval_groups_found:
        return "evaluation", eval_groups_found
    else:
        return "configuration", architectures_found


def find_common_prefix(strings):
    if not strings:
        return ""

    sorted_strings = sorted(strings)
    first = sorted_strings[0]
    last = sorted_strings[-1]

    common = []
    for i, char in enumerate(first):
        if i < len(last) and char == last[i]:
            common.append(char)
        else:
            break

    return "".join(common)


def extract_model_groups(model_names):
    """
    Extract model groups from model names.

    Supports three modes:
    1. Architecture comparison: When models have different architecture prefixes
    (e.g., resnet-..., swin-..., vit-...) -> Groups by architecture name
    2. Evaluation comparison: When comparing TTA, Calibration variants
       (e.g., tta/, -calibrated-, -tta-calibrated-) -> Groups by eval method
    3. Configuration comparison: When all models have same architecture
    (e.g., swin-5c-baseline, swin-5c-dataaug) -> Groups by config suffix

    Returns:
        Tuple of (group_mapping, unique_groups_ordered, comparison_mode) where:
        - group_mapping: Dict mapping full model name to group name
        - unique_groups_ordered: List of unique groups in order of first appearance
        - comparison_mode: One of 'architecture', 'evaluation', or 'configuration'
    """
    # First, detect which comparison mode we're in
    comparison_mode, groups_found = detect_comparison_mode(model_names)

    if comparison_mode == "architecture":
        # Architecture comparison mode: group by architecture display name
        print(
            f"\n[Architecture Comparison Mode] Detected {len(groups_found)} architectures:"
        )
        for arch in sorted(groups_found):
            print(f"  - {arch}")

        group_mapping = {}
        for name in model_names:
            arch, _ = extract_architecture_from_name(name)
            if arch:
                group_mapping[name] = arch
            else:
                # Fallback: use the part before first '-'
                if "/" in name:
                    base = name.split("/")[0]
                else:
                    base = name
                fallback = base.split("-")[0].title()
                group_mapping[name] = fallback
                print(
                    f"  Warning: No architecture mapping for '{base}', using '{fallback}'"
                )

        # Preserve order of first appearance
        unique_groups = list(dict.fromkeys(group_mapping.values()))

        print(f"\nGrouped into {len(unique_groups)} architecture groups:")
        for group in unique_groups:
            count = sum(1 for v in group_mapping.values() if v == group)
            print(f"  - '{group}' ({count} models)")

        return group_mapping, unique_groups, "architecture"

    elif comparison_mode == "evaluation":
        # Evaluation comparison mode: group by TTA/Calibration variant
        print(
            f"\n[Evaluation Comparison Mode] Detected {len(groups_found)} evaluation variants:"
        )
        for variant in sorted(groups_found):
            print(f"  - {variant}")

        group_mapping = {}

        for name in model_names:
            eval_group = detect_eval_group(name)
            if eval_group:
                group_mapping[name] = eval_group
            else:
                # No eval pattern detected -> this is the baseline
                group_mapping[name] = "Baseline"

        # Use predefined order for eval groups
        unique_groups = []
        for group in EVAL_GROUP_ORDER:
            if group in group_mapping.values():
                unique_groups.append(group)

        # Add any groups not in the predefined order (shouldn't happen, but just in case)
        for group in dict.fromkeys(group_mapping.values()):
            if group not in unique_groups:
                unique_groups.append(group)

        print(f"\nGrouped into {len(unique_groups)} evaluation groups:")
        for group in unique_groups:
            count = sum(1 for v in group_mapping.values() if v == group)
            print(f"  - '{group}' ({count} models)")

        return group_mapping, unique_groups, "evaluation"

    else:
        # Configuration comparison mode: original behavior
        print("\n[Configuration Comparison Mode]")

        base_names = []
        for name in model_names:
            if "/" in name:
                base = name.split("/")[0]
            else:
                base = name

            base = re.sub(r"-\d+$", "", base)
            base_names.append(base)

        unique_bases = list(set(base_names))

        if len(unique_bases) < 2:
            print(
                f"Warning: Found only {len(unique_bases)} unique model group(s). Need at least 2 for comparison."
            )
            return None

        common_prefix = find_common_prefix(unique_bases)

        group_mapping = {}
        for name in model_names:
            if "/" in name:
                base = name.split("/")[0]
            else:
                base = name
            base = re.sub(r"-\d+$", "", base)

            if common_prefix:
                group_name = base[len(common_prefix) :].lstrip("-_")
            else:
                group_name = base

            if not group_name:
                group_name = "baseline"

            group_mapping[name] = group_name

        # Preserve order of first appearance in CSV (dict preserves insertion order in Python 3.7+)
        unique_groups = list(dict.fromkeys(group_mapping.values()))
        print(f"\nDetected {len(unique_groups)} model groups:")
        for group in unique_groups:
            count = sum(1 for v in group_mapping.values() if v == group)
            print(f"  - '{group}' ({count} models)")

        return group_mapping, unique_groups, "configuration"


def compute_aggregated_metrics(df_clean, group_mapping, all_metrics):
    """
    Compute aggregated metrics (mean and 95% CI) for each architecture group across seeds.

    Args:
        df_clean: DataFrame with model results (excluding CI rows)
        group_mapping: Dict mapping model names to architecture groups
        all_metrics: List of metric column names

    Returns:
        DataFrame with aggregated results
    """
    df_clean["Architecture"] = df_clean["Model Name"].map(group_mapping)

    # Group by architecture and compute statistics
    agg_results = []

    for arch in sorted(set(group_mapping.values())):
        arch_data = df_clean[df_clean["Architecture"] == arch]

        if arch_data.empty:
            continue

        row_data = {"Model Name": arch}

        for metric in all_metrics:
            if metric not in arch_data.columns:
                continue

            values = arch_data[metric].dropna()

            if len(values) == 0:
                row_data[metric] = ""
                continue

            # Calculate mean
            mean_val = values.mean()

            # Calculate 95% CI using t-distribution
            if len(values) > 1:
                stderr = values.sem()
                ci = stderr * stats.t.ppf((1 + 0.95) / 2, len(values) - 1)
                lower = mean_val - ci
                upper = mean_val + ci
                # Format as "mean [lower-upper]"
                row_data[metric] = f"{mean_val:.2f} [{lower:.2f}-{upper:.2f}]"
            else:
                # Single value, no CI
                row_data[metric] = f"{mean_val:.2f}"

        agg_results.append(row_data)

    return pd.DataFrame(agg_results)


def analyze_std_instability(
    file_path, output_format="png", no_title=False, no_sublegend=False
):
    file_path = Path(file_path)

    df = pd.read_csv(file_path)

    if "Model Name" not in df.columns:
        print("Error: 'Model Name' column not found in CSV.")
        sys.exit(1)

    df_clean = df[
        (~df["Model Name"].fillna("").str.contains(r"\[95% CI\]", regex=True))
        & (df["Model Name"] != "Model Name")  # Filter duplicate header rows
    ].copy()

    if df_clean.empty:
        print("Warning: DataFrame is empty after filtering [95% CI] rows.")
        return

    model_names = df_clean["Model Name"].tolist()
    result = extract_model_groups(model_names)

    if result is None:
        print("Error: Could not extract model groups for comparison.")
        return

    group_mapping, unique_groups_ordered, comparison_mode = result

    df_clean["Task"] = df_clean["Model Name"].map(group_mapping)

    metrics_id = [
        "ID-ACC",
        # "ID-BACC",
        "ID-MCC",
        # "ID-ROC-AUC",
        "ID-PR-AUC",
        "ID-MACRO-F1",
        "ID-ECE",
        "ID-BRIER",
    ]
    metrics_od = [
        "OD-ACC",
        # "OD-BACC",
        "OD-MCC",
        # "OD-ROC-AUC",
        "OD-PR-AUC",
        "OD-MACRO-F1",
        "OD-ECE",
        "OD-BRIER",
    ]

    # Per-class F1 scores (separate figure)
    f1_id_columns = [col for col in df_clean.columns if col.startswith("ID-F1:")]
    f1_od_columns = [col for col in df_clean.columns if col.startswith("OD-F1:")]
    f1_metrics = f1_id_columns + f1_od_columns

    all_metrics = metrics_id + metrics_od

    error_metrics = ["ID-ECE", "ID-BRIER", "OD-ECE", "OD-BRIER"]

    for col in list(all_metrics):
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        else:
            print(f"Warning: Metric '{col}' not found in CSV. Skipping.")
            all_metrics.remove(col)

    # Convert F1 metrics to numeric and filter out empty columns
    for col in list(f1_metrics):
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            # Remove columns that are entirely NaN (no data for this class)
            if df_clean[col].isna().all():
                print(f"Warning: F1 metric '{col}' has no data. Skipping.")
                f1_metrics.remove(col)
        else:
            print(f"Warning: F1 metric '{col}' not found in CSV. Skipping.")
            f1_metrics.remove(col)

    std_df = df_clean.groupby("Task")[all_metrics].std()
    mean_df = df_clean.groupby("Task")[all_metrics].mean()

    cv_df = std_df / mean_df.replace(0, float("nan"))

    # Use the preserved order from CSV appearance
    unique_groups = unique_groups_ordered
    if len(unique_groups) < 2:
        print(
            f"Error: Found only {len(unique_groups)} group(s) in the data. Need at least 2 for comparison."
        )
        return

    print(f"\nComparing {len(unique_groups)} model groups: {', '.join(unique_groups)}")

    display_data = []
    for m in all_metrics:
        if m not in cv_df.columns:
            continue

        cv_values = {group: cv_df.loc[group, m] for group in unique_groups}

        sorted_groups = sorted(cv_values.items(), key=lambda x: x[1])

        stability_ranking = " > ".join([g[0] for g in sorted_groups])

        clean_name = m.replace("ID-", "").replace("OD-", "")
        if m in error_metrics:
            clean_name += " (↓)"

        row = {
            "Domain": "ID" if "ID-" in m else "OOD",
            "Metric": clean_name,
        }

        for group in unique_groups:
            row[f"CV {group}"] = cv_values[group]

        row["Stability Ranking"] = stability_ranking

        display_data.append(row)

    disp_df = pd.DataFrame(display_data)

    print("\n" + "=" * 80)
    print(f"{'INSTABILITY ANALYSIS (Coefficient of Variation)':^80}")
    print(f"{file_path.name:^80}")
    print("=" * 80)

    for domain in ["ID", "OOD"]:
        print(f"\n--- {domain} ---")
        subset = disp_df[disp_df["Domain"] == domain].drop(columns=["Domain"])
        if not subset.empty:
            print(
                tabulate(
                    subset,
                    headers="keys",
                    tablefmt="fancy_grid",
                    floatfmt=".4f",
                    showindex=False,
                )
            )
        else:
            print("No metrics found for this domain.")

    if cv_df.empty or mean_df.empty or std_df.empty:
        print("No data available for plotting.")
        return

    # Prepare data for mean, std, and CV
    mean_plot_df = mean_df.reset_index().melt(
        id_vars="Task", value_vars=all_metrics, var_name="Metric", value_name="Mean"
    )
    std_plot_df = std_df.reset_index().melt(
        id_vars="Task", value_vars=all_metrics, var_name="Metric", value_name="Std"
    )
    cv_plot_df = cv_df.reset_index().melt(
        id_vars="Task", value_vars=all_metrics, var_name="Metric", value_name="CV"
    )

    # Add domain and base metric info to all dataframes
    for df in [mean_plot_df, std_plot_df, cv_plot_df]:
        df["Domain"] = df["Metric"].apply(lambda x: "ID" if "ID-" in x else "OOD")
        df["Base Metric"] = df["Metric"].apply(
            lambda x: x.replace("ID-", "").replace("OD-", "")
        )
        # Simplified metric labels without "(Error)" annotation
        df["Metric Label"] = df["Base Metric"]

    # Publication-ready style setup
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.1)

    # Use order from CSV appearance (unique_groups_ordered is already in correct order)
    if len(unique_groups_ordered) <= 10:
        color_palette = sns.color_palette("colorblind", len(unique_groups_ordered))
    else:
        color_palette = sns.color_palette("husl", len(unique_groups_ordered))
    colors = dict(zip(unique_groups_ordered, color_palette))

    # Define output formats
    formats_to_save = ["png"]
    if output_format == "pdf":
        formats_to_save.append("pdf")

    def create_publication_figure(
        data_df,
        value_col,
        title,
        ylabel,
        filename_suffix,
        use_log_scale=False,
        is_f1_plot=False,
    ):
        """Create a publication-ready figure with ID and OOD panels.

        Args:
            use_log_scale: If True, use logarithmic scale on Y-axis for better visualization
                           of data with large dynamic range.
            is_f1_plot: If True, don't rotate x-axis labels (for per-class F1 plots)
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=False)

        # Reduce spacing between subplots
        plt.subplots_adjust(wspace=0.15)

        id_data = data_df[data_df["Domain"] == "ID"].copy()
        ood_data = data_df[data_df["Domain"] == "OOD"].copy()

        legend_handles = []
        legend_labels = []

        # Calculate position offsets for each group based on category
        # We want gaps between: baseline | single mods | combined mods
        gap_size = 0.4  # Size of gap between categories (in bar width units)

        # Build offset mapping based on group categories
        group_offsets = {}
        current_offset = 0.0
        prev_category = None

        for group in unique_groups_ordered:
            category = get_group_category(group)
            if prev_category is not None and category != prev_category:
                current_offset += gap_size
            group_offsets[group] = current_offset
            prev_category = category

        def adjust_bar_positions(ax):
            """Adjust bar positions to add gaps between categories."""
            # Get all bar containers
            containers = ax.containers
            if not containers:
                return

            n_metrics = len(containers[0]) if containers else 0

            if n_metrics == 0:
                return

            # Calculate bar width from first container
            first_bar = containers[0][0] if containers[0] else None
            if first_bar is None:
                return
            bar_width = first_bar.get_width()

            # Adjust each bar's position
            for group_idx, group in enumerate(unique_groups_ordered):
                if group_idx >= len(containers):
                    continue
                container = containers[group_idx]
                offset = group_offsets[group] * bar_width

                for bar in container:
                    bar.set_x(bar.get_x() + offset)

            # Adjust x-axis limits to accommodate the gaps
            total_offset = (
                max(group_offsets.values()) * bar_width if group_offsets else 0
            )
            xlim = ax.get_xlim()
            ax.set_xlim(xlim[0], xlim[1] + total_offset)

        # Plot ID (left panel)
        ax_id = axes[0]
        if not id_data.empty:
            sns.barplot(
                data=id_data,
                x="Metric Label",
                y=value_col,
                hue="Task",
                hue_order=unique_groups_ordered,
                palette=colors,
                ax=ax_id,
                edgecolor="none",
                linewidth=0,
            )
            ax_id.set_title("In-domain", fontsize=11)
            ax_id.set_ylabel(ylabel, fontsize=10)
            ax_id.set_xlabel("")
            # Rotate labels only for non-F1 plots
            rotation = 0 if is_f1_plot else 45
            ax_id.tick_params(axis="x", rotation=rotation, labelsize=9)
            ax_id.tick_params(axis="y", labelsize=9)

            # Store legend handles
            if ax_id.get_legend():
                legend_handles, legend_labels = ax_id.get_legend_handles_labels()
                ax_id.get_legend().remove()

            # Add grid for better readability
            ax_id.yaxis.grid(True, linestyle="--", alpha=0.7)
            ax_id.set_axisbelow(True)

            # Apply log scale if requested
            if use_log_scale:
                ax_id.set_yscale("log")

            # Adjust bar positions to add gaps between categories
            adjust_bar_positions(ax_id)
        else:
            ax_id.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=11)
            ax_id.set_title("In-domain", fontsize=11)

        # Plot OOD (right panel)
        ax_ood = axes[1]
        if not ood_data.empty:
            sns.barplot(
                data=ood_data,
                x="Metric Label",
                y=value_col,
                hue="Task",
                hue_order=unique_groups_ordered,
                palette=colors,
                ax=ax_ood,
                edgecolor="none",
                linewidth=0,
            )
            ax_ood.set_title("Out-of-domain", fontsize=11)
            ax_ood.set_ylabel("")
            ax_ood.set_xlabel("")
            # Rotate labels only for non-F1 plots
            rotation = 0 if is_f1_plot else 45
            ax_ood.tick_params(axis="x", rotation=rotation, labelsize=9)
            ax_ood.tick_params(axis="y", labelsize=9)

            if ax_ood.get_legend():
                ax_ood.get_legend().remove()

            # Add grid
            ax_ood.yaxis.grid(True, linestyle="--", alpha=0.7)
            ax_ood.set_axisbelow(True)

            # Apply log scale if requested
            if use_log_scale:
                ax_ood.set_yscale("log")

            # Adjust bar positions to add gaps between categories
            adjust_bar_positions(ax_ood)
        else:
            ax_ood.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=11)
            ax_ood.set_title("Out-of-domain", fontsize=11)

        # Share y-axis limits between ID and OOD for better comparison
        if not id_data.empty and not ood_data.empty:
            y_min = min(ax_id.get_ylim()[0], ax_ood.get_ylim()[0])
            y_max = max(ax_id.get_ylim()[1], ax_ood.get_ylim()[1])

            if use_log_scale:
                # For log scale, ensure positive limits and add multiplicative padding
                y_min = max(y_min, 1e-3)  # Ensure positive minimum
                ax_id.set_ylim(y_min * 0.8, y_max * 1.2)
                ax_ood.set_ylim(y_min * 0.8, y_max * 1.2)
            else:
                # Add some padding for linear scale
                y_range = y_max - y_min
                ax_id.set_ylim(y_min - 0.02 * y_range, y_max + 0.05 * y_range)
                ax_ood.set_ylim(y_min - 0.02 * y_range, y_max + 0.05 * y_range)

        # Add common legend at bottom with formatted labels
        if legend_handles:
            # Format legend labels for publication (short form)
            formatted_labels = [
                format_legend_label(lbl, use_short=True) for lbl in legend_labels
            ]

            # Put all items on a single row
            n_items = len(legend_handles)
            ncol = n_items  # All items on one line

            fig.legend(
                handles=legend_handles,
                labels=formatted_labels,
                loc="lower center",
                ncol=ncol,
                bbox_to_anchor=(0.5, -0.02),
                frameon=True,
                fancybox=False,
                edgecolor="black",
                fontsize=9,
                columnspacing=0.8,
                handletextpad=0.3,
                handlelength=1.2,
            )

            # Add footnote with abbreviation explanations (unless disabled)
            if not no_sublegend:
                footnote = get_legend_footnote(mode=comparison_mode)
                fig.text(
                    0.5,
                    -0.08,
                    footnote,
                    ha="center",
                    va="top",
                    fontsize=7,
                    style="italic",
                    transform=fig.transFigure,
                )

        # Determine layout rect based on options
        bottom_margin = 0.02 if no_sublegend else 0.08

        # Main title (only if not disabled)
        if not no_title:
            fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
            plt.tight_layout(rect=[0, bottom_margin, 1, 0.96])
        else:
            plt.tight_layout(rect=[0, bottom_margin, 1, 1.0])

        # Save in all requested formats
        for fmt in formats_to_save:
            output_path = file_path.parent / f"{file_path.stem}_{filename_suffix}.{fmt}"
            plt.savefig(
                output_path,
                bbox_inches="tight",
                dpi=300,
                facecolor="white",
                edgecolor="none",
                format=fmt,
                pad_inches=0.02,  # Minimal padding
            )
            print(f"  Saved: {output_path}")

        plt.close(fig)

    # Generate 3 separate figures
    print("\n" + "-" * 40)
    print("Generating publication-ready figures...")
    print("-" * 40)

    print("\n1. Mean Values:")
    create_publication_figure(
        mean_plot_df, "Mean", "Mean Performance Across Seeds", "Mean Value", "mean"
    )

    print("\n2. Standard Deviation:")
    create_publication_figure(
        std_plot_df,
        "Std",
        "Standard Deviation Across Seeds",
        "Standard Deviation",
        "std",
    )

    print("\n3. Coefficient of Variation (CV = σ/μ):")
    create_publication_figure(
        cv_plot_df,
        "CV",
        "Coefficient of Variation Across Seeds",
        "CV (lower = more stable)",
        "cv",
    )

    # Compute normalized CV for grouped metrics by domain sample size
    # CV_normalized = CV * sqrt(N_d) where N_d is the total number of subjects in the domain
    def normalize_cv_by_domain_size(row):
        """Normalize CV by sqrt(domain sample size) to account for dataset size differences."""
        domain = row["Domain"]
        cv = row["CV"]

        if domain in DOMAIN_SAMPLE_SIZES:
            n_d = DOMAIN_SAMPLE_SIZES[domain]
            return cv * (n_d**0.5)
        else:
            print(f"Warning: Domain '{domain}' not found in DOMAIN_SAMPLE_SIZES")
            return cv

    cv_normalized_plot_df = cv_plot_df.copy()
    cv_normalized_plot_df["CV_Normalized"] = cv_normalized_plot_df.apply(
        normalize_cv_by_domain_size, axis=1
    )

    print("\n4. Normalized CV by Domain Size (CV × √N_domain) - Log Scale:")
    create_publication_figure(
        cv_normalized_plot_df,
        "CV_Normalized",
        "Normalized CV Across Seeds (CV × √N_domain)",
        "Normalized CV (lower = more stable)",
        "cv_normalized",
        use_log_scale=True,
    )

    # Generate F1 per-class figures if F1 metrics exist
    if f1_metrics:
        print("\n" + "-" * 40)
        print("Generating per-class F1 figures...")
        print("-" * 40)

        # Compute stats for F1 metrics
        f1_std_df = df_clean.groupby("Task")[f1_metrics].std()
        f1_mean_df = df_clean.groupby("Task")[f1_metrics].mean()
        f1_cv_df = f1_std_df / f1_mean_df.replace(0, float("nan"))

        # Prepare F1 data for plotting
        f1_mean_plot_df = f1_mean_df.reset_index().melt(
            id_vars="Task", value_vars=f1_metrics, var_name="Metric", value_name="Mean"
        )
        f1_std_plot_df = f1_std_df.reset_index().melt(
            id_vars="Task", value_vars=f1_metrics, var_name="Metric", value_name="Std"
        )
        f1_cv_plot_df = f1_cv_df.reset_index().melt(
            id_vars="Task", value_vars=f1_metrics, var_name="Metric", value_name="CV"
        )

        # Add domain and class info to F1 dataframes
        for df in [f1_mean_plot_df, f1_std_plot_df, f1_cv_plot_df]:
            df["Domain"] = df["Metric"].apply(lambda x: "ID" if "ID-" in x else "OOD")
            # Extract original class name (e.g., "ID-F1:CN" -> "CN") - keep for lookup
            df["Original Class"] = df["Metric"].apply(
                lambda x: x.split(":")[1]
                if ":" in x
                else x.replace("ID-", "").replace("OD-", "")
            )
            # Map to display name for visualization
            df["Metric Label"] = df["Original Class"].apply(
                lambda x: F1_CLASS_LABEL_MAPPING.get(x, x)
            )

        print("\n5. Per-Class F1 Mean Values:")
        create_publication_figure(
            f1_mean_plot_df,
            "Mean",
            "Per-Class F1 Mean Across Seeds",
            "Mean F1 Score",
            "f1_mean",
            is_f1_plot=True,
        )

        print("\n6. Per-Class F1 Standard Deviation:")
        create_publication_figure(
            f1_std_plot_df,
            "Std",
            "Per-Class F1 Standard Deviation Across Seeds",
            "Standard Deviation",
            "f1_std",
            is_f1_plot=True,
        )

        print("\n7. Per-Class F1 Coefficient of Variation:")
        create_publication_figure(
            f1_cv_plot_df,
            "CV",
            "Per-Class F1 Coefficient of Variation Across Seeds",
            "CV (lower = more stable)",
            "f1_cv",
            is_f1_plot=True,
        )

        print(
            "\n8. Per-Class F1 Normalized CV (CV × √N_class × √N_domain) - Log Scale:"
        )

        # Double normalization: by class size AND domain size
        def normalize_f1_cv_double(row):
            """Normalize F1 CV by sqrt(class size) × sqrt(domain size)."""
            domain = row["Domain"]
            original_class_name = row["Original Class"]  # Use original name for lookup
            cv = row["CV"]

            domain_key = "ID" if domain == "ID" else "OOD"

            n_class = 1  # default if not found
            n_domain = 1  # default if not found

            if (
                domain_key in CLASS_SAMPLE_SIZES
                and original_class_name in CLASS_SAMPLE_SIZES[domain_key]
            ):
                n_class = CLASS_SAMPLE_SIZES[domain_key][original_class_name]
            else:
                print(
                    f"Warning: Class '{original_class_name}' not found in CLASS_SAMPLE_SIZES for domain '{domain_key}'"
                )

            if domain_key in DOMAIN_SAMPLE_SIZES:
                n_domain = DOMAIN_SAMPLE_SIZES[domain_key]
            else:
                print(
                    f"Warning: Domain '{domain_key}' not found in DOMAIN_SAMPLE_SIZES"
                )

            return cv * (n_class**0.5) * (n_domain**0.5)

        f1_cv_normalized_plot_df = f1_cv_plot_df.copy()
        f1_cv_normalized_plot_df["CV_Normalized"] = f1_cv_normalized_plot_df.apply(
            normalize_f1_cv_double, axis=1
        )

        create_publication_figure(
            f1_cv_normalized_plot_df,
            "CV_Normalized",
            "Per-Class F1 Normalized CV Across Seeds (CV × √N_class × √N_domain)",
            "Normalized CV (lower = more stable)",
            "f1_cv_normalized",
            use_log_scale=True,
            is_f1_plot=True,
        )
    else:
        print("\nNo per-class F1 metrics found in CSV.")

    # Generate aggregated CSV with mean and 95% CI per architecture
    print("\n" + "=" * 80)
    print("Generating aggregated metrics CSV...")
    print("=" * 80)

    agg_df = compute_aggregated_metrics(df_clean, group_mapping, all_metrics)

    if not agg_df.empty:
        csv_output_path = file_path.parent / (file_path.stem + "_aggregated.csv")
        agg_df.to_csv(csv_output_path, index=False)
        print(f"\nAggregated CSV saved to: {csv_output_path}")
        print(f"Total architectures: {len(agg_df)}")
    else:
        print("Warning: No aggregated data to save.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze stability (Coefficient of Variation) of ID vs OOD metrics."
    )
    parser.add_argument("file_path", type=Path, help="Path to CSV file.")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["png", "pdf"],
        default="png",
        help="Output format for figures. 'png' saves only PNG (300 dpi), "
        "'pdf' saves both PNG (300 dpi) and PDF.",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Generate figures without main title (for publication).",
    )
    parser.add_argument(
        "--no-sublegend",
        action="store_true",
        help="Remove abbreviation footnote below legend (to put in LaTeX caption instead).",
    )

    args = parser.parse_args()

    analyze_std_instability(
        args.file_path,
        output_format=args.output_format,
        no_title=args.no_title,
        no_sublegend=args.no_sublegend,
    )

# Example:
# python visualizations/results/visualize_intability_seeds.py visualizations/outputs/midl_results_all_3c5s.csv --no-title --no-sublegend --output-format pdf
# python visualizations/results/visualize_intability_seeds.py visualizations/outputs/midl_results_all_5c5s.csv --no-title --no-sublegend --output-format pdf
# python visualizations/results/visualize_intability_seeds.py visualizations/outputs/midl_adni_nifd_results_5c.csv --no-title --no-sublegend --output-format pdf
