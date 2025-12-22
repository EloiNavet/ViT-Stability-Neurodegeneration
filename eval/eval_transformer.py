"""Evaluate trained transformer checkpoints with optional TTA and bootstrap CI."""

import argparse
import json
import logging
import os
import random
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb as w
import yaml
from monai.transforms import Compose, NormalizeIntensity, Resize
from monai.utils import set_determinism as monai_set_determinism
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))
from dataset.dataset import NormalDataset
from dataset.preprocessing import DataPrepa
from eval.test_time_augmentation import create_tta_predictor
from utils import (
    _MAX_UINT32,
    compute_bootstrap_metrics,
    dir_path,
    file_path,
    get_train_val_test,
    normalize_seed,
    seed_everything,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate transformer checkpoints.")

    data_group = parser.add_argument_group("Data paths")
    data_group.add_argument(
        "--training-csv-dir",
        type=dir_path,
        required=True,
        help="Directory containing k-fold CSV files.",
    )
    data_group.add_argument(
        "--intermediate-dir",
        type=dir_path,
        required=True,
        help="Directory containing or storing preprocessed tensors.",
    )
    data_group.add_argument(
        "--eval-csv",
        type=file_path,
        default=None,
        nargs="?",
        help="Optional CSV file for out-of-domain evaluation.",
    )
    data_group.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="One or more checkpoint paths to evaluate.",
    )

    hardware_group = parser.add_argument_group("Hardware")
    hardware_group.add_argument(
        "--cuda-device",
        type=str,
        default="0",
        help="CUDA device index to use for evaluation.",
    )

    run_group = parser.add_argument_group("Run options")
    run_group.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Evaluation batch size (per device).",
    )
    run_group.add_argument(
        "--use-amp",
        action="store_true",
        help="Use automatic mixed precision (FP16) for faster inference.",
    )
    run_group.add_argument(
        "--force-eval",
        action="store_true",
        help="Re-run evaluation even if prediction files already exist.",
    )
    run_group.add_argument(
        "--log-to-wandb",
        action="store_true",
        help="Log evaluation metrics to W&B instead of just writing to file.",
    )
    run_group.add_argument(
        "--project-name",
        type=str,
        help="W&B project name (used when --log-to-wandb is enabled).",
    )

    tta_group = parser.add_argument_group("Test-Time Augmentation (TTA)")
    tta_group.add_argument(
        "--use-tta",
        action="store_true",
        help="Enable test-time augmentation for more robust predictions.",
    )
    tta_group.add_argument(
        "--tta-num-samples",
        type=int,
        default=5,
        help="Number of augmented samples per input for TTA (default: 5).",
    )
    tta_group.add_argument(
        "--tta-no-flip",
        action="store_true",
        help="Disable sagittal (L/R) flip in TTA (enabled by default).",
    )
    tta_group.add_argument(
        "--tta-no-affine",
        action="store_true",
        help="Disable small affine perturbations in TTA (enabled by default).",
    )
    tta_group.add_argument(
        "--tta-no-center-crop",
        action="store_true",
        help="Disable center crop augmentation in TTA (enabled by default).",
    )
    tta_group.add_argument(
        "--tta-crop-scale",
        type=float,
        default=0.9,
        help="Crop ROI scale factor for center-crop TTA (default: 0.9).",
    )

    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Relative subfolder (within checkpoint directory) to save results. "
        "E.g., 'tta' will save to <checkpoint_dir>/tta/. Default: same as checkpoint directory.",
    )

    return parser.parse_args()


def setup_logger(model_path: Path, output_dir: Path) -> logging.Logger:
    """Setup logger for evaluation results.

    Args:
        model_path: Path to the model checkpoint (used for naming).
        output_dir: Directory where log files will be saved.
    """
    logger = logging.getLogger(model_path.stem)
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    logger.setLevel(logging.INFO)
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    logfile = output_dir / f"results_{model_path.stem}.txt"
    file_handler = logging.FileHandler(logfile)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def build_model(device: torch.device) -> nn.Module:
    if w.config.ARCHITECTURE == "Swin":
        from models.swin_transformer_3d import SwinTransformerT as SwinTransformer

        model = SwinTransformer(
            in_channels=1,
            patch_size=tuple(w.config.PATCH_SHAPE),
            embed_dim=w.config.EMBED_DIM,
            depths=w.config.DEPTH,
            num_heads=w.config.HEADS,
            window_size=tuple(w.config.WINDOW_SIZE),
            mlp_ratio=w.config.MLP_RATIO,
            qkv_bias=w.config.QKV_BIAS,
            dropout=w.config.DROPOUT,
            attention_dropout=w.config.ATTENTION_DROPOUT,
            stochastic_depth_prob=w.config.STOCHASTIC_DEPTH_PROB,
            num_classes=len(w.config.DISEASES),
            norm_layer=(eval(w.config.NORM_LAYER) if w.config.NORM_LAYER else None),
            use_checkpoint=w.config.USE_CHECKPOINT,
            post_norm=w.config.POST_NORM,
            enable_stable=w.config.ENABLE_STABLE,
            stable_k=w.config.STABLE_K,
            stable_alpha=w.config.STABLE_ALPHA,
            use_shakedrop=w.config.USE_SHAKEDROP,
            shakedrop_alpha_range=tuple(w.config.SHAKEDROP_ALPHA_RANGE),
            layer_scale=w.config.LAYER_SCALE,
            layer_scale_init_value=w.config.LAYER_SCALE_INIT_VALUE,
        ).to(device)
    elif w.config.ARCHITECTURE == "SwinDPL":
        from models.swin_transformer_dpl_3d import (
            SwinTransformerT as SwinTransformer,
        )

        model = SwinTransformer(
            img_size=(
                tuple(w.config.RESHAPE_SIZE)
                if w.config.RESHAPE_SIZE
                else tuple(w.config.IMG_SIZE)
            ),
            in_channels=1,
            patch_size=tuple(w.config.PATCH_SHAPE),
            embed_dim=w.config.EMBED_DIM,
            depths=w.config.DEPTH,
            num_heads=w.config.HEADS,
            window_size=tuple(w.config.WINDOW_SIZE),
            mlp_ratio=w.config.MLP_RATIO,
            qkv_bias=w.config.QKV_BIAS,
            dropout=w.config.DROPOUT,
            attention_dropout=w.config.ATTENTION_DROPOUT,
            stochastic_depth_prob=w.config.STOCHASTIC_DEPTH_PROB,
            num_classes=len(w.config.DISEASES),
            norm_layer=(eval(w.config.NORM_LAYER) if w.config.NORM_LAYER else None),
            patch_norm=w.config.PATCH_NORM,
            use_checkpoint=w.config.USE_CHECKPOINT,
            post_norm=w.config.POST_NORM,
            use_shakedrop=w.config.USE_SHAKEDROP,
            shakedrop_alpha_range=tuple(w.config.SHAKEDROP_ALPHA_RANGE),
            layer_scale=w.config.LAYER_SCALE,
            layer_scale_init_value=w.config.LAYER_SCALE_INIT_VALUE,
        ).to(device)
    elif w.config.ARCHITECTURE == "MedViT":
        from models.medvit_3d import MedViTV1S as MedViTv1

        model = MedViTv1(
            in_channels=1,
            depths=w.config.DEPTH,
            stochastic_depth_prob=w.config.STOCHASTIC_DEPTH_PROB,
            attention_dropout=w.config.ATTENTION_DROPOUT,
            dropout=w.config.DROPOUT,
            num_classes=len(w.config.DISEASES),
            head_dim=w.config.HEADS,
            mlp_ratio=w.config.MLP_RATIO,
            use_checkpoint=w.config.USE_CHECKPOINT,
            enable_stable=w.config.ENABLE_STABLE,
            stable_k=w.config.STABLE_K,
            stable_alpha=w.config.STABLE_ALPHA,
            use_shakedrop=w.config.USE_SHAKEDROP,
            shakedrop_alpha_range=tuple(w.config.SHAKEDROP_ALPHA_RANGE),
            layer_scale=w.config.LAYER_SCALE,
            layer_scale_init_value=w.config.LAYER_SCALE_INIT_VALUE,
        ).to(device)
    elif w.config.ARCHITECTURE == "ViT":
        from models.vit_3d import ViTS as ViT

        model = ViT(
            img_size=(
                tuple(w.config.RESHAPE_SIZE)
                if w.config.RESHAPE_SIZE
                else tuple(w.config.IMG_SIZE)
            ),
            num_classes=len(w.config.DISEASES),
            in_channels=1,
            patch_size=tuple(w.config.PATCH_SHAPE),
            mlp_ratio=w.config.MLP_RATIO,
            dropout=w.config.DROPOUT,
            attention_dropout=w.config.ATTENTION_DROPOUT,
            use_checkpoint=w.config.USE_CHECKPOINT,
            embed_dim=w.config.EMBED_DIM,
            num_heads=int(w.config.HEADS),
            depth=w.config.DEPTH,
            post_norm=w.config.POST_NORM,
            layer_scale=w.config.LAYER_SCALE,
            layer_scale_init_value=w.config.LAYER_SCALE_INIT_VALUE,
        ).to(device)
    elif w.config.ARCHITECTURE == "ResNet":
        from models.resnet_3d import ResNet3DMedical

        model = ResNet3DMedical(
            img_size=(
                tuple(w.config.RESHAPE_SIZE)
                if w.config.RESHAPE_SIZE
                else tuple(w.config.IMG_SIZE)
            ),
            num_classes=len(w.config.DISEASES),
            in_channels=1,
            resnet_variant="resnet18",
            shortcut_type="B",
            dropout=w.config.DROPOUT,
        ).to(device)
    else:
        raise ValueError(f"Unknown architecture: {w.config.ARCHITECTURE}")

    model = model.to(memory_format=torch.channels_last_3d)

    return model


def load_checkpoint(
    model: nn.Module, checkpoint_path: Path, device: torch.device
) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    return checkpoint


def ensure_preprocessed(
    metadata: pd.DataFrame,
    preprocess_dir: Path,
    device: torch.device,
    crop_size: Tuple[int, int, int],
) -> pd.DataFrame:
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    if metadata.empty:
        return metadata

    data_prepa = DataPrepa(
        metadata,
        preprocess_data_dir=preprocess_dir,
        device=device,
    )
    data_prepa.preprocess_data(
        crop=tuple(crop_size),
        downsample=None,
        tqdm_kwargs={
            "desc": f"Preprocessing -> {preprocess_dir.name}",
            "dynamic_ncols": True,
        },
    )
    return data_prepa.metadata.reset_index(drop=True)


def create_loader(
    preprocess_dir: Path,
    metadata: pd.DataFrame,
    batch_size: int,
) -> DataLoader:
    dataset = NormalDataset(
        preprocess_dir,
        metadata.reset_index(drop=True),
        device="cpu",
        diseases=w.config.DISEASES,
        transform=Compose(
            [
                Resize(
                    tuple(w.config.RESHAPE_SIZE)
                    if w.config.RESHAPE_SIZE
                    else tuple(w.config.IMG_SIZE)
                ),
                NormalizeIntensity(),
            ]
        ),
    )

    num_workers = int(w.config.get("NUM_WORKERS", 4))

    # Configure deterministic worker seeding if a global SEED is provided
    base_seed = normalize_seed(w.config.get("SEED"))
    worker_init_fn = None
    generator = None
    if base_seed is not None:
        # For evaluation we use a stable rank=0 adjustment
        dataloader_seed = (int(base_seed) + 0) % _MAX_UINT32

        def _seed_worker(worker_id: int) -> None:
            worker_seed = (dataloader_seed + worker_id) % _MAX_UINT32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            if monai_set_determinism is not None:
                monai_set_determinism(seed=worker_seed)

        worker_init_fn = _seed_worker
        generator = torch.Generator()
        generator.manual_seed(dataloader_seed)

    # Optimize DataLoader configuration for evaluation
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,  # False
    use_tta: bool,  # False
    tta_config: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []

    tta_predictor = None
    if use_tta:
        tta_predictor = create_tta_predictor(model, device, tta_config, use_amp=use_amp)

    with torch.inference_mode():
        for inputs, labels in loader:
            # For TTA, keep inputs on CPU and let TTA handle device transfers
            # This avoids redundant CPU->GPU->CPU->GPU transfers during augmentation
            if use_tta and tta_predictor is not None:
                # TTA will handle device transfer internally for each augmentation
                # Keep inputs on CPU for efficient augmentation
                labels = labels.to(device, non_blocking=True)
                probabilities = tta_predictor(inputs)  # Returns probabilities on device
            else:
                # Standard inference: move to device and apply memory format
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # MedViT uses NATTEN which is not compatible with channels_last_3d
                if w.config.ARCHITECTURE != "MedViT":
                    inputs = inputs.to(memory_format=torch.channels_last_3d)

                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(inputs)
                else:
                    logits = model(inputs)

                probabilities = torch.softmax(logits, dim=1)

            preds.append(probabilities.cpu())
            targets.append(labels.cpu())

    if preds:
        preds_tensor = torch.cat(preds, dim=0)
        targets_tensor = torch.cat(targets, dim=0)
    else:
        preds_tensor = torch.empty((0, len(w.config.DISEASES)))
        targets_tensor = torch.empty((0, len(w.config.DISEASES)))

    return preds_tensor.numpy(), targets_tensor.numpy()


def attach_predictions(metadata: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    df = metadata.reset_index(drop=True).copy()
    for class_idx, disease in enumerate(w.config.DISEASES):
        df[f"pred_{disease}"] = predictions[:, class_idx]
    return df


def filter_synthetic(metadata: pd.DataFrame) -> pd.DataFrame:
    if "Subject" not in metadata.columns:
        return metadata
    mask = ~metadata["Subject"].astype(str).str.lower().str.contains("factor")
    return metadata.loc[mask].reset_index(drop=True)


def remove_duplicates(
    metadata: pd.DataFrame, logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """Remove duplicate subjects (based on 'Subject') and log how many were removed.

    If 'Subject' column is missing, returns metadata unchanged.
    """
    if metadata.empty or "Subject" not in metadata.columns:
        return metadata

    before = len(metadata)
    deduped = metadata.drop_duplicates(subset=["Subject"]).reset_index(drop=True)
    removed = before - len(deduped)
    if logger is not None:
        logger.info(
            f"Removed {removed} duplicate subjects (from {before} -> {len(deduped)})"
        )
    return deduped


def compute_bootstrap_from_predictions(
    preds: np.ndarray, targets: np.ndarray
) -> Optional[Dict]:
    """Compute bootstrap metrics from predictions and targets.

    Returns None if predictions or targets are empty.
    """
    if preds.size == 0 or targets.size == 0:
        return None
    labels = targets.argmax(axis=1)
    return compute_bootstrap_metrics(labels, preds)


def log_metrics(
    logger: logging.Logger,
    split_name: str,
    bootstrap_metrics: Dict,
    preds: np.ndarray,
    targets: np.ndarray,
) -> None:
    """Log all metrics with confidence intervals from bootstrap results."""
    logger.info(f"=== {split_name} ===")

    if preds.size == 0 or targets.size == 0 or not bootstrap_metrics:
        logger.info("No predictions available for this split.")
        return

    # Extract main metrics with CIs
    acc_ci = bootstrap_metrics["accuracy"]
    bacc_ci = bootstrap_metrics["balanced_accuracy"]
    auc_ci = bootstrap_metrics["roc_auc"]
    pr_auc_ci = bootstrap_metrics["pr_auc"]
    mcc_ci = bootstrap_metrics.get("mcc", {})

    # Extract F1 macro from classification report
    f1_macro = (
        bootstrap_metrics.get("classification_report", {})
        .get("macro avg", {})
        .get("f1-score", {})
    )

    logger.info(
        f"Accuracy: {acc_ci['mean'] * 100:.2f}% [{acc_ci['lower'] * 100:.2f} - {acc_ci['upper'] * 100:.2f}] | "
        f"Balanced Accuracy: {bacc_ci['mean'] * 100:.2f}% [{bacc_ci['lower'] * 100:.2f} - {bacc_ci['upper'] * 100:.2f}]"
    )
    logger.info(
        f"ROC-AUC: {auc_ci['mean'] * 100:.2f}% [{auc_ci['lower'] * 100:.2f} - {auc_ci['upper'] * 100:.2f}] | "
        f"PR-AUC: {pr_auc_ci['mean'] * 100:.2f}% [{pr_auc_ci['lower'] * 100:.2f} - {pr_auc_ci['upper'] * 100:.2f}]"
    )

    if mcc_ci:
        logger.info(
            f"MCC: {mcc_ci['mean'] * 100:.2f}% [{mcc_ci['lower'] * 100:.2f} - {mcc_ci['upper'] * 100:.2f}]"
        )

    if f1_macro:
        logger.info(
            f"F1 (macro): {f1_macro['mean'] * 100:.2f}% [{f1_macro['lower'] * 100:.2f} - {f1_macro['upper'] * 100:.2f}]"
        )

    # Log per-class F1 scores
    logger.info("Per-class F1 scores:")
    for class_idx, disease in enumerate(w.config.DISEASES):
        f1_class = bootstrap_metrics.get("f1", {}).get(class_idx, {})
        if f1_class:
            logger.info(
                f"  {disease}: {f1_class['mean'] * 100:.2f}% [{f1_class['lower'] * 100:.2f} - {f1_class['upper'] * 100:.2f}]"
            )


def prepare_wandb_config(model_path: Path) -> Path:
    parts = model_path.stem.split("_")
    if len(parts) < 2 or not re.fullmatch(r"[a-z0-9]{8}", parts[1]):
        raise ValueError(
            f"Unable to infer W&B run id from checkpoint name '{model_path.stem}'."
        )
    runname = parts[1]

    wandb_parent = model_path.parent / "wandb"
    candidates = list(wandb_parent.glob(f"run-*-{runname}")) + list(
        wandb_parent.glob(f"offline-run-*-{runname}")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No W&B directory found for run id '{runname}' under {wandb_parent}"
        )

    wandb_dir = sorted(candidates, key=lambda p: p.stem)[0]

    config_candidates = [
        wandb_dir / "files" / "config.yaml",
        wandb_dir / "files" / "config-defaults.yaml",
        wandb_dir / "files" / "files" / "config.yaml",
    ]

    for candidate in config_candidates:
        if candidate.exists():
            return candidate

    available_files = list(wandb_dir.rglob("*.yaml"))
    raise FileNotFoundError(
        f"No config YAML found in {wandb_dir}. "
        f"Available YAML files: {[f.relative_to(wandb_dir) for f in available_files]}"
    )


def evaluation(
    args: argparse.Namespace,
    model_path: Path,
    device: torch.device,
    model_index: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    fname = model_path.stem

    # Determine output directory: either relative subfolder or checkpoint parent
    if args.output_folder:
        output_dir = model_path.parent / args.output_folder
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = model_path.parent

    output_id = output_dir / f"prediction_{fname}_id.csv"
    output_od = output_dir / f"prediction_{fname}_od.csv"

    if (
        not args.force_eval
        and output_id.exists()
        and (args.eval_csv is None or output_od.exists())
    ):
        print(
            f"Model {fname} already evaluated in {output_dir}. Use --force-eval to re-evaluate. Skipping."
        )
        id_df = pd.read_csv(output_id)
        od_df = pd.read_csv(output_od) if output_od.exists() else None
        return id_df, od_df

    logger = setup_logger(model_path, output_dir)

    checkpoint_info = torch.load(model_path, map_location=device, weights_only=False)
    train_progress = checkpoint_info.get("step", checkpoint_info.get("epoch", 0))
    progress_label = "steps" if "step" in checkpoint_info else "epochs"

    logger.info(f"===== {model_path.stem} ({train_progress} {progress_label}) =====")
    logger.info(
        f"Optimization settings: batch_size={args.batch_size}, use_amp={args.use_amp}, channels_last_3d=True, inference_mode=True"
    )
    if args.use_tta:
        logger.info(
            f"TTA enabled: num_samples={args.tta_num_samples}, flip={not args.tta_no_flip}, "
            f"affine={not args.tta_no_affine}, center_crop={not args.tta_no_center_crop}, "
            f"crop_scale={args.tta_crop_scale}"
        )

    # Dataset preparation -------------------------------------------------
    _, val_df, test_df, all_df = get_train_val_test(
        Path(args.training_csv_dir),
        w.config.FOLD,
        w.config.KFOLD,
        split=w.config.SPLIT,
    )

    if not w.config.get("USE_SYNTHETIC_DATA", False):
        val_df = filter_synthetic(val_df)
        test_df = filter_synthetic(test_df)

    preprocess_train_dir = Path(args.intermediate_dir) / "train"
    preprocess_test_dir = Path(args.intermediate_dir) / "testset"

    ensure_preprocessed(all_df, preprocess_train_dir, device, tuple(w.config.IMG_SIZE))

    logger.info("=== Dataset Sizes ===")
    logger.info(f"In-domain Validation set: {len(val_df)} samples")
    logger.info(f"In-domain Test set: {len(test_df)} samples")

    if "Diagnosis" in test_df.columns:
        class_counts = test_df["Diagnosis"].value_counts().sort_index()
        logger.info("In-domain Test distribution:")
        for cls, count in class_counts.items():
            logger.info(f"  {cls}: {count} samples ({count / len(test_df) * 100:.1f}%)")

    # Build evaluation model (reload to ensure clean weights)
    model = build_model(device)
    load_checkpoint(model, model_path, device)

    val_loader = create_loader(preprocess_train_dir, val_df, args.batch_size)
    test_loader = create_loader(preprocess_train_dir, test_df, args.batch_size)

    # Prepare TTA configuration if enabled
    tta_config = None
    if args.use_tta:
        # Get target shape from config
        target_shape = (
            tuple(w.config.RESHAPE_SIZE)
            if w.config.RESHAPE_SIZE
            else tuple(w.config.IMG_SIZE)
        )
        tta_config = {
            "num_samples": args.tta_num_samples,
            "use_flip": not args.tta_no_flip,
            "use_affine": not args.tta_no_affine,
            "use_scaled_center_crop": not args.tta_no_center_crop,
            "crop_roi_scale": args.tta_crop_scale,
            "affine_rotate_range": (3.0, 3.0, 3.0),  # \pm 3 degrees
            "affine_translate_range": (5.0, 5.0, 5.0),  # \pm 5 voxels
            "target_shape": target_shape,
            "use_channels_last": w.config.ARCHITECTURE != "MedViT",
            "use_entropy_weighting": True,
        }

    val_preds, val_targets = evaluate_loader(
        model,
        val_loader,
        device,
        use_amp=args.use_amp,
        use_tta=args.use_tta,
        tta_config=tta_config,
    )
    val_bootstrap = compute_bootstrap_from_predictions(val_preds, val_targets)

    log_metrics(logger, "Validation (ID)", val_bootstrap, val_preds, val_targets)

    test_preds, test_targets = evaluate_loader(
        model,
        test_loader,
        device,
        use_amp=args.use_amp,
        use_tta=args.use_tta,
        tta_config=tta_config,
    )
    test_bootstrap = compute_bootstrap_from_predictions(test_preds, test_targets)

    log_metrics(logger, "Test (ID)", test_bootstrap, test_preds, test_targets)

    id_predictions = attach_predictions(test_df, test_preds)
    id_predictions.to_csv(output_id, index=False)
    logger.info(f"Saved in-domain predictions to {output_id}")

    od_predictions_df = None
    od_bootstrap = None

    if args.eval_csv is not None:
        metadata_od = pd.read_csv(args.eval_csv)
        metadata_od = ensure_preprocessed(
            metadata_od, preprocess_test_dir, device, tuple(w.config.IMG_SIZE)
        )

        logger.info(f"Out-of-domain Test set: {len(metadata_od)} samples")
        if "Diagnosis" in metadata_od.columns and len(metadata_od) > 0:
            class_counts = metadata_od["Diagnosis"].value_counts().sort_index()
            logger.info("Out-of-domain Test distribution:")
            for cls, count in class_counts.items():
                logger.info(
                    f"  {cls}: {count} samples ({count / len(metadata_od) * 100:.1f}%)"
                )

        od_loader = create_loader(preprocess_test_dir, metadata_od, args.batch_size)
        od_preds, od_targets = evaluate_loader(
            model,
            od_loader,
            device,
            use_amp=args.use_amp,
            use_tta=args.use_tta,
            tta_config=tta_config,
        )
        od_bootstrap = compute_bootstrap_from_predictions(od_preds, od_targets)

        split_name = (
            "Test (OD)"
            if "Diagnosis" in metadata_od.columns
            else "Test (OD, no ground-truth)"
        )
        log_metrics(logger, split_name, od_bootstrap, od_preds, od_targets)

        od_predictions_df = attach_predictions(metadata_od, od_preds)
        od_predictions_df.to_csv(output_od, index=False)
        logger.info(f"Saved out-of-domain predictions to {output_od}")

    log_metrics_to_wandb(
        logger,
        model_path,
        val_bootstrap,
        test_bootstrap,
        od_bootstrap,
        id_csv_path=output_id,
        od_csv_path=output_od if args.eval_csv is not None else None,
        model_index=model_index,
    )

    return id_predictions, od_predictions_df


def log_metrics_to_wandb(
    logger: logging.Logger,
    model_path: Path,
    val_bootstrap: Dict,
    test_bootstrap: Dict,
    od_bootstrap: Optional[Dict] = None,
    id_csv_path: Optional[Path] = None,
    od_csv_path: Optional[Path] = None,
    model_index: Optional[str] = None,
) -> None:
    """Log metrics to W&B as a table and upload prediction CSV files as artifacts.

    All metrics are expected to come from bootstrap computation with CIs.
    """

    if w.run is None or w.run.settings.mode == "disabled":
        logger.info("W&B logging is disabled. Skipping metric logging to W&B.")
        return

    def fmt_pct(value: float) -> float:
        return round(value * 100, 2)

    # Helper function to create row with model index in split name
    def create_row(split_name: str, bootstrap: Dict):
        # Use OrderedDict-like approach by building in order
        from collections import OrderedDict

        row = OrderedDict()

        # Column 1: Split name
        row["Split"] = split_name

        # Columns 2-5: Main metrics (Accuracy, BACC, ROC-AUC, PR-AUC) - mean values
        row["Accuracy"] = fmt_pct(bootstrap["accuracy"]["mean"])
        row["Balanced Accuracy"] = fmt_pct(bootstrap["balanced_accuracy"]["mean"])
        row["ROC-AUC"] = fmt_pct(bootstrap["roc_auc"]["mean"])
        row["PR-AUC"] = fmt_pct(bootstrap["pr_auc"]["mean"])

        # Column 5.5: MCC
        mcc_data = bootstrap.get("mcc", {})
        row["MCC"] = fmt_pct(mcc_data.get("mean", 0.0)) if mcc_data else 0.0

        # Column 6: F1 macro
        f1_macro = (
            bootstrap.get("classification_report", {})
            .get("macro avg", {})
            .get("f1-score", {})
            .get("mean", 0.0)
        )
        row["F1 (macro)"] = fmt_pct(f1_macro)

        # Columns 7+: Per-class F1 scores
        for class_idx, disease in enumerate(w.config.DISEASES):
            f1_class = bootstrap.get("f1", {}).get(class_idx, {}).get("mean", 0.0)
            row[f"F1_{disease}"] = fmt_pct(f1_class)

        # Now add all CIs in the same order: ACC, BACC, AUC, PR-AUC, F1 macro, F1 per-class

        # Accuracy CI
        row["Accuracy_CI_lower"] = fmt_pct(bootstrap["accuracy"]["lower"])
        row["Accuracy_CI_upper"] = fmt_pct(bootstrap["accuracy"]["upper"])

        # BACC CI
        row["BACC_CI_lower"] = fmt_pct(bootstrap["balanced_accuracy"]["lower"])
        row["BACC_CI_upper"] = fmt_pct(bootstrap["balanced_accuracy"]["upper"])

        # AUC CI
        row["AUC_CI_lower"] = fmt_pct(bootstrap["roc_auc"]["lower"])
        row["AUC_CI_upper"] = fmt_pct(bootstrap["roc_auc"]["upper"])

        # PR-AUC CI
        row["PR_AUC_CI_lower"] = fmt_pct(bootstrap["pr_auc"]["lower"])
        row["PR_AUC_CI_upper"] = fmt_pct(bootstrap["pr_auc"]["upper"])

        # MCC CI
        if mcc_data:
            row["MCC_CI_lower"] = fmt_pct(mcc_data.get("lower", 0.0))
            row["MCC_CI_upper"] = fmt_pct(mcc_data.get("upper", 0.0))
        else:
            row["MCC_CI_lower"] = 0.0
            row["MCC_CI_upper"] = 0.0

        # F1 macro CI
        f1_macro_lower = (
            bootstrap.get("classification_report", {})
            .get("macro avg", {})
            .get("f1-score", {})
            .get("lower", 0.0)
        )
        f1_macro_upper = (
            bootstrap.get("classification_report", {})
            .get("macro avg", {})
            .get("f1-score", {})
            .get("upper", 0.0)
        )
        row["F1_macro_CI_lower"] = fmt_pct(f1_macro_lower)
        row["F1_macro_CI_upper"] = fmt_pct(f1_macro_upper)

        # Per-class F1 CIs
        for class_idx, disease in enumerate(w.config.DISEASES):
            f1_class_lower = (
                bootstrap.get("f1", {}).get(class_idx, {}).get("lower", 0.0)
            )
            f1_class_upper = (
                bootstrap.get("f1", {}).get(class_idx, {}).get("upper", 0.0)
            )
            row[f"F1_{disease}_CI_lower"] = fmt_pct(f1_class_lower)
            row[f"F1_{disease}_CI_upper"] = fmt_pct(f1_class_upper)

        return row

    new_rows = []
    model_label = f" ({model_index})" if model_index else ""

    if val_bootstrap:
        new_rows.append(create_row(f"Validation{model_label}", val_bootstrap))

    if test_bootstrap:
        new_rows.append(create_row(f"Test{model_label} (ID)", test_bootstrap))

    if od_bootstrap:
        new_rows.append(create_row(f"Test{model_label} (OD)", od_bootstrap))

    if not new_rows:
        return

    # Table name to use consistently
    table_name = "evaluation_metrics"
    all_rows = []

    # Try to fetch existing table artifact to append to it
    try:
        # Look for existing table artifact with the new naming convention
        api = w.Api()
        artifact_name = (
            f"{w.run.entity}/{w.run.project}/eval-table-data-{w.run.id}:latest"
        )
        try:
            artifact = api.artifact(artifact_name)
            # Download and load existing table
            artifact_dir = artifact.download()

            table_path = Path(artifact_dir) / "table.json"
            if table_path.exists():
                with open(table_path, "r") as f:
                    table_data = json.load(f)
                    all_rows = table_data.get("data", [])
                logger.info(
                    f"Found existing table with {len(all_rows)} rows, checking for duplicates"
                )
            shutil.rmtree(artifact_dir)
        except Exception:
            # Artifact doesn't exist yet, will create new
            logger.info("No existing table found, creating new one")
    except Exception:
        logger.info("Creating new evaluation metrics table")

    # Remove existing rows for this model_index to avoid duplicates
    if model_index and all_rows:
        model_patterns = [f"({model_index})", f" ({model_index}) "]
        rows_before = len(all_rows)
        all_rows = [
            row
            for row in all_rows
            if not any(pattern in str(row[0]) for pattern in model_patterns)
        ]
        rows_removed = rows_before - len(all_rows)
        if rows_removed > 0:
            logger.info(
                f"Removed {rows_removed} existing rows for model '{model_index}' to avoid duplicates"
            )

    # Add new rows to existing data
    columns = list(new_rows[0].keys())
    for row in new_rows:
        all_rows.append([row[col] for col in columns])

    # Sort all rows by model index for consistent ordering
    def extract_sort_key(row_data):
        split_name = row_data[0]  # First column is "Split"
        # Extract model identifier from split name
        match = re.search(r"\(([^)]+)\)", split_name)
        if match:
            model_str = match.group(1)
            # Extract number from "best0", "best1", etc.
            num_match = re.search(r"(\d+)", model_str)
            if num_match:
                return int(num_match.group(1))
        return 999  # Put at end if no number found

    all_rows.sort(key=extract_sort_key)

    # Create table with all rows (existing + new)
    table = w.Table(columns=columns, data=all_rows)

    # Log the table
    w.log({table_name: table})

    # Save table data as a separate artifact for persistence across evaluations
    # Use a different name to avoid conflict with W&B's automatic table artifact
    artifact = w.Artifact(
        name=f"eval-table-data-{w.run.id}",
        type="evaluation_data",
        description="Cumulative evaluation metrics table data for persistence",
    )

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"columns": columns, "data": all_rows}, f)
            temp_path = f.name

        artifact.add_file(temp_path, name="table.json")
        w.log_artifact(artifact)
    finally:
        # Ensure temp file is deleted even if there's an error
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    logger.info(
        f"Logged {len(all_rows)} total rows to W&B table '{table_name}' ({len(new_rows)} new)"
    )

    # Upload prediction CSV files as W&B artifacts with model_index in the name
    if id_csv_path and id_csv_path.exists():
        artifact_name = (
            f"predictions_id_{model_index}"
            if model_index
            else f"predictions_id_{model_path.stem}"
        )
        artifact = w.Artifact(
            name=artifact_name,
            type="predictions",
            description=f"In-domain predictions for {model_index or model_path.stem}",
        )
        artifact.add_file(
            str(id_csv_path),
            name=f"predictions_id_{model_index or model_path.stem}.csv",
        )
        w.log_artifact(artifact)
        logger.info(f"Uploaded in-domain predictions CSV to W&B as '{artifact_name}'")

    if od_csv_path and od_csv_path.exists():
        artifact_name = (
            f"predictions_od_{model_index}"
            if model_index
            else f"predictions_od_{model_path.stem}"
        )
        artifact = w.Artifact(
            name=artifact_name,
            type="predictions",
            description=f"Out-of-domain predictions for {model_index or model_path.stem}",
        )
        artifact.add_file(
            str(od_csv_path),
            name=f"predictions_od_{model_index or model_path.stem}.csv",
        )
        w.log_artifact(artifact)
        logger.info(
            f"Uploaded out-of-domain predictions CSV to W&B as '{artifact_name}'"
        )


def main() -> None:
    args = get_args()

    cuda_device = int(args.cuda_device)
    if torch.cuda.is_available() and cuda_device < torch.cuda.device_count():
        device = torch.device(f"cuda:{cuda_device}")
    else:
        device = torch.device("cpu")

    for ckpt in args.checkpoints:
        model_path = Path(ckpt)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        config_path = prepare_wandb_config(model_path)

        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        config = {
            key: (
                value["value"]
                if isinstance(value, dict) and "value" in value
                else value
            )
            for key, value in raw_config.items()
        }

        # Extract W&B run ID and model index from checkpoint filename
        # Format: model_{wandb_id}_{fold}_best{index}.pt or model_{wandb_id}_{fold}_last.pt
        checkpoint_stem = model_path.stem
        parts = checkpoint_stem.split("_")

        wandb_id = None
        model_index = None

        if len(parts) >= 2 and re.fullmatch(r"[a-z0-9]{8}", parts[1]):
            wandb_id = parts[1]

            # Extract model index from the last part (e.g., "best0" -> "best0", "last" -> "last")
            if len(parts) >= 4:
                model_index = parts[3]  # e.g., "best0", "best1", "last"
            else:
                model_index = "unknown"

        # Add retry logic and longer timeout to handle parallel W&B run resumption
        max_retries = 3
        retry_delay = 5  # seconds
        run = None

        for attempt in range(max_retries):
            try:
                run = w.init(
                    mode="online" if args.log_to_wandb else "disabled",
                    project=args.project_name,
                    name=model_path.parent.name,
                    id=wandb_id,
                    resume="allow",
                    dir=config_path.parent.parent,
                    settings=w.Settings(
                        allow_val_change=True,
                        init_timeout=180,  # Increased timeout for parallel runs
                        _disable_stats=True,  # Reduce API calls
                        _disable_meta=True,
                    ),
                )
                break  # Success, exit retry loop
            except (w.errors.CommError, TimeoutError) as e:
                if attempt < max_retries - 1:
                    logging.warning(
                        f"W&B init failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    import time

                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logging.error(
                        f"W&B init failed after {max_retries} attempts. "
                        f"Continuing without W&B logging."
                    )
                    # Fall back to disabled mode
                    run = w.init(mode="disabled")
                    break

        if run is None:
            run = w.init(mode="disabled")
        run.config.update(config, allow_val_change=True)

        # Apply global seed from saved config (if any) to make evaluation deterministic
        normalized_seed = normalize_seed(run.config.get("SEED"))
        run.config.update({"SEED": normalized_seed}, allow_val_change=True)
        if normalized_seed is not None:
            # For single-process evaluation, rank is 0
            rank_adjusted_seed = (int(normalized_seed) + 0) % _MAX_UINT32
            seed_everything(rank_adjusted_seed)
            print(
                f"Evaluation seeded with SEED={int(normalized_seed)} (rank-adjusted {rank_adjusted_seed})"
            )

        try:
            evaluation(args, model_path, device, model_index)
        finally:
            w.finish()


if __name__ == "__main__":
    main()
