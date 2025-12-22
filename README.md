# On the Stability and Robustness of Vision Transformers for Neurodegenerative Disease Classification

Official PyTorch training framework for 3D Vision Transformers on brain MRI, accompanying the paper **"On the Stability and Robustness of Vision Transformers for Neurodegenerative Disease Classification"**.

This repository provides the **stabilization protocols** (augmentation, optimization strategies, uncertainty quantification) described in the study, allowing reproduction of the training dynamics and stability analysis on standard architectures.

**Paper Status:** Under Review 
**OpenReview:** [https://openreview.net/forum?id=MiS54B5arR](https://openreview.net/forum?id=MiS54B5arR)

## Note on code availability

This repository focuses on the **stabilization framework** proposed in the paper.
- **Included:** Full training/evaluation pipeline, stabilization techniques (EMA, SAM, MixUp, Balanced Sampling), and standard backbones (Swin-Transformer, MedViT, ViT, ResNet-3D).
- **Not Included:** - The specific **Swin-DPL (Deformable Patch Location)** module is proprietary and not included in this release. A placeholder entry exists in the code structure for reference, but the core implementation is private.
    - **AssemblyNet** (used as a baseline in the paper) is a proprietary external tool and is not included.

## Features

- **Stabilization Techniques**: 
    - Data Augmentation (3D-specific)
    - Optimization: EMA (Exponential Moving Average), SAM (Sharpness-Aware Minimization), Label Smoothing
    - Sampling: Class-balanced sampling, MixUp
- **Architectures**: 
    - Swin Transformer 3D (Standard)
    - MedViT 3D
    - ViT 3D
    - ResNet 3D
- **Evaluation Protocols**: 
    - Test-Time Augmentation (TTA)
    - Snapshot Ensembling
    - Temperature Scaling
    - Calibration metrics (ECE, Brier Score)

## Citation

If you use this framework or the stabilization protocols in your research, please cite:

```bibtex
@inproceedings{navet2025on,
  title={On the Stability and Robustness of Vision Transformers for Neurodegenerative Disease Classification},
  author={Eloi Navet and R{\'e}mi Giraud and Boris Mansencal and Pierrick Coupe},
  booktitle={Submitted to Medical Imaging with Deep Learning - Validation Papers},
  year={2025},
  url={[https://openreview.net/forum?id=MiS54B5arR](https://openreview.net/forum?id=MiS54B5arR)},
  note={under review}
}

```

## Installation

```bash
git clone https://github.com/EloiNavet/TransformerTraining.git
cd TransformerTraining
pip install -r requirements.txt

```

**Requirements**: Python ≥3.9, PyTorch ≥2.0, CUDA ≥11.8

## Data Preparation

### 1. Preprocessing Assumption

**Crucial:** The training pipeline assumes that input NIfTI files listed in your CSVs are **already anatomically preprocessed**.

* The code does **not** perform registration or skull-stripping.
* Input images should be registered to MNI space (e.g., using ANTs), skull-stripped, and bias-corrected *before* being passed to this framework.
* The pipeline handles on-the-fly `NIfTI -> Tensor` conversion, intensity Z-scoring, and random augmentations.

### 2. CSV Format

Prepare CSV files with columns:

* `Subject`: unique identifier
* `Diagnosis`: class label (must match `DISEASES` in config)
* `T1_path`: path to the preprocessed NIfTI file (.nii.gz)
* `Mask_path` (optional): brain mask path

### 3. Cross-Validation Splits

Create K-fold splits in `--training-csv-dir`:
```
fold_0.csv
fold_1.csv
...
fold_K-1.csv
```

## Training

```bash
./scripts/transformer.sh \
    --training-csv-dir /path/to/Kfold_CV/ \
    --intermediate-dir /path/to/cache_dir/ \
    --eval-csv /path/to/test.csv \
    --save-dir /path/to/models/ \
    --runname my-experiment \
    --cuda-devices 0,1 \
    --config configs/swin-5c-no_seed-baseline.yaml

```

**Options**:

* `--fold N`: train single fold (default: all 10)
* `--checkpoint /path/to/model.pt`: resume training
* `--wandb-mode disabled`: disable W&B logging

## Evaluation

```bash
python -m eval.eval_transformer \
    --training-csv-dir /path/to/Kfold_CV/ \
    --intermediate-dir /path/to/cache_dir/ \
    --checkpoints /path/to/run_dir/model_<wandb_id>_<fold>_best*.pt

```

## Reproducibility Checklist

To reproduce the stabilization experiments (on standard backbones):

* Use the provided `configs/*` overrides without editing values.
* Ensure your input data is correctly registered to MNI space (1mm isotropic recommended).
* Use the `configs/*no_seed*` configs as used in the paper (TF32 enabled, non-deterministic CUDA kernels).
* Preprocessing cache: tensors are saved as float16 with shape `[1, D, H, W]`.

## Configuration

Base config: `config-defaults.yaml`

Override configs: `configs/*.yaml`

Key parameters:

* `ARCHITECTURE`: `Swin`, `MedViT`, `ViT`, `ResNet`
* `DISEASES`: list of class labels (e.g., `["CN", "AD", "BV", "SD", "PNFA"]`)
* `STEPS`: total training steps
* `EFFECTIVE_BATCH_SIZE`: virtual batch size (gradient accumulation)

## Project Structure

```
├── train/              # Training scripts
├── eval/               # Evaluation and metrics
├── models/             # Architecture definitions
├── dataset/            # Data loading and tensor conversion
├── utils/              # Helpers (EMA, seeding, schedulers)
├── regularization/     # SAM, Label Smoothing, ShakeDrop
├── configs/            # Experiment configurations
└── scripts/            # Shell scripts for training

```

## Datasets

Based on: **ADNI**, **ALLFTD**, **NIFD**, **NACC** cohorts.  
Due to data usage agreements, we cannot share the raw data, but subject splits will be provided for reproducibility (upon acceptance).

## License

MIT License. See [LICENSE](LICENSE).

Third-party notices and attributions are documented in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
