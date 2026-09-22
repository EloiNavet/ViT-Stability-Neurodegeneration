# On the Stability and Robustness of Vision Transformers for Neurodegenerative Disease Classification

**Official PyTorch implementation accompanying the MIDL 2026 paper**  
*On the Stability and Robustness of Vision Transformers for Neurodegenerative Disease Classification*

[![MIDL 2026](https://img.shields.io/badge/MIDL-2026-5B5BD6.svg)](https://proceedings.mlr.press/v315/navet26a.html)
[![PMLR 315](https://img.shields.io/badge/PMLR-315-blue.svg)](https://proceedings.mlr.press/v315/navet26a.html)
[![OpenReview](https://img.shields.io/badge/OpenReview-MiS54B5arR-b31b1b.svg)](https://openreview.net/forum?id=MiS54B5arR)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

Vision Transformers are increasingly used for structural brain MRI classification, but their behaviour can become highly variable in limited, heterogeneous clinical cohorts. This work studies how that variability changes when moving from simpler classification settings to more difficult **differential-diagnosis tasks**, where class imbalance, phenotype overlap, noisy labels, and inter-site heterogeneity become more important.

We investigate a cumulative stabilization framework spanning **data augmentation, sampling, optimization, architecture, inference, calibration, and uncertainty-aware evaluation**. The repository contains the training and evaluation framework used to study these effects across several 3D backbones and both 3-class and 5-class neurodegenerative-disease classification settings.

**Paper:** [PMLR 315:4518–4554](https://proceedings.mlr.press/v315/navet26a.html)  
**OpenReview:** [MiS54B5arR](https://openreview.net/forum?id=MiS54B5arR)

---

## Key Findings

- **Stability does not automatically transfer to harder diagnostic settings.** Variability becomes more consequential as the task moves from simpler classification toward multiclass differential diagnosis.
- **Single-run point estimates can be misleading.** Apparent improvements may disappear once stochastic training variability and uncertainty are explicitly quantified.
- **Stabilization must be considered as a pipeline.** Data, optimization, inference, and calibration choices jointly influence predictive performance and run-to-run reliability.
- **Uncertainty-aware comparisons are essential.** The study combines patient-level paired bootstrapping, calibration analysis, paired statistical comparisons, and estimates of false outperformance rather than relying only on a single test score.
- **Reliable neuroimaging classification requires reporting variability alongside performance.** The paper therefore treats stability as a first-class evaluation target rather than a secondary implementation detail.

---

## Stabilization Framework

| Category | Components represented in this repository |
| --- | --- |
| **Architectures** | ViT-3D, Swin Transformer 3D, MedViT-3D, ResNet-3D |
| **Data / sampling** | 3D augmentation, MixUp, class-balanced sampling |
| **Optimization** | Exponential Moving Average (EMA), Sharpness-Aware Minimization (SAM), label smoothing |
| **Inference** | Test-Time Augmentation (TTA), checkpoint / snapshot ensembling |
| **Calibration** | Temperature scaling, ECE, Brier score |
| **Stability analysis** | repeated-run / seed analysis and bootstrap-based uncertainty tooling |
| **Tasks** | 3-class (`CN`, `AD`, `FTD`) and 5-class (`CN`, `AD`, `PNFA`, `BV`, `SD`) classification |

The supplied configuration files provide baseline experiment definitions for the main standard backbones in both diagnostic settings.

---

## Code Availability

This release focuses on the reproducible **stabilization and evaluation framework** used in the paper.

### Included

- Full PyTorch training and evaluation pipeline.
- Standard 3D backbones used for the stability experiments.
- 3-class and 5-class experiment configurations.
- Stabilization components including EMA, SAM, MixUp, balanced sampling, and label smoothing.
- Inference and calibration utilities.
- Scripts used for stability and uncertainty visualizations.

### Not included

- **Swin-DPL (Deformable Patch Location):** the implementation used in the study is proprietary and is not distributed. The repository contains only an interface/placeholder for compatibility with the experimental framework.
- **AssemblyNet:** used as an external baseline and not distributed with this repository.

---

## Repository Structure

```text
├── configs/             # 3-class / 5-class experiment configurations
├── dataset/             # Data loading and preprocessing utilities
├── eval/                # Evaluation and metrics
├── models/
│   ├── medvit_3d.py
│   ├── resnet_3d.py
│   ├── swin_transformer_3d.py
│   ├── swin_transformer_dpl_3d.py   # interface / placeholder
│   ├── vit_3d.py
│   └── modules/
├── regularization/      # Regularization and stabilization components
├── scripts/             # Training orchestration scripts
├── train/               # Training code
├── utils/               # EMA, seeding, schedulers, calibration, helpers
├── visualizations/      # Stability / uncertainty analysis scripts
├── config-defaults.yaml
├── requirements.txt
└── LICENSE
```

---

## Installation

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA ≥ 11.8 for GPU training

### Setup

```bash
git clone https://github.com/EloiNavet/ViT-Stability-Neurodegeneration.git
cd ViT-Stability-Neurodegeneration
pip install -r requirements.txt
```

---

## Data Availability

The experiments use structural T1-weighted MRI from multiple research cohorts, including **ADNI**, **NACC**, **NIFD / FTLDNI**, and **ALLFTD**.

| Cohort | Access |
| --- | --- |
| **ADNI** | [adni.loni.usc.edu](https://adni.loni.usc.edu/) |
| **NACC** | [naccdata.org](https://naccdata.org/) |
| **NIFD / FTLDNI** | [ida.loni.usc.edu](https://ida.loni.usc.edu/) |
| **ALLFTD** | [allftd.org](https://www.allftd.org/) |

The aggregated imaging data cannot be redistributed through this repository because access is governed by the individual cohorts and their respective data-use agreements. Researchers should obtain the source datasets from the corresponding providers.

---

## Data Preparation

### Preprocessing Assumption

The training pipeline assumes that input NIfTI files listed in the CSV metadata are **already anatomically preprocessed**.

- Registration and skull stripping are not performed by the training code.
- Input images should be registered to a common anatomical space and bias-corrected before training.
- The pipeline handles NIfTI-to-tensor conversion, intensity normalization, caching, and on-the-fly augmentation.

### Expected CSV Format

At minimum, CSV files should contain:

| Column | Required | Description |
| --- | --- | --- |
| `Subject` | Yes | Unique subject identifier |
| `Diagnosis` | Yes | Class label matching `DISEASES` in the configuration |
| `T1_path` | Yes | Path to the preprocessed T1-weighted NIfTI image |
| `Mask_path` | No | Optional brain-mask path |

### Cross-Validation Splits

Provide K-fold CSV files under `--training-csv-dir`:

```text
fold_0.csv
fold_1.csv
...
fold_K-1.csv
```

The configurations distributed with this release use the experiment definitions from the paper; avoid changing them when attempting to reproduce the reported setup.

---

## Available Configurations

Baseline configurations are provided for the standard backbones evaluated in the released framework:

| Architecture | 3-class | 5-class |
| --- | :---: | :---: |
| ViT-3D | `vit-3c-no_seed-baseline.yaml` | `vit-5c-no_seed-baseline.yaml` |
| Swin Transformer 3D | `swin-3c-no_seed-baseline.yaml` | `swin-5c-no_seed-baseline.yaml` |
| MedViT-3D | `medvit-3c-no_seed-baseline.yaml` | `medvit-5c-no_seed-baseline.yaml` |
| ResNet-3D | `resnet-3c-no_seed-baseline.yaml` | `resnet-5c-no_seed-baseline.yaml` |

Swin-DPL configuration files are also retained for experimental compatibility, but the proprietary model implementation is not distributed.

---

## Training

Example using the 5-class Swin Transformer baseline:

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

Useful options include:

| Option | Description |
| --- | --- |
| `--fold N` | Train a single cross-validation fold |
| `--checkpoint /path/to/model.pt` | Resume from a checkpoint |
| `--wandb-mode disabled` | Disable Weights & Biases logging |

---

## Evaluation

```bash
python -m eval.eval_transformer \
    --training-csv-dir /path/to/Kfold_CV/ \
    --intermediate-dir /path/to/cache_dir/ \
    --checkpoints /path/to/run_dir/model_<wandb_id>_<fold>_best*.pt
```

The repository contains utilities for predictive metrics, calibration assessment, and uncertainty analysis. The paper emphasizes that comparisons should not be interpreted from a single run alone: seed-level variation and patient-level uncertainty are central to the evaluation protocol.

---

## Reproducibility Notes

To reproduce the released experiments on the standard backbones:

- Use the supplied `configs/*` files without modifying the experiment hyperparameters.
- Use the corresponding 3-class or 5-class disease definition consistently across folds and evaluation sets.
- Ensure that all images follow the same preprocessing and registration convention.
- The `*no_seed*` configurations reproduce the non-deterministic setting investigated in the paper, including TF32 / non-deterministic CUDA behaviour where configured.
- Cached preprocessing tensors are stored as `float16` tensors with shape `[1, D, H, W]`.
- Report variation across repeated runs rather than selecting a single favourable seed.

Base configuration: [`config-defaults.yaml`](config-defaults.yaml)  
Experiment overrides: [`configs/`](configs/)

---

## Citation

If you use this framework or the stabilization protocols in your research, please cite:

```bibtex
@InProceedings{pmlr-v315-navet26a,
  title = {On the Stability and Robustness of Vision Transformers for Neurodegenerative Disease Classification},
  author = {Navet, Eloi and Giraud, R{\'e}mi and Mansencal, Boris and Coup{\'e}, Pierrick},
  booktitle = {Proceedings of The 9th International Conference on Medical Imaging with Deep Learning},
  pages = {4518--4554},
  year = {2026},
  editor = {Huo, Yuankai and Gao, Mingchen and Kuo, Chang-Fu and Jin, Yueming and Deng, Ruining},
  volume = {315},
  series = {Proceedings of Machine Learning Research},
  month = {08--10 Jul},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v315/navet26a.html}
}
```

---

## License

This repository is released under the [MIT License](LICENSE).

Third-party notices and attributions are documented in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
