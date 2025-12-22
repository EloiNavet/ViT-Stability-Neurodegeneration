# Third-Party Notices

This repository includes (a) source code adapted from third-party projects and (b) runtime dependencies installed via `pip`.

This file is meant as a practical attribution/notice document and is not legal advice.

---

## Included / Adapted Source Code

### PyTorch `trunc_normal_` implementation
- **Upstream project**: https://github.com/pytorch/pytorch
- **Local file**: `utils/helper.py`
- **License**: BSD-style (see upstream repository)
- **Notes**: The `trunc_normal_` implementation includes a copy of the PyTorch master implementation (see comment in file).

### 3D Vision Transformer (ViT)
- **Upstream project**: https://github.com/lucidrains/vit-pytorch
- **Local file**: `models/vit_3d.py`
- **License**: MIT (see upstream repository)
- **Notes**: The local implementation is adapted from `vit_pytorch/vit_3d.py`.

---

### MedViT V1
- **Upstream project**: https://github.com/Omid-Nejati/MedViT
- **Local files**: `models/medvit_3d.py`, `models/modules/medvit_utils.py`
- **License**: MIT (see upstream repository)
- **Notes**: The local implementation includes MedViT V1 building blocks and utilities.

---

### Swin Transformer (architecture)
- **Upstream reference implementation**: https://github.com/microsoft/Swin-Transformer
- **Local file**: `models/swin_transformer_3d.py`
- **License**: MIT (see upstream repository)
- **Notes**: This repository contains a 3D implementation of Swin V1 concepts; it also relies on `timm` utility layers.

---

### 3D ResNet
- **Upstream project**: https://github.com/dongzhuoyao/3D-ResNets-PyTorch
- **Local file**: `models/resnet_3d.py`
- **License**: MIT (see upstream repository)
- **Copyright**: (c) 2017 Kensho Hara

---

### Sharpness-Aware Minimization (SAM)
- **Upstream project**: https://github.com/davda54/sam
- **Local file**: `regularization/sam.py`
- **License**: MIT (see upstream repository)

---

### ShakeDrop Regularization
- **Upstream project**: https://github.com/owruby/shake-drop_pytorch
- **Local file**: `regularization/shakedrop.py`
- **License**: MIT (see upstream repository)

---

### SLANT BrainCOLOR lookup table (labels)
- **Upstream project**: https://github.com/MASILab/SLANTbrainSeg
- **Upstream file**: `BrainColorLUT.txt`
- **Local usage**: Label names are embedded as `LABELS_SLANT` in `dataset/preprocessing.py`.
- **License**: Custom Vanderbilt license with non-commercial terms (see https://github.com/MASILab/SLANTbrainSeg/blob/master/license.md)

---

## Implementations of Published Techniques (not third-party code)

Some features mentioned in the README (e.g., MixUp/CutMix, Post-LN, LayerScale, StableInit-style residual scaling) are implemented in this repository from paper descriptions.
They are not imported/copied from third-party repositories unless explicitly listed above.

---

## Key Python Dependencies (installed via `pip`)

The project relies on the following third-party Python packages (see their upstream licenses):

- **PyTorch**: https://github.com/pytorch/pytorch (BSD-style license; see upstream)
- **NumPy**: https://github.com/numpy/numpy (BSD-3-Clause)
- **pandas**: https://github.com/pandas-dev/pandas (BSD-3-Clause)
- **scikit-learn**: https://github.com/scikit-learn/scikit-learn (BSD-3-Clause)
- **MONAI**: https://github.com/Project-MONAI/MONAI (Apache-2.0)
- **timm**: https://github.com/rwightman/timm (Apache-2.0)
- **einops**: https://github.com/arogozhnikov/einops (MIT)
- **Weights & Biases (wandb)**: https://github.com/wandb/wandb (MIT)
- **tqdm**: https://github.com/tqdm/tqdm (MPL-2.0)
- **PyYAML**: https://github.com/yaml/pyyaml (MIT)
- **joblib**: https://github.com/joblib/joblib (BSD-3-Clause)
- **NiBabel**: https://github.com/nipy/nibabel (MIT)
- **Nilearn**: https://github.com/nilearn/nilearn (BSD-3-Clause)

---

If you believe a third-party component is missing from this list, please open an issue or PR with:
1) upstream link, 2) license name/link, and 3) the local files where it is used.
