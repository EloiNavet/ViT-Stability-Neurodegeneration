"""PyTorch datasets for 3D brain MRI classification with MixUp/CutMix support."""

import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.distributions.beta import Beta
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.seed import _MAX_UINT32


def load_metadata(
    metadata_paths: str | list[str] | pd.DataFrame | pd.Series,
    accept_datasets: list[str] = None,
) -> pd.DataFrame:
    """Load and concatenate metadata from CSV files or DataFrame."""
    if isinstance(metadata_paths, str):
        metadata = pd.read_csv(metadata_paths).reset_index(drop=True)
    elif isinstance(metadata_paths, list):
        metadata = [pd.read_csv(e) for e in metadata_paths]
        metadata = pd.concat(metadata, ignore_index=True).reset_index(drop=True)
    elif isinstance(metadata_paths, (pd.DataFrame, pd.Series)):
        metadata = metadata_paths.reset_index(drop=True)
    else:
        raise NotImplementedError

    if accept_datasets is not None:
        metadata = metadata[metadata.Dataset.isin(accept_datasets)].reset_index(
            drop=True
        )

    return metadata


class NormalDataset(Dataset):
    """Dataset for loading preprocessed MRI tensors with optional transforms."""

    def __init__(
        self,
        data_root: str,
        meta_data: pd.DataFrame,
        device: torch.device,
        diseases: list[str],
        transform: Optional[callable] = None,
        preload: bool = False,
        preload_transform: Optional[callable] = None,
    ):
        super(NormalDataset, self).__init__()

        self.data_root = data_root
        self.meta_data = meta_data
        self.transform = transform
        self.preload_transform = preload_transform
        self.device = device
        self.diseases = diseases

        # Pre-create label tensors for each diagnosis (memory optimization)
        self._label_cache = {}
        for diagnosis in meta_data.Diagnosis.unique():
            label_tensor = torch.zeros(len(diseases), dtype=torch.float32)
            if diagnosis in diseases:
                label_tensor[diseases.index(diagnosis)] = 1.0
            self._label_cache[diagnosis] = label_tensor

        # Preload all data if requested (only for small datasets!)
        self.preloaded_data = None
        if preload:
            print(f"Preloading {len(meta_data)} samples into memory...")
            self._preload_all_data()
            print("Preloading complete!")

    def _preload_all_data(self):
        """Preload all data into memory. Use only for small datasets!

        If preload_transform is provided, it will be applied to each sample
        during loading. This is useful for ensuring consistent data shapes
        (e.g., applying Resize) before caching.
        """
        self.preloaded_data = {}
        for idx in range(len(self.meta_data)):
            subject = self.meta_data.Subject.iloc[idx]
            path = os.path.join(self.data_root, f"{subject}.pt")
            data = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
            # Apply preload transform if provided (e.g., Resize for shape consistency)
            if self.preload_transform is not None:
                data = self.preload_transform(data)
            self.preloaded_data[idx] = data

    def _load_sample(self, idx: int) -> torch.Tensor:
        """Load a sample from disk or preloaded cache."""
        # Fast path: preloaded data
        if self.preloaded_data is not None:
            return self.preloaded_data[idx]

        # Load from disk
        subject = self.meta_data.Subject.iloc[idx]
        path = os.path.join(self.data_root, f"{subject}.pt")
        return torch.load(path, map_location="cpu", weights_only=False)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._load_sample(idx)

        # Clone if preloaded to avoid modifying cached data
        if self.preloaded_data is not None:
            x = x.clone()

        # Apply transform if specified
        if self.transform is not None:
            x = self.transform(x)

        # Use pre-cached label tensor (memory optimization)
        diagnosis = self.meta_data.Diagnosis.iloc[idx]
        y = self._label_cache[diagnosis].clone()

        return x, y

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.meta_data)


class SVMDataset(Dataset):
    """
    Dataset class for SVM feature extraction from preprocessed segmentation volumes.

    This dataset loads preprocessed segmentation volume features (from DataPrepaSVM)
    and returns both features and one-hot encoded labels for SVM training.

    Parameters
    ----------
    data_root : str
        Directory containing the preprocessed .pt files.
    meta_data : pd.DataFrame
        Metadata DataFrame containing subject information.
    diseases : list[str]
        List of disease labels (e.g., ['CN', 'AD', 'FTD']).
    device : torch.device or str
        The device to load the data onto.
    """

    def __init__(
        self,
        data_root: str,
        meta_data: pd.DataFrame,
        diseases: list[str],
        device: torch.device | str = "cpu",
    ):
        super(SVMDataset, self).__init__()
        self.data_root = data_root
        self.meta_data = meta_data
        self.device = torch.device(device) if isinstance(device, str) else device
        self.diseases = diseases

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.load(
            os.path.join(self.data_root, f"{self.meta_data.Subject.iloc[idx]}.pt"),
            map_location=self.device,
            weights_only=False,
        )

        disease = self.meta_data.Diagnosis.iloc[idx]
        y = torch.zeros(len(self.diseases), device=self.device)
        y[self.diseases.index(disease)] = 1.0

        x = x.to(self.device)

        return x, y

    def __len__(self) -> int:
        return len(self.meta_data)


class MRIMixUp(Dataset):
    """MixUp augmentation for 3D MRI. Reference: https://arxiv.org/abs/1710.09412"""

    def __init__(
        self,
        dataset: Dataset,
        num_samples: int,
        alpha: float,
        mixup_prob: float,
        transform: Optional[torch.nn.Module] = None,
        seed: Optional[int] = None,
    ):
        super(MRIMixUp, self).__init__()
        assert 0 < alpha < 1, "alpha should be between 0 and 1"
        assert 0 <= mixup_prob <= 1, "mixup_prob should be between 0 and 1"
        assert num_samples > 0, "num_samples should be greater than 0"
        self.dataset = dataset
        self.num_samples = num_samples
        self.alpha = alpha
        self.mixup_prob = mixup_prob

        self.dist = Beta(torch.tensor([alpha]), torch.tensor([alpha]))
        self.transform = transform
        self.seed = int(seed) if seed is not None else None
        self._current_epoch = 0

        # Precompute indices grouped by class
        self.class_indices = {
            cls: torch.tensor(list(meta.index), dtype=torch.long)
            for cls, meta in dataset.meta_data.groupby("Diagnosis")
        }
        self.class_list = list(self.class_indices.keys())

        self._regenerate_mixup_params()

    def _regenerate_mixup_params(self):
        """Pre-generate random decisions for the epoch (lambda sampled on-the-fly)."""
        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed((self.seed + self._current_epoch) % _MAX_UINT32)

        self.mixup_decisions = (
            torch.rand(self.num_samples, generator=generator) > self.mixup_prob
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.seed is not None:
            seed = int((self.seed + self._current_epoch + idx) % _MAX_UINT32)
            rng = np.random.RandomState(seed)
            do_skip = bool(rng.rand() > self.mixup_prob)
            if do_skip:
                sample, target = self.dataset[idx]
                if self.transform is not None:
                    sample = self.transform(sample)
                return sample, target
        else:
            if self.mixup_decisions[idx]:
                sample, target = self.dataset[idx]
                if self.transform is not None:
                    sample = self.transform(sample)
                return sample, target

        # Get first sample
        sample1, target1 = self.dataset[idx]

        # Get its class
        cls1 = self.dataset.meta_data.Diagnosis.iloc[idx]

        # Randomly sample from a different class.
        if self.seed is not None:
            # Use the same numpy RNG to pick partner and alpha deterministically
            available_classes = [cls for cls in self.class_list if cls != cls1]
            cls2_idx = int(rng.randint(0, len(available_classes)))
            cls2 = available_classes[cls2_idx]
            cls2_indices = self.class_indices[cls2]
            idx2_pos = int(rng.randint(0, len(cls2_indices)))
            idx2 = int(cls2_indices[idx2_pos].item())
            sample2, target2 = self.dataset[idx2]

            # Sample alpha from Beta using numpy RNG
            alpha = float(rng.beta(self.alpha, self.alpha))
        else:
            # Uses Python's random module (seeded via worker_init_fn in DataLoader)
            available_classes = [cls for cls in self.class_list if cls != cls1]
            cls2 = random.choice(available_classes)
            cls2_indices = self.class_indices[cls2]
            idx2 = cls2_indices[random.randint(0, len(cls2_indices) - 1)].item()
            sample2, target2 = self.dataset[idx2]

            # Sample alpha on-the-fly using torch.distributions (worker RNG)
            alpha = self.dist.sample().item()

        # Clone before in-place ops to avoid corrupting cached data in the underlying dataset.
        sample1 = sample1.clone()
        target1 = target1.clone()
        sample1.mul_(alpha).add_(sample2, alpha=(1 - alpha))
        target1.mul_(alpha).add_(target2, alpha=(1 - alpha))

        if self.transform is not None:
            sample1 = self.transform(sample1)

        return sample1, target1

    def __len__(self) -> int:
        """Return the number of samples to generate."""
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Regenerate mixup parameters for a new epoch."""
        self._current_epoch = int(epoch)
        self._regenerate_mixup_params()


class MRICutMix(Dataset):
    """CutMix augmentation for 3D MRI. Reference: https://arxiv.org/abs/1905.04899"""

    def __init__(
        self,
        dataset: Dataset,
        num_samples: int,
        alpha: float,
        cutmix_prob: float,
        transform: Optional[callable] = None,
        seed: Optional[int] = None,
    ):
        super(MRICutMix, self).__init__()
        assert 0 < alpha, "alpha should be greater than 0"
        assert 0 <= cutmix_prob <= 1, "cutmix_prob should be between 0 and 1"
        assert num_samples > 0, "num_samples should be greater than 0"
        self.dataset = dataset
        self.num_samples = num_samples
        self.alpha = alpha
        self.cutmix_prob = cutmix_prob
        self.dist = Beta(
            torch.tensor([alpha]), torch.tensor([alpha])
        )  # Beta distribution
        self.transform = transform
        self.seed = int(seed) if seed is not None else None
        self._current_epoch = 0

        # Precompute indices grouped by class (same optimization as MixUp)
        self.class_indices = {
            cls: torch.tensor(list(meta.index), dtype=torch.long)
            for cls, meta in dataset.meta_data.groupby("Diagnosis")
        }
        self.class_list = list(self.class_indices.keys())

        self._regenerate_cutmix_params()

    def _regenerate_cutmix_params(self):
        """Pre-generate random decisions for the epoch (lambda sampled on-the-fly)."""
        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed((self.seed + self._current_epoch) % _MAX_UINT32)

        self.cutmix_decisions = (
            torch.rand(self.num_samples, generator=generator) > self.cutmix_prob
        )

    def _compute_cuboid_bounds(
        self,
        shape: tuple[int, int, int, int],
        lam: float,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[int, int, int, int, int, int]:
        """Compute cuboid boundaries for CutMix.

        Parameters
        ----------
        shape : tuple[int, int, int, int]
            Shape of the input sample (C, D, W, H).
        lam : float
            Mix ratio sampled from Beta distribution.
        generator : Optional[torch.Generator], optional
            Torch generator for deterministic sampling (default: None).

        Returns
        -------
        tuple[int, int, int, int, int, int]
            Start and end coordinates of the cuboid (depth_start, width_start,
            height_start, depth_end, width_end, height_end).
        """
        _, D, W, H = shape
        cut_ratio = float(torch.sqrt(torch.tensor(1.0 - lam)).item())
        cut_d = int(D * cut_ratio)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)

        if generator is not None:
            # Deterministic path using torch generator
            cd = int(torch.randint(0, D + 1, (1,), generator=generator).item())
            cx = int(torch.randint(0, W + 1, (1,), generator=generator).item())
            cy = int(torch.randint(0, H + 1, (1,), generator=generator).item())
        else:
            # Non-deterministic path using Python's random (respects worker_init_fn)
            cd = random.randint(0, D)
            cx = random.randint(0, W)
            cy = random.randint(0, H)

        depth_start = max(0, cd - cut_d // 2)
        width_start = max(0, cx - cut_w // 2)
        height_start = max(0, cy - cut_h // 2)
        depth_end = min(D, cd + cut_d // 2)
        width_end = min(W, cx + cut_w // 2)
        height_end = min(H, cy + cut_h // 2)

        return depth_start, width_start, height_start, depth_end, width_end, height_end

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform CutMix on the dataset.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing the mixed sample and the mixed target.
        """
        # Check if we should skip CutMix for this sample
        if self.seed is not None:
            seed = int((self.seed + self._current_epoch + idx) % _MAX_UINT32)
            rng = np.random.RandomState(seed)
            do_skip = bool(rng.rand() > self.cutmix_prob)
            if do_skip:
                sample, target = self.dataset[idx]
                if self.transform is not None:
                    sample = self.transform(sample)
                return sample, target
        else:
            if self.cutmix_decisions[idx]:
                sample, target = self.dataset[idx]
                if self.transform is not None:
                    sample = self.transform(sample)
                return sample, target

        # Get first sample
        sample1, target1 = self.dataset[idx]

        # Must clone before in-place modification to avoid corrupting cached data
        sample1 = sample1.clone()

        if self.seed is not None:
            # Deterministic path: use numpy RNG for partner selection (reuse seed from above)
            # Note: rng is already initialized above

            # Select partner from different class
            cls2_idx = int(rng.randint(0, len(self.class_list)))
            cls2 = self.class_list[cls2_idx]
            cls2_indices = self.class_indices[cls2]
            idx2_pos = int(rng.randint(0, len(cls2_indices)))
            idx2 = int(cls2_indices[idx2_pos].item())
            sample2, target2 = self.dataset[idx2]

            # Sample lambda from Beta distribution using numpy RNG
            lam = float(rng.beta(self.alpha, self.alpha))

            # Compute cuboid bounds using torch generator for consistency
            gen = torch.Generator()
            gen.manual_seed(seed)
            depth_start, width_start, height_start, depth_end, width_end, height_end = (
                self._compute_cuboid_bounds(sample1.shape, lam, generator=gen)
            )
        else:
            # Non-deterministic path: use Python's random module (respects worker_init_fn)
            cls2 = random.choice(self.class_list)
            cls2_indices = self.class_indices[cls2]
            idx2 = cls2_indices[random.randint(0, len(cls2_indices) - 1)].item()
            sample2, target2 = self.dataset[idx2]

            # Sample lambda from Beta distribution on-the-fly
            lam = self.dist.sample().item()

            # Compute cuboid bounds using Python's random
            depth_start, width_start, height_start, depth_end, width_end, height_end = (
                self._compute_cuboid_bounds(sample1.shape, lam)
            )

        # Replace the corresponding region in sample1 with sample2
        sample1[
            :, depth_start:depth_end, width_start:width_end, height_start:height_end
        ] = sample2[
            :, depth_start:depth_end, width_start:width_end, height_start:height_end
        ]

        cuboid_volume = (
            (depth_end - depth_start)
            * (width_end - width_start)
            * (height_end - height_start)
        )
        total_volume = sample1.shape[1] * sample1.shape[2] * sample1.shape[3]
        actual_lam = 1 - (cuboid_volume / total_volume)

        target = actual_lam * target1 + (1 - actual_lam) * target2

        if self.transform is not None:
            sample1 = self.transform(sample1)

        return sample1, target

    def __len__(self) -> int:
        """Return the number of samples to generate."""
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Regenerate cutmix parameters for a new epoch."""
        self._current_epoch = int(epoch)
        self._regenerate_cutmix_params()
