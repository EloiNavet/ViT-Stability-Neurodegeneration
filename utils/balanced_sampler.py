"""
Balanced Sampler for Imbalanced Datasets with Distributed Training Support.

This module provides samplers that perform class-balanced sampling by weighting
samples according to the inverse of their class frequency. This helps mitigate
class imbalance during training.

Classes
-------
DistributedWeightedSampler
    A distributed sampler that performs weighted sampling with replacement.
"""

import math
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from typing import Optional, Iterator
import pandas as pd
import numpy as np


def compute_class_weights(
    metadata: pd.DataFrame,
    diagnosis_column: str = "Diagnosis",
    normalize: bool = True,
) -> dict[str, float]:
    """
    Compute inverse frequency class weights from metadata.

    Parameters
    ----------
    metadata : pd.DataFrame
        DataFrame containing sample metadata with diagnosis labels.
    diagnosis_column : str, optional
        Name of the column containing class labels (default: "Diagnosis").
    normalize : bool, optional
        Whether to normalize weights so the minimum weight is 1.0 (default: True).
        This improves numerical stability.

    Returns
    -------
    dict[str, float]
        Dictionary mapping class labels to their inverse frequency weights.

    Examples
    --------
    >>> metadata = pd.DataFrame({'Diagnosis': ['CN', 'CN', 'AD', 'FTD']})
    >>> weights = compute_class_weights(metadata)
    >>> print(weights)
    {'CN': 1.0, 'AD': 2.0, 'FTD': 2.0}
    """
    # Validate input
    if len(metadata) == 0:
        raise ValueError("Cannot compute class weights for empty metadata")

    if diagnosis_column not in metadata.columns:
        raise ValueError(f"Column '{diagnosis_column}' not found in metadata")

    # Count samples per class
    class_counts = metadata[diagnosis_column].value_counts()

    # Compute inverse frequency weights
    total_samples = len(metadata)
    class_weights = {}

    for cls, count in class_counts.items():
        # Weight = total_samples / (num_classes * count)
        # This ensures that expected samples per class per epoch = total_samples / num_classes
        weight = total_samples / (len(class_counts) * count)
        class_weights[cls] = weight

    if normalize:
        # Normalize so minimum weight is 1.0 for numerical stability
        min_weight = min(class_weights.values())
        class_weights = {cls: w / min_weight for cls, w in class_weights.items()}

    return class_weights


def compute_sample_weights(
    metadata: pd.DataFrame,
    class_weights: dict[str, float],
    diagnosis_column: str = "Diagnosis",
) -> np.ndarray:
    """
    Assign weights to each sample based on its class.

    Parameters
    ----------
    metadata : pd.DataFrame
        DataFrame containing sample metadata with diagnosis labels.
    class_weights : dict[str, float]
        Dictionary mapping class labels to their weights.
    diagnosis_column : str, optional
        Name of the column containing class labels (default: "Diagnosis").

    Returns
    -------
    np.ndarray
        Array of sample weights, one per sample in the metadata.

    Examples
    --------
    >>> metadata = pd.DataFrame({'Diagnosis': ['CN', 'AD', 'CN']})
    >>> class_weights = {'CN': 1.0, 'AD': 2.0}
    >>> weights = compute_sample_weights(metadata, class_weights)
    >>> print(weights)
    [1.0, 2.0, 1.0]
    """
    # Validate that all diagnoses have corresponding weights
    unique_diagnoses = set(metadata[diagnosis_column].unique())
    missing_classes = unique_diagnoses - set(class_weights.keys())
    if missing_classes:
        raise ValueError(
            f"Found diagnoses in metadata not present in class_weights: {missing_classes}"
        )

    sample_weights = np.array(
        [class_weights[diagnosis] for diagnosis in metadata[diagnosis_column]],
        dtype=np.float32,
    )
    return sample_weights


class DistributedWeightedSampler(Sampler):
    """
    Distributed sampler that performs weighted sampling with replacement.

    This sampler combines the functionality of PyTorch's WeightedRandomSampler
    with DistributedSampler to enable balanced sampling in distributed training.
    Each rank samples from its own partition of the dataset using class weights.

    Parameters
    ----------
    dataset : Dataset
        The dataset to sample from.
    weights : np.ndarray or torch.Tensor
        Weight for each sample in the dataset.
    num_samples : int, optional
        Number of samples to draw per epoch per rank. If None, defaults to
        len(dataset) // world_size.
    replacement : bool, optional
        Whether to sample with replacement (default: True). For balanced sampling,
        this should typically be True.
    num_replicas : int, optional
        Number of processes participating in distributed training. If None,
        retrieved from the current distributed group.
    rank : int, optional
        Rank of the current process. If None, retrieved from the current
        distributed group.
    seed : int, optional
        Random seed for reproducibility (default: 0).
    drop_last : bool, optional
        Whether to drop the last incomplete batch (default: False).

    Attributes
    ----------
    num_replicas : int
        Number of distributed processes.
    rank : int
        Rank of the current process.
    epoch : int
        Current epoch number (used for seeding).
    num_samples : int
        Number of samples to draw per epoch on this rank.
    total_size : int
        Total number of samples across all ranks.

    Notes
    -----
    - Sampling is performed with replacement by default to ensure class balance.
    - Each rank gets approximately equal number of samples.
    - Call set_epoch() at the beginning of each epoch for deterministic shuffling.
    - Compatible with gradient accumulation and multi-GPU training.

    Examples
    --------
    >>> # Compute class weights from training metadata
    >>> class_weights = compute_class_weights(train_metadata)
    >>> sample_weights = compute_sample_weights(train_metadata, class_weights)
    >>>
    >>> # Create balanced sampler
    >>> sampler = DistributedWeightedSampler(
    ...     train_dataset,
    ...     weights=sample_weights,
    ...     num_samples=len(train_dataset),
    ...     seed=42
    ... )
    >>>
    >>> # Use in DataLoader
    >>> train_loader = DataLoader(
    ...     train_dataset,
    ...     batch_size=32,
    ...     sampler=sampler,
    ...     num_workers=4
    ... )
    >>>
    >>> # Set epoch for each training epoch
    >>> for epoch in range(num_epochs):
    ...     sampler.set_epoch(epoch)
    ...     for batch in train_loader:
    ...         # training code
    ...         pass
    """

    def __init__(
        self,
        dataset,
        weights: np.ndarray | torch.Tensor,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
    ):
        # Distributed setup
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank() if dist.is_initialized() else 0

        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.replacement = replacement
        self.seed = seed

        # Convert weights to tensor if needed
        if isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights).float()
        elif isinstance(weights, torch.Tensor):
            weights = weights.float()
        else:
            raise TypeError(
                f"weights should be np.ndarray or torch.Tensor, got {type(weights)}"
            )

        if len(weights) != len(dataset):
            raise ValueError(
                f"Length of weights ({len(weights)}) must match dataset size ({len(dataset)})"
            )

        # Validate dataset is not empty
        if len(dataset) == 0:
            raise ValueError("Cannot create sampler for empty dataset")

        self.weights = weights

        # Validate weights are valid for sampling
        if torch.any(torch.isnan(self.weights)) or torch.any(torch.isinf(self.weights)):
            raise ValueError("Weights contain NaN or Inf values")

        if torch.any(self.weights < 0):
            raise ValueError("Weights must be non-negative for weighted sampling")

        if torch.sum(self.weights) == 0:
            raise ValueError(
                "Sum of weights is zero - cannot perform weighted sampling"
            )

        # Calculate number of samples per rank
        if num_samples is None:
            # Default: distribute dataset evenly across ranks
            if self.drop_last and len(self.dataset) % self.num_replicas != 0:
                # Drop last to maintain equal size across ranks
                self.num_samples = math.floor(len(self.dataset) / self.num_replicas)
            else:
                # Pad last rank if needed
                self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        else:
            self.num_samples = num_samples

        self.total_size = self.num_samples * self.num_replicas

        # Validate parameters
        if not self.replacement and self.num_samples > len(self.dataset):
            raise ValueError(
                f"Cannot sample {self.num_samples} samples without replacement "
                f"from dataset of size {len(self.dataset)}"
            )

    def __iter__(self) -> Iterator[int]:
        """
        Generate sample indices for the current rank.

        Yields
        ------
        int
            Sample indices for the current rank.
        """
        # Set random seed for this epoch (deterministic if seed is set)
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        # Sample indices using weighted sampling
        # This samples from the entire dataset with class weights
        if self.replacement:
            # Sample with replacement using multinomial sampling
            # This is the standard approach for balanced sampling
            indices = torch.multinomial(
                self.weights,
                self.total_size,
                replacement=True,
                generator=generator,
            ).tolist()
        else:
            # Sample without replacement (rare case, not recommended for balancing)
            # Create indices repeated by their weights (floor)
            # This is a simplified approach and may not give exact distribution
            rand_tensor = torch.rand(len(self.weights), generator=generator)
            indices = torch.argsort(rand_tensor / self.weights, descending=True)[
                : self.total_size
            ].tolist()

        # Partition indices for this rank
        # Each rank gets a contiguous slice of the sampled indices
        indices = indices[self.rank :: self.num_replicas]

        # Ensure we have exactly num_samples (in case of rounding issues)
        assert len(indices) == self.num_samples, (
            f"Sampler produced {len(indices)} samples but expected {self.num_samples}. "
            f"This should not happen - please report this bug."
        )

        return iter(indices)

    def __len__(self) -> int:
        """
        Return the number of samples per rank.

        Returns
        -------
        int
            Number of samples this rank will produce per epoch.
        """
        return self.num_samples

    def set_epoch(self, epoch: int):
        """
        Set the epoch for this sampler.

        This ensures deterministic shuffling across epochs when using a seed.
        Should be called at the start of each epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        """
        self.epoch = epoch


def create_balanced_sampler(
    dataset,
    metadata: pd.DataFrame,
    num_samples: Optional[int] = None,
    diagnosis_column: str = "Diagnosis",
    seed: int = 0,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
    drop_last: bool = False,
) -> DistributedWeightedSampler:
    """
    Factory function to create a balanced distributed sampler.

    This is a convenience function that combines class weight computation
    and sampler creation into a single call.

    Parameters
    ----------
    dataset : Dataset
        The dataset to sample from.
    metadata : pd.DataFrame
        DataFrame containing sample metadata with diagnosis labels.
    num_samples : int, optional
        Number of samples per epoch per rank. If None, defaults to
        len(dataset) // world_size.
    diagnosis_column : str, optional
        Name of the column containing class labels (default: "Diagnosis").
    seed : int, optional
        Random seed for reproducibility (default: 0).
    num_replicas : int, optional
        Number of distributed processes. If None, auto-detected.
    rank : int, optional
        Rank of current process. If None, auto-detected.
    drop_last : bool, optional
        Whether to drop the last incomplete batch (default: False).

    Returns
    -------
    DistributedWeightedSampler
        Configured balanced sampler ready for use in DataLoader.

    Examples
    --------
    >>> sampler = create_balanced_sampler(
    ...     train_dataset,
    ...     train_metadata,
    ...     seed=42
    ... )
    >>> train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    """
    # Compute class weights
    class_weights = compute_class_weights(metadata, diagnosis_column=diagnosis_column)

    # Compute sample weights
    sample_weights = compute_sample_weights(
        metadata, class_weights, diagnosis_column=diagnosis_column
    )

    # Create sampler
    sampler = DistributedWeightedSampler(
        dataset=dataset,
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,  # Always use replacement for balanced sampling
        num_replicas=num_replicas,
        rank=rank,
        seed=seed,
        drop_last=drop_last,
    )

    return sampler
