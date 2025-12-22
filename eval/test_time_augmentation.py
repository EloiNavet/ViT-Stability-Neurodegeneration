import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import numpy as np
from monai.transforms import (
    Compose,
    RandFlip,
    RandAffine,
    CenterSpatialCrop,
    Resize,
)


class TestTimeAugmentation:
    """
    Test-Time Augmentation wrapper for 3D medical imaging models.

    This class applies multiple augmentations to input samples during inference,
    generates predictions for each augmented version, and averages the results
    to produce a final robust prediction.

    Expected number of augmentations per sample:
    - Identity (no augmentation): 1
    - Flip (deterministic, spatial_axis=0): 1 if use_flip=True
    - Affine (stochastic): num_samples if use_affine=True
    - Center crop: 1 if use_scaled_center_crop=True

    Example with defaults (num_samples=5, all enabled):
    Total = 1 (identity) + 1 (flip) + 5 (affine) + 1 (crop) = 8 augmentations

    Parameters
    ----------
    model : nn.Module
        The trained model to use for predictions.
    device : torch.device
        Device to run predictions on (CPU or CUDA).
    num_samples : int, optional
        Number of stochastic augmented samples to generate per input (default: 5).
        Higher values give more robust predictions but increase inference time.
        Only applies to stochastic transforms (e.g., RandAffine).
    use_flip : bool, optional
        Whether to include flip along spatial_axis=0 (D dimension) (default: True).
        Applied once (deterministic).
    use_affine : bool, optional
        Whether to include small affine perturbations (default: True).
        Applied num_samples times (stochastic).
    use_scaled_center_crop : bool, optional
        Whether to use center-crop augmentation (default: True).
        Applied once with crop_roi_scale factor.
    crop_roi_scale : float, optional
        Scale factor for crop ROI relative to input size (default: 0.9).
        Only used if use_scaled_center_crop=True.
    affine_rotate_range : Tuple[float, float, float], optional
        Rotation range in degrees for each axis (default: ±3°).
    affine_translate_range : Tuple[float, float, float], optional
        Translation range in voxels for each axis (default: ±5 voxels).
    target_shape : Optional[Tuple[int, int, int]], optional
        Target shape for resizing after crops (default: None).
        If None, uses the shape of the first input.
    use_amp : bool, optional
        Whether to use automatic mixed precision (default: False).
    use_channels_last : bool, optional
        Whether to convert inputs to channels_last_3d before inference (default: True).
        Disable for architectures that do not support channels_last (e.g., MedViT/NATTEN).
    use_entropy_weighting : bool, optional
        Whether to weight predictions by inverse entropy (default: True).
        If True, predictions with lower entropy (higher confidence) get more weight.
        If False, all predictions are averaged uniformly.

    Notes
    -----
    - Input tensors should be on CPU for efficient augmentation; device transfers
      are handled internally for each augmented sample.
    - Flip is deterministic (prob=1.0) so applied only once.
    - Affine is stochastic (random rotations/translations) so applied num_samples times.
    - Entropy weighting: H(p) = -sum(p * log(p)). Lower entropy = higher confidence = higher weight.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_samples: int = 5,
        use_flip: bool = True,
        use_affine: bool = True,
        use_scaled_center_crop: bool = True,
        crop_roi_scale: float = 0.9,
        affine_rotate_range: Tuple[float, float, float] = (3.0, 3.0, 3.0),
        affine_translate_range: Tuple[float, float, float] = (5.0, 5.0, 5.0),
        target_shape: Optional[Tuple[int, int, int]] = None,
        use_amp: bool = False,
        use_channels_last: bool = True,
        use_entropy_weighting: bool = True,
    ):
        self.model = model
        self.device = device
        self.num_samples = num_samples
        self.use_flip = use_flip
        self.use_affine = use_affine
        self.use_scaled_center_crop = use_scaled_center_crop
        self.crop_roi_scale = crop_roi_scale
        self.target_shape = target_shape
        self.use_amp = use_amp
        self.use_channels_last = use_channels_last
        self.use_entropy_weighting = use_entropy_weighting

        # Build augmentation transforms
        self.transforms = self._build_transforms(
            affine_rotate_range, affine_translate_range
        )

    def _build_transforms(
        self,
        rotate_range: Tuple[float, float, float],
        translate_range: Tuple[float, float, float],
    ) -> List[Compose]:
        """
        Build a list of augmentation transforms based on configuration.

        Parameters
        ----------
        rotate_range : Tuple[float, float, float]
            Rotation range in degrees for each spatial axis.
        translate_range : Tuple[float, float, float]
            Translation range in voxels for each spatial axis.

        Returns
        -------
        List[Compose]
            List of MONAI Compose transforms to apply during TTA.
        """
        transforms_list = []

        # Always include identity (no augmentation)
        transforms_list.append(Compose([]))

        # Flip augmentation (deterministic, prob=1.0 means always flip)
        # For brain MRI: spatial_axis=0 corresponds to the D dimension (C, D, H, W)
        # This is applied once (deterministic flip)
        if self.use_flip:
            transforms_list.append(
                Compose(
                    [
                        RandFlip(
                            prob=1.0, spatial_axis=0
                        )  # spatial_axis=0 flips along D axis
                    ]
                )
            )

        # Small affine perturbations (stochastic, generates random transformations)
        # This will be applied multiple times (num_samples) to get diverse augmentations
        if self.use_affine:
            # Convert degrees to radians for MONAI
            rotate_range_rad = tuple(np.deg2rad(r) for r in rotate_range)

            transforms_list.append(
                Compose(
                    [
                        RandAffine(
                            prob=1.0,
                            rotate_range=rotate_range_rad,
                            translate_range=translate_range,
                            mode="bilinear",
                            padding_mode="border",
                            scale_range=None,
                        )
                    ]
                )
            )

        return transforms_list

    def _get_crop_transform(self, input_shape: Tuple[int, int, int, int]) -> Compose:
        """
        Get crop transform.

        Parameters
        ----------
        input_shape : Tuple[int, int, int, int]
            Shape of input tensor (C, D, H, W).

        Returns
        -------
        Compose
            MONAI transform for the crop.
        """
        _, D, H, W = input_shape

        crop_d = int(D * self.crop_roi_scale)
        crop_h = int(H * self.crop_roi_scale)
        crop_w = int(W * self.crop_roi_scale)
        roi_size = (crop_d, crop_h, crop_w)

        return Compose([CenterSpatialCrop(roi_size=roi_size)])

    def _compute_entropy(self, probs: torch.Tensor, epsilon: float = 1e-10) -> float:
        """
        Compute the entropy of a probability distribution.

        Parameters
        ----------
        probs : torch.Tensor
            Probability distribution of shape (1, num_classes) or (num_classes,).
        epsilon : float, optional
            Small constant to avoid log(0), default 1e-10.

        Returns
        -------
        float
            Entropy value. Higher entropy = more uncertain prediction.
        """
        # Ensure probs is 1D
        if probs.ndim > 1:
            probs = probs.squeeze()

        probs = torch.clamp(probs, min=epsilon)
        entropy = -torch.sum(probs * torch.log(probs))
        return entropy

    def _apply_tta_single_sample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply TTA to a single sample and return averaged predictions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (C, D, H, W).

        Returns
        -------
        torch.Tensor
            Averaged or entropy-weighted softmax predictions of shape (num_classes,).
        """
        all_predictions = []
        all_entropies = []  # Track entropy for each prediction
        original_shape = x.shape

        # Store target shape for resizing after crops
        if self.target_shape is None:
            target_spatial_shape = original_shape[1:]  # (D, H, W)
        else:
            target_spatial_shape = self.target_shape

        # Apply each base transform multiple times (for stochastic transforms)
        for transform in self.transforms:
            # Detect if transform contains stochastic operations that should be applied multiple times
            is_stochastic = False
            if hasattr(transform, "transforms") and len(transform.transforms) > 0:
                for t in transform.transforms:
                    if hasattr(t, "prob"):
                        # It's a MONAI RandXXX transform
                        # RandAffine is stochastic even with prob=1.0 (random rotations/translations)
                        # RandFlip with prob=1.0 is deterministic (always flips)
                        if "Affine" in t.__class__.__name__ and t.prob > 0:
                            is_stochastic = True
                            break
                        # For other Rand transforms, only consider stochastic if prob < 1.0
                        elif t.prob < 1.0:
                            is_stochastic = True
                            break

            num_applications = self.num_samples if is_stochastic else 1

            for _ in range(num_applications):
                x_aug = transform(x.clone())

                # Add batch dimension (input is already on CPU from DataLoader)
                x_aug = x_aug.unsqueeze(0).to(self.device, non_blocking=True)

                # Convert to channels_last_3d for better performance
                # Note: Skip this for MedViT architecture
                if self.use_channels_last:
                    try:
                        x_aug = x_aug.to(memory_format=torch.channels_last_3d)
                    except RuntimeError:
                        # If the backend refuses the conversion, disable it for subsequent calls
                        self.use_channels_last = False

                # Forward pass
                with torch.inference_mode():
                    if self.use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            logits = self.model(x_aug)
                    else:
                        logits = self.model(x_aug)

                # Convert to probabilities and accumulate
                probs = torch.softmax(logits, dim=1)
                all_predictions.append(probs)

                # Compute entropy if weighting is enabled
                if self.use_entropy_weighting:
                    entropy = self._compute_entropy(probs)
                    all_entropies.append(entropy)

        # Multi-crop augmentation
        if self.use_scaled_center_crop:
            crop_transform = self._get_crop_transform(original_shape)
            x_cropped = crop_transform(x.clone())
            if x_cropped.shape[1:] != target_spatial_shape:
                resize_transform = Resize(
                    spatial_size=target_spatial_shape, mode="trilinear"
                )
                x_cropped = resize_transform(x_cropped)
            x_cropped = x_cropped.unsqueeze(0).to(self.device, non_blocking=True)
            if self.use_channels_last:
                try:
                    x_cropped = x_cropped.to(memory_format=torch.channels_last_3d)
                except RuntimeError:
                    self.use_channels_last = False
            with torch.inference_mode():
                if self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = self.model(x_cropped)
                else:
                    logits = self.model(x_cropped)
            probs = torch.softmax(logits, dim=1)
            all_predictions.append(probs)

            # Compute entropy for crop augmentation
            if self.use_entropy_weighting:
                entropy = self._compute_entropy(probs)
                all_entropies.append(entropy)

        # Average all predictions efficiently (uniform or entropy-weighted)
        if len(all_predictions) == 1:
            avg_predictions = all_predictions[0].squeeze(0)
        else:
            if self.use_entropy_weighting and len(all_entropies) > 0:
                entropies_tensor = torch.stack(all_entropies)

                # Inverse entropy: w = 1 / (entropy + epsilon)
                # Add small epsilon to avoid division by zero
                epsilon = 1e-6
                weights = 1.0 / (entropies_tensor + epsilon)

                # Normalize weights to sum to 1
                weights = weights / weights.sum()

                # Stack predictions and apply weighted average
                stacked_predictions = torch.stack(all_predictions, dim=0).squeeze(
                    1
                )  # (num_aug, num_classes)
                avg_predictions = torch.sum(
                    stacked_predictions * weights.unsqueeze(1), dim=0
                )
            else:
                # Uniform average (original behavior)
                avg_predictions = (
                    torch.stack(all_predictions, dim=0).mean(dim=0).squeeze(0)
                )

        return avg_predictions

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict with test-time augmentation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, D, H, W) or (C, D, H, W).
            If coming from DataLoader, should already be on CPU to avoid
            redundant device transfers during augmentation.

        Returns
        -------
        torch.Tensor
            Averaged predictions of shape (B, num_classes) or (num_classes,).
        """
        self.model.eval()

        # Handle single sample vs batch
        if x.ndim == 4:
            # Single sample (C, D, H, W) - process directly
            return self._apply_tta_single_sample(x)
        elif x.ndim == 5:
            # Batch (B, C, D, H, W)
            # Move to CPU if not already there to enable efficient augmentation
            if x.device != torch.device("cpu"):
                x = x.cpu()

            batch_predictions = []
            for i in range(x.shape[0]):
                preds = self._apply_tta_single_sample(x[i])
                batch_predictions.append(preds)
            return torch.stack(batch_predictions)
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.ndim}D")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for predict method."""
        return self.predict(x)


def create_tta_predictor(
    model: nn.Module,
    device: torch.device,
    tta_config: dict,
    use_amp: bool = False,
) -> TestTimeAugmentation:
    """
    Factory function to create a TTA predictor from a configuration dict.

    Parameters
    ----------
    model : nn.Module
        Trained model for inference.
    device : torch.device
        Device to run predictions on.
    tta_config : dict
        Configuration dictionary with TTA parameters:
        - num_samples: Number of augmented samples
        - use_flip: Enable sagittal flip
        - use_affine: Enable affine perturbations
        - use_scaled_center_crop: Enable scaled center crop
        - crop_roi_scale: Crop ROI scale factor
        - affine_rotate_range: Rotation range in degrees
        - affine_translate_range: Translation range in voxels
        - target_shape: Target shape for resizing
        - use_entropy_weighting: Enable entropy-based weighting (optional)
    use_amp : bool, optional
        Whether to use automatic mixed precision

    Returns
    -------
    TestTimeAugmentation
        Configured TTA predictor.
    """
    return TestTimeAugmentation(
        model=model,
        device=device,
        num_samples=tta_config["num_samples"],
        use_flip=tta_config["use_flip"],
        use_affine=tta_config["use_affine"],
        use_scaled_center_crop=tta_config["use_scaled_center_crop"],
        crop_roi_scale=tta_config["crop_roi_scale"],
        affine_rotate_range=tta_config["affine_rotate_range"],
        affine_translate_range=tta_config["affine_translate_range"],
        target_shape=tta_config["target_shape"],
        use_amp=use_amp,
        use_channels_last=tta_config["use_channels_last"],
        use_entropy_weighting=tta_config["use_entropy_weighting"],
    )
