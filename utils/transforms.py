import torch
from monai.transforms import Transform, MapTransform
from typing import Hashable, Mapping, Any


class AdaptiveGaussianNoise(Transform):
    """
    Applies temporary normalization, adds Gaussian noise, then restores original scale.

    Uses PyTorch's random state for full reproducibility. All randomness (probability
    check and noise generation) is controlled by torch RNG state.
    """

    def __init__(self, prob: float = 0.1, noise_factor: float = 0.1):
        super().__init__()
        self.prob = prob
        self.noise_factor = noise_factor

    def __call__(self, img):
        if torch.rand(1).item() < self.prob:
            orig_min = torch.min(img)
            orig_max = torch.max(img)

            img_normalized = (img - orig_min) / (orig_max - orig_min + 1e-8)

            noise = torch.randn_like(img_normalized) * self.noise_factor
            img_normalized = img_normalized + noise

            img = img_normalized * (orig_max - orig_min) + orig_min

        return img


class AdaptiveRicianNoise(Transform):
    """
    Applies Rician noise while preserving the original image scale.
    Rician noise is the actual noise distribution in magnitude MR images.

    Note: Uses torch's global random state for reproducibility.
    Ensure torch is seeded for deterministic behavior.
    """

    def __init__(self, prob: float = 0.1, noise_factor: float = 0.1):
        super().__init__()
        self.prob = prob
        self.noise_factor = noise_factor

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.prob:
            orig_min = torch.min(img)
            orig_max = torch.max(img)

            img_normalized = (img - orig_min) / (orig_max - orig_min)

            # Generate Rician noise
            # Rician noise is sqrt((v + n1)^2 + n2^2) where v is the signal
            # and n1, n2 are independent Gaussian noise components
            sigma = self.noise_factor * torch.mean(img_normalized)
            n1 = torch.randn_like(img_normalized) * sigma
            n2 = torch.randn_like(img_normalized) * sigma

            img_noisy = torch.sqrt((img_normalized + n1) ** 2 + n2**2)
            img = img_noisy * (orig_max - orig_min) + orig_min
            img = torch.clamp(img, min=orig_min, max=orig_max)

        return img


class AdaptiveGaussianNoiseD(MapTransform):
    """
    Dictionary-aware version.
    Applies temporary normalization, adds Gaussian noise, then restores original scale.

    Uses PyTorch's random state for full reproducibility.
    """

    def __init__(
        self,
        keys: list[Hashable],
        prob: float = 0.1,
        noise_factor: float = 0.1,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.prob = prob
        self.noise_factor = noise_factor

    def __call__(self, data: Mapping[Hashable, Any]) -> Mapping[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if torch.rand(1).item() < self.prob:
                img = d[key]
                if not isinstance(img, torch.Tensor):
                    raise TypeError(
                        f"Image associated with key '{key}' must be a PyTorch tensor."
                    )

                orig_min = torch.min(img)
                orig_max = torch.max(img)

                if orig_max == orig_min:
                    d[key] = img
                else:
                    img_normalized = (img - orig_min) / (orig_max - orig_min)

                    noise = torch.randn_like(img_normalized) * self.noise_factor
                    img_normalized_noisy = img_normalized + noise

                    d[key] = img_normalized_noisy * (orig_max - orig_min) + orig_min
        return d


class AdaptiveRicianNoiseD(MapTransform):
    """
    Dictionary-aware version.
    Applies Rician noise while preserving the original image scale.
    Rician noise is the actual noise distribution in magnitude MR images.

    Note: Uses torch's global random state for reproducibility.
    """

    def __init__(
        self,
        keys: list[Hashable],
        prob: float = 0.1,
        noise_factor: float = 0.1,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.prob = prob
        self.noise_factor = noise_factor

    def __call__(self, data: Mapping[Hashable, Any]) -> Mapping[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            if torch.rand(1).item() < self.prob:
                img = d[key]
                if not isinstance(img, torch.Tensor):
                    raise TypeError(
                        f"Image associated with key '{key}' must be a PyTorch tensor."
                    )

                orig_min = torch.min(img)
                orig_max = torch.max(img)

                if orig_max == orig_min:
                    img_normalized = (
                        torch.zeros_like(img) if orig_max == 0 else torch.ones_like(img)
                    )
                else:
                    img_normalized = (img - orig_min) / (orig_max - orig_min)

                mean_signal = torch.mean(img_normalized)
                if mean_signal <= 0:
                    sigma = self.noise_factor
                else:
                    sigma = self.noise_factor * mean_signal

                if sigma >= 1e-6:
                    n1 = torch.randn_like(img_normalized) * sigma
                    n2 = torch.randn_like(img_normalized) * sigma
                    img_noisy_normalized = torch.sqrt(
                        torch.clamp((img_normalized + n1), min=0) ** 2 + n2**2
                    )

                    if orig_max == orig_min:
                        d[key] = orig_min
                    else:
                        img_denormalized_noisy = (
                            img_noisy_normalized * (orig_max - orig_min) + orig_min
                        )
                        d[key] = torch.clamp(
                            img_denormalized_noisy, min=orig_min, max=orig_max
                        )
        return d
