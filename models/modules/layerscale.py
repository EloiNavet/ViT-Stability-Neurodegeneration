"""
LayerScale module for stabilizing deep transformers.

Based on:
    Touvron et al., "Going Deeper with Image Transformers", ICCV 2021
    https://arxiv.org/abs/2103.17239

LayerScale adds a learnable, per-channel scaling parameter after each residual
branch in a transformer block. This stabilizes very deep transformers by starting
close to the identity function and letting each residual branch progressively
scale during training.

Unlike ReZero (one scalar per block), LayerScale uses one learnable gain per
channel, improving optimization flexibility while keeping negligible parameter overhead.
"""

import torch
import torch.nn as nn
from typing import Optional


class LayerScale(nn.Module):
    """Per-channel learnable scaling for transformer residual branches.

    Applies element-wise multiplication with a learnable diagonal matrix:
        output = diag(gamma) * input

    where gamma is initialized to a small value (e.g., 1e-5 or 1e-6) to start
    close to the identity function.

    Args:
        dim: Number of channels/embedding dimension
        init_value: Initial value for the scaling parameters (default: 1e-5)

    Example:
        >>> layer_scale = LayerScale(768, init_value=1e-5)
        >>> x = torch.randn(2, 196, 768)  # [B, N, C]
        >>> out = layer_scale(x)  # Element-wise scale by learnable gamma
    """

    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-channel scaling.

        Args:
            x: Input tensor of any shape where last dimension is `dim`

        Returns:
            Scaled tensor with same shape as input
        """
        return self.gamma * x


def create_layerscale(
    dim: int, layer_scale: bool = False, init_value: float = 1e-5
) -> Optional[nn.Module]:
    """Factory function to create LayerScale module or None.

    Args:
        dim: Number of channels/embedding dimension
        layer_scale: Whether to enable LayerScale
        init_value: Initial value for the scaling parameters

    Returns:
        LayerScale module if enabled, otherwise None

    Example:
        >>> ls = create_layerscale(768, layer_scale=True, init_value=1e-5)
        >>> if ls is not None:
        >>>     x = ls(x)  # Apply scaling
        >>> else:
        >>>     pass  # No scaling applied
    """
    if layer_scale:
        return LayerScale(dim, init_value=init_value)
    return None
