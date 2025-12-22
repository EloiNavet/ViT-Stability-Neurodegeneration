"""
Swin Transformer V1 3D with Deformable Point Learning (DPL).

This module provides the public interface for the SwinDPL architecture.
The implementation details are proprietary and not included in this repository.

For access to the full implementation, please contact the authors.
"""

import torch
import torch.nn as nn


class _SwinTransformerStub(nn.Module):
    """Stub class when implementation is not available."""

    def __init__(self, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "SwinDPL implementation is not publicly available. "
            "Please contact the authors for access to the full implementation."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


SwinTransformerT = _SwinTransformerStub
SwinTransformerS = _SwinTransformerStub
SwinTransformerB = _SwinTransformerStub
SwinTransformerL = _SwinTransformerStub

__all__ = [
    "SwinTransformerT",
    "SwinTransformerS",
    "SwinTransformerB",
    "SwinTransformerL",
]
