"""
Shake Drop from https://github.com/owruby/shake-drop_pytorch/blob/master/models/shakedrop.py
"""

import torch
import torch.nn as nn
from typing import Tuple


class _ShakeDropFn(torch.autograd.Function):
    """
    Custom autograd function for ShakeDrop regularization.

    Forward pass:
        - With probability p_drop, scales input by random alpha in [alpha_low, alpha_high].
        - Otherwise, passes input unchanged.
    Backward pass:
        - If dropped, scales gradient by random beta in [0, 1].
        - Otherwise, passes gradient unchanged.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        training: bool,
        p_drop: float,
        alpha_low: float,
        alpha_high: float,
    ) -> torch.Tensor:
        ctx.training = training
        ctx.p_drop = float(p_drop)

        if training:
            gate = (
                torch.rand((), device=x.device, dtype=torch.float32) < (1.0 - p_drop)
            ).to(x.dtype)
            ctx.save_for_backward(gate)

            if gate.item() == 0:
                N = x.shape[0]
                view_shape = (N,) + (1,) * (x.ndim - 1)
                alpha = (
                    torch.empty((N,), device=x.device, dtype=x.dtype)
                    .uniform_(alpha_low, alpha_high)
                    .view(view_shape)
                )
                return alpha * x
            else:
                return x
        else:
            return (1.0 - p_drop) * x

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, None, None]:
        if not ctx.training:
            p_drop = getattr(ctx, "p_drop", 0.5)
            return (1.0 - p_drop) * grad_output, None, None, None, None

        (gate,) = ctx.saved_tensors
        if gate.item() == 0:
            N = grad_output.shape[0]
            view_shape = (N,) + (1,) * (grad_output.ndim - 1)
            beta = (
                torch.empty((N,), device=grad_output.device, dtype=grad_output.dtype)
                .uniform_(0.0, 1.0)
                .view(view_shape)
            )
            return beta * grad_output, None, None, None, None
        else:
            return grad_output, None, None, None, None


class ShakeDrop(nn.Module):
    """
    ShakeDrop regularization layer.

    Args:
        p_drop (float): Probability of dropping the residual branch.
        alpha_range (Tuple[float, float]): Range for random alpha scaling (forward pass).
    """

    def __init__(
        self, p_drop: float = 0.5, alpha_range: Tuple[float, float] = (-1.0, 1.0)
    ) -> None:
        super().__init__()
        self.p_drop = float(p_drop)
        self.alpha_range = (float(alpha_range[0]), float(alpha_range[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a0, a1 = self.alpha_range
        return _ShakeDropFn.apply(x, self.training, self.p_drop, a0, a1)
