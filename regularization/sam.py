"""
Sharpness Aware Minimization from https://github.com/davda54/sam/blob/main/sam.py
"""

import torch
from typing import Callable, Optional, Any, Dict, Type


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer.

    Args:
        params: Iterable of model parameters to optimize.
        base_optimizer: PyTorch optimizer class (e.g., torch.optim.Adam).
        rho: Neighborhood size parameter (default: 0.05).
        adaptive: If True, use adaptive SAM (default: False).
        **kwargs: Additional arguments for the base optimizer.
    """

    def __init__(
        self,
        params,
        base_optimizer: Type[torch.optim.Optimizer],
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """
        Perturb parameters to the sharpness-maximizing direction.

        Args:
            zero_grad: If True, zero gradients after step.
        """
        grad_norm = self._grad_norm()

        # Safety check: skip perturbation if grad_norm is invalid
        if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm == 0:
            if zero_grad:
                self.zero_grad()
            return

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )

                # Additional safety: check if perturbation is valid
                if torch.isnan(e_w).any() or torch.isinf(e_w).any():
                    # Skip perturbation for this parameter
                    continue

                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False, scaler=None) -> None:
        """
        Restore parameters and perform the optimizer update.

        Args:
            zero_grad: If True, zero gradients after step.
            scaler: GradScaler for FP16 training (if used).
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        # do the actual "sharpness-aware" update
        if scaler is None:
            self.base_optimizer.step()
            if zero_grad:
                self.zero_grad()
        else:
            scaler.step(self.base_optimizer)
            scaler.update()
            if zero_grad:
                self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Any]] = None) -> None:
        """
        Perform a SAM optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self) -> torch.Tensor:
        """
        Compute the norm of the gradients.

        Returns:
            Gradient norm as a scalar tensor.
        """
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism

        grad_norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grad_for_norm = (
                        torch.abs(p) if group["adaptive"] else 1.0
                    ) * p.grad
                    param_norm = grad_for_norm.norm(p=2).to(shared_device)
                    # Skip parameters with invalid gradients
                    if not (torch.isnan(param_norm) or torch.isinf(param_norm)):
                        grad_norms.append(param_norm)

        if len(grad_norms) == 0:
            # Return a small positive value instead of 0 to avoid division by zero
            return torch.tensor(1e-12, device=shared_device)

        norm = torch.norm(torch.stack(grad_norms), p=2)

        # Final safety check
        if torch.isnan(norm) or torch.isinf(norm):
            return torch.tensor(1e-12, device=shared_device)

        return norm

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load optimizer state.

        Args:
            state_dict: State dictionary.
        """
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
