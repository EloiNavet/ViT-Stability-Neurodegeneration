"""
DeepScaleLM-style stable transforms for vision transformers.

Based on "Transformers Get Stable" (ICML 2024).
Implements fully-normalized residual connections and stable initialization
to maintain constant forward/backward variance across arbitrary depth.

Key components:
1. Fully-normalized residuals: λ·x + β·f(x) where λ² + β² = 1
2. Depth-aware initialization for embeddings, attention, and MLP
3. Variance-preserving scaling that handles dropout

Reference: https://arxiv.org/abs/2403.04235
"""

import math
import torch
import torch.nn as nn
from typing import Tuple


def compute_residual_gains(
    N: int, k: float = 2.0, alpha: float = 1.0
) -> Tuple[float, float]:
    """
    Compute λ and β for fully-normalized residual connections.

    Uses the DSLM formulation where:
    - β² = k / N^α
    - λ² = 1 - β²

    This ensures λ² + β² = 1 (fully normalized) and balances stability
    vs expressivity via the α parameter.

    Args:
        N: Total number of transformer blocks in the model
        k: Scaling factor (default 2.0, recommended in paper)
        alpha: Depth exponent (default 1.0 for best balance)
               α > 1 increases stability but may reduce expressivity
               α < 1 increases expressivity but may reduce stability

    Returns:
        (lambda, beta): Scaling factors for residual connection

    Example:
        >>> lam, beta = compute_residual_gains(N=24, k=2.0, alpha=1.0)
        >>> # Use as: x = lam * x + beta * f(x)
    """
    # β² = k / N^α
    beta2 = k / (N**alpha)

    # Clamp to valid range [0, 1]
    beta2 = min(max(beta2, 0.0), 1.0)

    # λ² = 1 - β² (fully normalized)
    lam2 = 1.0 - beta2

    # Return square roots
    lam = math.sqrt(lam2)
    beta = math.sqrt(beta2)

    return lam, beta


def apply_stable_residual(
    x: torch.Tensor, fx: torch.Tensor, lam: float, beta: float
) -> torch.Tensor:
    """
    Apply fully-normalized residual connection.

    Replaces standard residual (x + f(x)) with scaled version:
        x_out = λ·x + β·f(x)

    where λ² + β² = 1, ensuring variance is preserved across layers.

    Args:
        x: Input tensor (shortcut path)
        fx: Transformed tensor (residual path through attention/MLP)
        lam: Scaling factor for input (λ)
        beta: Scaling factor for residual (β)

    Returns:
        Scaled residual connection output
    """
    return lam * x + beta * fx


def stable_embedding_std(num_tables: int = 1, dropout_prob: float = 0.0) -> float:
    """
    Compute stable initialization std for embedding/patch projection.

    For vision transformers, num_tables is typically 1 (patch embedding only).
    For language models with token+position+segment embeddings, num_tables=3.

    Formula: σ_e² = (1 - p) / num_tables
    This ensures unit variance at the transformer input after dropout.

    Args:
        num_tables: Number of embedding tables that sum together
        dropout_prob: Dropout probability applied to embeddings

    Returns:
        Standard deviation for embedding weight initialization
    """
    variance = (1.0 - dropout_prob) / num_tables
    return math.sqrt(variance)


def stable_ffn_std(d: int, dropout_prob: float = 0.0) -> float:
    """
    Compute stable initialization std for FFN (MLP) weights.

    Formula from paper: σ_f² = (1/d) * √(1-p) / 2
    Applied to both linear layers in the FFN block.

    Args:
        d: Input dimension (hidden size)
        dropout_prob: Dropout probability in FFN

    Returns:
        Standard deviation for FFN weight initialization
    """
    # σ_f² = (1/d) * √(1-p) / 2
    variance = (math.sqrt(1.0 - dropout_prob) / 2.0) / d
    return math.sqrt(variance)


def stable_attention_qk_std(d: int) -> float:
    """
    Compute stable initialization std for Q/K attention projections.

    Formula: σ_qk² = 1/d (standard attention scaling)

    Args:
        d: Dimension (head_dim or model dim)

    Returns:
        Standard deviation for Q/K weight initialization
    """
    return 1.0 / math.sqrt(d)


def _fan_in(m: nn.Module) -> int:
    if isinstance(m, nn.Linear):
        return m.in_features
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        k = 1
        for s in m.kernel_size:
            k *= s
        return m.in_channels * k
    raise ValueError(
        f"Unsupported module type for fan_in calculation: {type(m).__name__}"
    )


def init_stable_embeddings(
    module: nn.Module, dropout_prob: float = 0.0, num_tables: int = 1
):
    """
    Apply stable initialization to patch embedding/projection layers.

    Searches for Conv3d or Linear layers used for patch embedding
    and initializes with variance-preserving std. Targets layers with
    'patch_embed' in their name path or ending with 'proj' to be more
    specific about embedding layers vs deeper layers.

    Args:
        module: Module containing embedding layers (e.g., PatchEmbed3D)
        dropout_prob: Dropout probability after embeddings
        num_tables: Number of embedding tables (usually 1 for vision)
    """
    sigma_e2 = (1.0 - dropout_prob) / num_tables
    for name, m in module.named_modules():
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            # More explicit targeting: only initialize embedding/projection layers
            if "patch_embed" in name.lower() or name.endswith("proj"):
                fan_in = _fan_in(m)
                std = math.sqrt(sigma_e2 / fan_in)
                nn.init.normal_(m.weight, 0.0, std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


def init_stable_attention(attn_module: nn.Module, dropout_prob: float = 0.0):
    """
    Apply stable initialization to attention module.

    Auto-detects dimensions from the QKV weight shape and applies:
    - Q/K projections: σ_qk = 1/√d (standard attention scaling)
    - V projection and output projection: σ_f (same as FFN, with dropout correction)

    Args:
        attn_module: Attention module (e.g., WindowAttention3D)
                     Must have 'qkv' and/or 'proj' attributes as nn.Linear
        dropout_prob: Dropout probability in attention
    """
    if hasattr(attn_module, "qkv") and isinstance(attn_module.qkv, nn.Linear):
        out3, d_in = attn_module.qkv.weight.shape  # [3*D, D]
        assert out3 % 3 == 0, f"QKV output dimension {out3} must be divisible by 3"

        D3 = out3 // 3
        # Q/K: standard attention scaling (1/√d)
        std_qk = 1.0 / math.sqrt(d_in)

        # V: FFN-style scaling with dropout correction
        # Factor of 2.0 accounts for residual path scaling
        sigma_f2 = (math.sqrt(1.0 - dropout_prob) / 2.0) / d_in
        std_v = math.sqrt(sigma_f2)

        with torch.no_grad():
            w = attn_module.qkv.weight
            nn.init.normal_(w[0:D3], 0.0, std_qk)  # Q
            nn.init.normal_(w[D3 : 2 * D3], 0.0, std_qk)  # K
            nn.init.normal_(w[2 * D3 : 3 * D3], 0.0, std_v)  # V
            if attn_module.qkv.bias is not None:
                nn.init.constant_(attn_module.qkv.bias, 0.0)

    # Output projection: uses actual input dimension from the layer
    if hasattr(attn_module, "proj") and isinstance(attn_module.proj, nn.Linear):
        # Factor of 2.0 accounts for residual path scaling
        sigma_f2 = (math.sqrt(1.0 - dropout_prob) / 2.0) / attn_module.proj.in_features
        std = math.sqrt(sigma_f2)
        nn.init.normal_(attn_module.proj.weight, 0.0, std)
        if attn_module.proj.bias is not None:
            nn.init.constant_(attn_module.proj.bias, 0.0)


def init_stable_mlp(mlp_module: nn.Module, d: int, dropout_prob: float = 0.0):
    """
    Apply stable initialization to MLP/FFN module.

    Both linear layers (fc1 and fc2) get the same σ_f initialization
    to ensure unit variance output under dropout.

    Args:
        mlp_module: MLP module
        d: Input dimension
        dropout_prob: Dropout probability in MLP
    """
    for m in mlp_module.modules():
        if isinstance(m, nn.Linear):
            sigma_f2 = (math.sqrt(1.0 - dropout_prob) / 2.0) / m.in_features
            std = math.sqrt(sigma_f2)
            nn.init.normal_(m.weight, 0.0, std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


def init_stable_model(
    model: nn.Module,
    total_blocks: int,
    base_dim: int,
    dropout_prob: float = 0.0,
    attention_dropout_prob: float = 0.0,
    attention_module_types: tuple = (
        "WindowAttention3D",
        "Attention",
        "WindowAttention",
    ),
    mlp_module_types: tuple = ("MLP", "FFN", "Mlp", "FeedForward"),
):
    """
    Apply stable initialization to entire model.

    Recursively searches for embedding, attention, and MLP modules
    and applies appropriate stable initialization. Module detection
    is configurable to support different transformer architectures.

    Args:
        model: Model to initialize
        total_blocks: Total number of transformer blocks (N)
        base_dim: Base model dimension (currently unused, kept for API compatibility)
        dropout_prob: Dropout probability for embeddings/MLP
        attention_dropout_prob: Dropout probability for attention
        attention_module_types: Tuple of attention module class names to initialize
        mlp_module_types: Tuple of MLP module class names to initialize
    """
    # Initialize embeddings
    for _, module in model.named_modules():
        module_type = type(module).__name__
        if "PatchEmbed" in module_type or "Embed" in module_type:
            init_stable_embeddings(module, dropout_prob, num_tables=1)

    # Initialize attention and MLP in transformer blocks
    for _, module in model.named_modules():
        module_type = type(module).__name__

        # Attention modules - support multiple architecture types
        if module_type in attention_module_types:
            init_stable_attention(module, attention_dropout_prob)

        # MLP/FFN modules - support multiple architecture types
        if module_type in mlp_module_types:
            # Get input dimension from first linear layer
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    d = m.in_features
                    if d > 0:
                        init_stable_mlp(module, d, dropout_prob)
                    break


def verify_stable_init(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cuda",
    num_samples: int = 100,
    rtol: float = 0.5,
    target_modules: tuple = ("SwinTransformerBlock", "Block", "TransformerBlock"),
    normalize_input: bool = True,
) -> dict:
    """
    Verify stable initialization by checking forward variance.

    Runs multiple forward passes with random inputs and measures
    the output variance at each layer. With stable init, variance
    should remain roughly constant (within ~2x) across all layers.

    Note: For hierarchical vision transformers with downsampling,
    some variance drift is expected and normal.

    Args:
        model: Model to verify
        input_shape: Input tensor shape (B, C, D, H, W)
        device: Device to run on
        num_samples: Number of forward passes for statistics
        rtol: Relative tolerance for variance (0.5 = within 50% of 1.0)
        target_modules: Tuple of module class names to hook for variance measurement
        normalize_input: Whether to normalize input to unit variance

    Returns:
        dict with 'mean_variance', 'std_variance', 'min_variance', 'max_variance', 'passed'
    """
    model.eval()
    model.to(device)

    # Hook to capture layer outputs
    variances = []

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            # Measure variance across batch and spatial dims
            var = output.var().item()
            variances.append(var)

    # Register hooks on target transformer blocks
    hooks = []
    for name, module in model.named_modules():
        if type(module).__name__ in target_modules:
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

    if len(hooks) == 0:
        # Fallback to name-based matching if no type matches found
        for name, module in model.named_modules():
            if "block" in name.lower() or "layer" in name.lower():
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)

    # Run forward passes
    with torch.no_grad():
        for _ in range(num_samples):
            x = torch.randn(input_shape, device=device)
            if normalize_input:
                x = (x - x.mean()) / (x.std() + 1e-6)

            _ = model(x)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Analyze variance
    if len(variances) == 0:
        return {
            "mean_variance": 0.0,
            "std_variance": 0.0,
            "min_variance": 0.0,
            "max_variance": 0.0,
            "passed": False,
            "error": "No variances captured - check target_modules parameter",
        }

    variances = torch.tensor(variances)
    results = {
        "mean_variance": variances.mean().item(),
        "std_variance": variances.std().item(),
        "min_variance": variances.min().item(),
        "max_variance": variances.max().item(),
        "passed": (variances.min() > (1.0 - rtol)) and (variances.max() < (1.0 + rtol)),
    }

    return results


def verify_stable_gradients(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cuda",
    rtol: float = 1.0,
) -> dict:
    """
    Verify stable gradients by checking gradient variance across layers.

    With stable init and λ/β scaling, gradient variance should remain
    within ~2-3x across shallow vs deep layers (much better than
    exponential growth/vanishing in vanilla transformers).

    Args:
        model: Model to verify
        input_shape: Input tensor shape
        device: Device to run on
        rtol: Relative tolerance for gradient variance ratio

    Returns:
        dict with gradient statistics
    """
    model.train()
    model.to(device)

    # Forward pass
    x = torch.randn(input_shape, device=device, requires_grad=True)
    x = (x - x.mean()) / (x.std() + 1e-6)

    output = model(x)

    # Dummy loss (mean of output)
    loss = output.mean()
    loss.backward()

    # Collect gradient norms from all layers
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None and "weight" in name:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)

    grad_norms = torch.tensor(grad_norms)

    results = {
        "mean_grad_norm": grad_norms.mean().item(),
        "std_grad_norm": grad_norms.std().item(),
        "min_grad_norm": grad_norms.min().item(),
        "max_grad_norm": grad_norms.max().item(),
        "ratio": (grad_norms.max() / (grad_norms.min() + 1e-10)).item(),
        "passed": (grad_norms.max() / (grad_norms.min() + 1e-10)) < (1.0 + rtol) * 3.0,
    }

    return results
