"""
Author: Omid Nejati
Email: omid_nejaty@alumni.iust.ac.ir

MedViT V1: A Robust Vision Transformer for Generalized Medical Image Classification.
"""

from functools import partial
from typing import Dict, List, Tuple, Union

import torch
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.layers import DropPath
from torch import nn

from .modules.layerscale import create_layerscale
from .modules.medvit_utils import (
    NORM_EPS,
    ConvBNReLU,
    E_MHSA,
    LocalityFeedForward,
    MHCA,
    PatchEmbed,
    _make_divisible,
    initialize_weights,
)
from regularization.shakedrop import ShakeDrop
from utils.stable_init import (
    apply_stable_residual,
    compute_residual_gains,
    init_stable_model,
)

# =============================================================================
# MedViT V1 Configurations
# =============================================================================

_MEDVITV1_CONFIGS: Dict[str, Dict[str, Union[List[int], float]]] = {
    "S": {
        "stem_chs": [64, 32, 64],
        "depths": [3, 4, 10, 3],
        "dims": [96, 192, 256, 384, 512, 768],
        "stochastic_depth_prob": 0.1,
    },
    "B": {
        "stem_chs": [64, 32, 64],
        "depths": [3, 4, 20, 3],
        "dims": [96, 192, 256, 384, 512, 768],
        "stochastic_depth_prob": 0.2,
    },
    "L": {
        "stem_chs": [64, 32, 64],
        "depths": [3, 4, 30, 3],
        "dims": [96, 192, 256, 384, 512, 768],
        "stochastic_depth_prob": 0.2,
    },
}


# =============================================================================
# MedViT V1 Specific Blocks
# =============================================================================


class ECB(nn.Module):
    """Efficient Convolution Block for local feature extraction.

    Uses Multi-Head Convolutional Attention (MHCA) for efficient local attention
    followed by locality-enhanced feed-forward network.

    This is the primary local processing block in MedViT V1, using grouped
    convolutions for attention instead of neighborhood attention.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for patch embedding downsampling.
        stochastic_depth_prob: Stochastic depth dropout probability.
        head_dim: Dimension of each attention head.
        mlp_ratio: Expansion ratio for feed-forward network.
        enable_stable: Whether to use stable residual connections.
        stable_lam: Lambda parameter for stable residual.
        stable_beta: Beta parameter for stable residual.
        use_shakedrop: Whether to use ShakeDrop regularization.
        shakedrop_alpha_range: Alpha range for ShakeDrop.
        layer_scale: Whether to use LayerScale.
        layer_scale_init_value: Initial value for LayerScale.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Union[int, Tuple[int, int, int]] = 1,
        stochastic_depth_prob: float = 0.0,
        head_dim: int = 32,
        mlp_ratio: float = 3.0,
        enable_stable: bool = False,
        stable_lam: float = 1.0,
        stable_beta: float = 0.0,
        use_shakedrop: bool = False,
        shakedrop_alpha_range: Tuple[float, float] = (-1.0, 1.0),
        layer_scale: bool = False,
        layer_scale_init_value: float = 1e-5,
    ) -> None:
        super(ECB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.enable_stable = enable_stable
        self.stable_lam = stable_lam
        self.stable_beta = stable_beta
        norm_layer = partial(nn.BatchNorm3d, eps=NORM_EPS)
        assert out_channels % head_dim == 0, (
            f"out_channels ({out_channels}) must be divisible by head_dim ({head_dim})"
        )

        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        self.norm1 = norm_layer(out_channels)
        self.mhca = MHCA(out_channels, head_dim)

        # Stochastic depth / ShakeDrop for attention branch
        if use_shakedrop and stochastic_depth_prob > 0.0:
            self.attention_stochastic_depth_prob = ShakeDrop(
                p_drop=stochastic_depth_prob, alpha_range=shakedrop_alpha_range
            )
        elif stochastic_depth_prob > 0.0:
            self.attention_stochastic_depth_prob = DropPath(stochastic_depth_prob)
        else:
            self.attention_stochastic_depth_prob = nn.Identity()

        self.conv = LocalityFeedForward(
            out_channels,
            out_channels,
            stride=1,
            expand_ratio=mlp_ratio,
            reduction=out_channels,
        )

        # Stochastic depth / ShakeDrop for FFN branch
        if use_shakedrop and stochastic_depth_prob > 0.0:
            self.ffn_stochastic_depth_prob = ShakeDrop(
                p_drop=stochastic_depth_prob, alpha_range=shakedrop_alpha_range
            )
        elif stochastic_depth_prob > 0.0:
            self.ffn_stochastic_depth_prob = DropPath(stochastic_depth_prob)
        else:
            self.ffn_stochastic_depth_prob = nn.Identity()

        self.norm2 = norm_layer(out_channels)

        # LayerScale for attention and FFN branches
        self.ls1 = create_layerscale(out_channels, layer_scale, layer_scale_init_value)
        self.ls2 = create_layerscale(out_channels, layer_scale, layer_scale_init_value)

        self.is_bn_merged = False

    def merge_bn(self) -> None:
        """Merge batch normalization layers for inference optimization."""
        if not self.is_bn_merged:
            self.is_bn_merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply efficient convolution block.

        Args:
            x: Input tensor of shape (B, C_in, D, H, W).

        Returns:
            Output tensor of shape (B, C_out, D', H', W').
        """
        x = self.patch_embed(x)
        shortcut = x

        # MHCA path
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm1(x)
        else:
            out = x

        attn_out = self.mhca(out)
        attn_out = (
            self.ls1(attn_out.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            if self.ls1 is not None
            else attn_out
        )
        attn_out = self.attention_stochastic_depth_prob(attn_out)

        # Apply stable residual for attention path
        if self.enable_stable:
            x = apply_stable_residual(
                shortcut, attn_out, self.stable_lam, self.stable_beta
            )
        else:
            x = shortcut + attn_out

        # Feed-forward path
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm2(x)
        else:
            out = x

        ffn_out = self.conv(out)
        ffn_out = (
            self.ls2(ffn_out.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            if self.ls2 is not None
            else ffn_out
        )
        ffn_out = self.ffn_stochastic_depth_prob(ffn_out)

        # Apply stable residual for FFN path
        if self.enable_stable:
            x = apply_stable_residual(x, ffn_out, self.stable_lam, self.stable_beta)
        else:
            x = x + ffn_out

        return x


class LTB(nn.Module):
    """Local Transformer Block for global-local feature processing.

    Combines efficient multi-head self-attention (E-MHSA) for global context
    and multi-head convolutional attention (MHCA) for local features.

    This is the primary global processing block in MedViT V1, providing
    long-range dependencies through spatial reduction attention.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stochastic_depth_prob: Stochastic depth dropout probability.
        stride: Stride for patch embedding downsampling.
        sr_ratio: Spatial reduction ratio for attention.
        mlp_ratio: Expansion ratio for feed-forward network.
        head_dim: Dimension of each attention head.
        mix_block_ratio: Ratio of channels allocated to MHSA vs MHCA.
        attention_dropout: Dropout probability for attention weights.
        dropout: General dropout probability.
        enable_stable: Whether to use stable residual connections.
        stable_lam: Lambda parameter for stable residual.
        stable_beta: Beta parameter for stable residual.
        use_shakedrop: Whether to use ShakeDrop regularization.
        shakedrop_alpha_range: Alpha range for ShakeDrop.
        layer_scale: Whether to use LayerScale.
        layer_scale_init_value: Initial value for LayerScale.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stochastic_depth_prob: float,
        stride: Union[int, Tuple[int, int, int]] = 1,
        sr_ratio: int = 1,
        mlp_ratio: float = 2.0,
        head_dim: int = 32,
        mix_block_ratio: float = 0.75,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        enable_stable: bool = False,
        stable_lam: float = 1.0,
        stable_beta: float = 0.0,
        use_shakedrop: bool = False,
        shakedrop_alpha_range: Tuple[float, float] = (-1.0, 1.0),
        layer_scale: bool = False,
        layer_scale_init_value: float = 1e-5,
    ) -> None:
        super(LTB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mix_block_ratio = mix_block_ratio
        self.enable_stable = enable_stable
        self.stable_lam = stable_lam
        self.stable_beta = stable_beta
        norm_func = partial(nn.BatchNorm3d, eps=NORM_EPS)

        self.mhsa_out_channels = _make_divisible(
            int(out_channels * mix_block_ratio), 32
        )
        self.mhca_out_channels = out_channels - self.mhsa_out_channels

        # E-MHSA branch
        self.patch_embed = PatchEmbed(in_channels, self.mhsa_out_channels, stride)
        self.norm1 = norm_func(self.mhsa_out_channels)
        self.e_mhsa = E_MHSA(
            self.mhsa_out_channels,
            head_dim=head_dim,
            sr_ratio=sr_ratio,
            attention_dropout=attention_dropout,
            proj_drop=dropout,
        )

        if use_shakedrop and stochastic_depth_prob * mix_block_ratio > 0.0:
            self.mhsa_stochastic_depth_prob = ShakeDrop(
                p_drop=stochastic_depth_prob * mix_block_ratio,
                alpha_range=shakedrop_alpha_range,
            )
        elif stochastic_depth_prob * mix_block_ratio > 0.0:
            self.mhsa_stochastic_depth_prob = DropPath(
                stochastic_depth_prob * mix_block_ratio
            )
        else:
            self.mhsa_stochastic_depth_prob = nn.Identity()

        # MHCA branch
        self.projection = PatchEmbed(
            self.mhsa_out_channels, self.mhca_out_channels, stride=1
        )
        self.mhca = MHCA(self.mhca_out_channels, head_dim=head_dim)

        if use_shakedrop and stochastic_depth_prob * (1 - mix_block_ratio) > 0.0:
            self.mhca_stochastic_depth_prob = ShakeDrop(
                p_drop=stochastic_depth_prob * (1 - mix_block_ratio),
                alpha_range=shakedrop_alpha_range,
            )
        elif stochastic_depth_prob * (1 - mix_block_ratio) > 0.0:
            self.mhca_stochastic_depth_prob = DropPath(
                stochastic_depth_prob * (1 - mix_block_ratio)
            )
        else:
            self.mhca_stochastic_depth_prob = nn.Identity()

        # FFN branch
        self.norm2 = norm_func(out_channels)
        self.conv = LocalityFeedForward(
            out_channels,
            out_channels,
            stride=1,
            expand_ratio=mlp_ratio,
            reduction=out_channels,
        )

        if use_shakedrop and stochastic_depth_prob > 0.0:
            self.mlp_stochastic_depth_prob = ShakeDrop(
                p_drop=stochastic_depth_prob, alpha_range=shakedrop_alpha_range
            )
        elif stochastic_depth_prob > 0.0:
            self.mlp_stochastic_depth_prob = DropPath(stochastic_depth_prob)
        else:
            self.mlp_stochastic_depth_prob = nn.Identity()

        # LayerScale for MHSA, MHCA, and FFN branches
        self.ls1 = create_layerscale(
            self.mhsa_out_channels, layer_scale, layer_scale_init_value
        )
        self.ls2 = create_layerscale(
            self.mhca_out_channels, layer_scale, layer_scale_init_value
        )
        self.ls3 = create_layerscale(out_channels, layer_scale, layer_scale_init_value)

        self.is_bn_merged = False

    def merge_bn(self) -> None:
        """Merge batch normalization layers for inference optimization."""
        if not self.is_bn_merged:
            self.e_mhsa.merge_bn(self.norm1)
            self.is_bn_merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply local transformer block.

        Args:
            x: Input tensor of shape (B, C_in, D, H, W).

        Returns:
            Output tensor of shape (B, C_out, D', H', W').
        """
        x = self.patch_embed(x)
        B, C, D, H, W = x.shape

        # E-MHSA path
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm1(x)
        else:
            out = x
        out = rearrange(out, "b c d h w -> b (d h w) c", d=D, h=H, w=W)
        mhsa_out = self.e_mhsa(out)
        mhsa_out = self.ls1(mhsa_out) if self.ls1 is not None else mhsa_out
        mhsa_out = self.mhsa_stochastic_depth_prob(mhsa_out)

        # Apply stable residual for MHSA path
        if self.enable_stable:
            x = apply_stable_residual(
                x,
                rearrange(mhsa_out, "b (d h w) c -> b c d h w", d=D, h=H, w=W),
                self.stable_lam,
                self.stable_beta,
            )
        else:
            x = x + rearrange(mhsa_out, "b (d h w) c -> b c d h w", d=D, h=H, w=W)

        # MHCA path
        out = self.projection(x)
        mhca_out = self.mhca(out)
        mhca_out = (
            self.ls2(mhca_out.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            if self.ls2 is not None
            else mhca_out
        )
        mhca_out = self.mhca_stochastic_depth_prob(mhca_out)

        # Apply stable residual for MHCA path
        if self.enable_stable:
            out = apply_stable_residual(
                out, mhca_out, self.stable_lam, self.stable_beta
            )
        else:
            out = out + mhca_out

        x = torch.cat([x, out], dim=1)

        # FFN path
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            ffn_input = self.norm2(x)
        else:
            ffn_input = x

        mlp_out = self.conv(ffn_input)
        mlp_out = (
            self.ls3(mlp_out.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            if self.ls3 is not None
            else mlp_out
        )
        mlp_out = self.mlp_stochastic_depth_prob(mlp_out)

        # Apply stable residual for FFN path
        if self.enable_stable:
            x = apply_stable_residual(x, mlp_out, self.stable_lam, self.stable_beta)
        else:
            x = x + mlp_out

        return x


# =============================================================================
# MedViT V1 Model
# =============================================================================


class MedViTV1(nn.Module):
    """MedViT V1: Original Vision Transformer for 3D Medical Image Classification.

    Hierarchical vision transformer using ECB (convolutional attention) for local
    processing and LTB (E-MHSA) for global processing. Uses standard MLP instead
    of KAN networks.

    Architecture pattern per stage:
    - Stage 1: [ECB] * depths[0]
    - Stage 2: [ECB] * (depths[1] - 1) + [LTB]
    - Stage 3: [ECB, ECB, ECB, ECB, LTB] * (depths[2] // 5)
    - Stage 4: [ECB] * (depths[3] - 1) + [LTB]

    Args:
        in_channels: Number of input channels (e.g., 1 for grayscale MRI).
        stem_chs: Channel dimensions for stem convolution layers [ch1, ch2, ch3].
        depths: Number of blocks in each stage [depth1, depth2, depth3, depth4].
        stochastic_depth_prob: Maximum stochastic depth dropout (linearly increases across depth).
        attention_dropout: Dropout probability for attention weights.
        dropout: General dropout probability.
        num_classes: Number of output classes for classification.
        strides: Spatial downsampling stride for each stage.
        sr_ratios: Spatial reduction ratios for attention in each stage.
        head_dim: Dimension of each attention head.
        mlp_ratio: Expansion ratio for feed-forward network (default 3.0).
        mix_block_ratio: Channel ratio for MHSA vs MHCA in LTB blocks.
        use_checkpoint: Whether to use gradient checkpointing to save memory.
        enable_stable: Whether to use stable residual connections.
        stable_k: K parameter for stable residual gains.
        stable_alpha: Alpha parameter for stable residual gains.
        use_shakedrop: Whether to use ShakeDrop regularization.
        shakedrop_alpha_range: Alpha range for ShakeDrop.
        layer_scale: Whether to use LayerScale.
        layer_scale_init_value: Initial value for LayerScale.

    Attributes:
        stem: Initial convolutional stem for feature extraction.
        features: Sequential container of ECB and LTB blocks.
        norm: Final batch normalization layer.
        avgpool: Global average pooling layer.
        proj_head: Classification head (linear layer).
    """

    def __init__(
        self,
        in_channels: int,
        stem_chs: List[int],
        depths: List[int],
        dims: List[int],
        stochastic_depth_prob: float,
        attention_dropout: float,
        dropout: float,
        num_classes: int,
        strides: List[int],
        sr_ratios: List[int],
        head_dim: int,
        mlp_ratio: float,
        mix_block_ratio: float,
        use_checkpoint: bool,
        enable_stable: bool = False,
        stable_k: float = 2.0,
        stable_alpha: float = 1.0,
        use_shakedrop: bool = False,
        shakedrop_alpha_range: Tuple[float, float] = (-1.0, 1.0),
        layer_scale: bool = False,
        layer_scale_init_value: float = 1e-5,
    ) -> None:
        super(MedViTV1, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.enable_stable = enable_stable

        # Compute total number of transformer blocks
        self.total_blocks = sum(depths)

        # Compute stable residual gains if enabled
        if enable_stable:
            stable_lam, stable_beta = compute_residual_gains(
                N=self.total_blocks, k=stable_k, alpha=stable_alpha
            )
            self.stable_lam = stable_lam
            self.stable_beta = stable_beta
        else:
            self.stable_lam = 1.0
            self.stable_beta = 0.0

        self.stage_out_channels = [
            [dims[0]] * (depths[0]),
            [dims[1]] * (depths[1] - 1) + [dims[2]],
            [dims[3], dims[3], dims[3], dims[3], dims[4]] * (depths[2] // 5),
            [dims[5]] * (depths[3]),
        ]

        # V1 Hybrid Strategy: ECB for local, LTB for global
        self.stage_block_types = [
            [ECB] * depths[0],
            [ECB] * (depths[1] - 1) + [LTB],
            [ECB, ECB, ECB, ECB, LTB] * (depths[2] // 5),
            [ECB] * (depths[3] - 1) + [LTB],
        ]

        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, stem_chs[0], kernel_size=3, stride=2),
            ConvBNReLU(stem_chs[0], stem_chs[1], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[1], stem_chs[2], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[2], stem_chs[2], kernel_size=3, stride=2),
        )
        input_channel = stem_chs[-1]
        features = []
        idx = 0
        dpr = [
            x.item() for x in torch.linspace(0, stochastic_depth_prob, sum(depths))
        ]  # stochastic depth decay rule

        for stage_id in range(len(depths)):
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            block_types = self.stage_block_types[stage_id]

            for block_id in range(numrepeat):
                if strides[stage_id] == 2 and block_id == 0:
                    stride = 2
                else:
                    stride = 1
                output_channel = output_channels[block_id]
                block_type = block_types[block_id]

                if block_type is ECB:
                    layer = ECB(
                        input_channel,
                        output_channel,
                        stride=stride,
                        stochastic_depth_prob=dpr[idx + block_id],
                        head_dim=head_dim,
                        mlp_ratio=mlp_ratio,
                        enable_stable=enable_stable,
                        stable_lam=self.stable_lam,
                        stable_beta=self.stable_beta,
                        use_shakedrop=use_shakedrop,
                        shakedrop_alpha_range=shakedrop_alpha_range,
                        layer_scale=layer_scale,
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    features.append(layer)
                elif block_type is LTB:
                    layer = LTB(
                        input_channel,
                        output_channel,
                        stochastic_depth_prob=dpr[idx + block_id],
                        stride=stride,
                        sr_ratio=sr_ratios[stage_id],
                        mlp_ratio=mlp_ratio,
                        head_dim=head_dim,
                        mix_block_ratio=mix_block_ratio,
                        attention_dropout=attention_dropout,
                        dropout=dropout,
                        enable_stable=enable_stable,
                        stable_lam=self.stable_lam,
                        stable_beta=self.stable_beta,
                        use_shakedrop=use_shakedrop,
                        shakedrop_alpha_range=shakedrop_alpha_range,
                        layer_scale=layer_scale,
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    features.append(layer)
                input_channel = output_channel
            idx += numrepeat

        self.features = nn.Sequential(*features)

        self.norm = nn.BatchNorm3d(output_channel, eps=NORM_EPS)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj_head = nn.Sequential(
            nn.Linear(output_channel, num_classes),
        )

        self.stage_out_idx = [sum(depths[: idx + 1]) - 1 for idx in range(len(depths))]

        # Apply appropriate initialization
        if enable_stable:
            init_stable_model(
                model=self,
                total_blocks=self.total_blocks,
                base_dim=stem_chs[-1],
                dropout_prob=dropout,
                attention_dropout_prob=attention_dropout,
                attention_module_types=("E_MHSA", "MHCA"),
                mlp_module_types=("Mlp", "LocalityFeedForward"),
            )
        else:
            self._initialize_weights()

    def merge_bn(self) -> None:
        """Merge all batch normalization layers into convolutions for faster inference."""
        self.eval()
        for _, module in self.named_modules():
            if isinstance(module, (ECB, LTB)) and hasattr(module, "merge_bn"):
                module.merge_bn()

    def _initialize_weights(self) -> None:
        """Initialize model weights using truncated normal and constant initialization."""
        initialize_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MedViT V1.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            Class logits of shape (B, num_classes).
        """
        x = self.stem(x)
        for layer in self.features:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.proj_head(x)
        return x


# =============================================================================
# MedViT V1 Factory Classes
# =============================================================================


class MedViTV1X(MedViTV1):
    """Base class for MedViT V1 variants using predefined configurations."""

    def __init__(self, config_name: str, **kwargs):
        if config_name not in _MEDVITV1_CONFIGS:
            raise ValueError(
                f"Unknown config_name '{config_name}'. Available: {list(_MEDVITV1_CONFIGS.keys())}"
            )
        config = dict(_MEDVITV1_CONFIGS[config_name])
        config.setdefault("stem_chs", [64, 32, 64])
        config.setdefault("strides", [1, 2, 2, 2])
        config.setdefault("sr_ratios", [8, 4, 2, 1])
        config.setdefault("head_dim", 32)
        config.setdefault("mix_block_ratio", 0.75)
        config.update(kwargs)
        super().__init__(**config)


class MedViTV1S(MedViTV1X):
    """MedViT V1 Small variant.

    Depths: [3, 4, 10, 3], dims: [96, 192, 256, 384, 512, 768], stochastic_depth_prob: 0.1
    """

    def __init__(self, **kwargs):
        super().__init__(config_name="S", **kwargs)


class MedViTV1B(MedViTV1X):
    """MedViT V1 Base variant.

    Depths: [3, 4, 20, 3], dims: [96, 192, 256, 384, 512, 768], stochastic_depth_prob: 0.2
    """

    def __init__(self, **kwargs):
        super().__init__(config_name="B", **kwargs)


class MedViTV1L(MedViTV1X):
    """MedViT V1 Large variant.

    Depths: [3, 4, 30, 3], dims: [96, 192, 256, 384, 512, 768], stochastic_depth_prob: 0.2
    """

    def __init__(self, **kwargs):
        super().__init__(config_name="L", **kwargs)
