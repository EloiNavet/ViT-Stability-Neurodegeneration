from functools import partial
import math
from typing import Optional, Tuple, Union
import torch
from torch import nn
from timm.layers import to_3tuple

NORM_EPS = 1e-5


def merge_pre_bn(
    module: Union[nn.Linear, nn.Conv3d],
    pre_bn_1: nn.BatchNorm3d,
    pre_bn_2: Optional[nn.BatchNorm3d] = None,
) -> None:
    """Merge batch normalization layers into preceding linear or conv layer for inference optimization.

    This function folds batch normalization parameters into the weights and biases of the
    preceding linear or convolutional layer, reducing computational overhead during inference.

    Args:
        module: Linear or Conv3d layer to merge batch norm into.
        pre_bn_1: First batch normalization layer to merge.
        pre_bn_2: Optional second batch normalization layer to merge (for sequential BN layers).

    Raises:
        AssertionError: If batch norm layers don't have track_running_stats or affine enabled.
        AssertionError: If Conv3d kernel is not 1x1x1.
    """
    weight = module.weight.data
    if module.bias is None:
        zeros = torch.zeros(module.out_channels, device=weight.device).type(
            weight.type()
        )
        module.bias = nn.Parameter(zeros)
    bias = module.bias.data
    if pre_bn_2 is None:
        assert pre_bn_1.track_running_stats is True, (
            "Unsupport bn_module.track_running_stats is False"
        )
        assert pre_bn_1.affine is True, "Unsupport bn_module.affine is False"

        scale_invstd = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        extra_weight = scale_invstd * pre_bn_1.weight
        extra_bias = (
            pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd
        )
    else:
        assert pre_bn_1.track_running_stats is True, (
            "Unsupport bn_module.track_running_stats is False"
        )
        assert pre_bn_1.affine is True, "Unsupport bn_module.affine is False"

        assert pre_bn_2.track_running_stats is True, (
            "Unsupport bn_module.track_running_stats is False"
        )
        assert pre_bn_2.affine is True, "Unsupport bn_module.affine is False"

        scale_invstd_1 = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        scale_invstd_2 = pre_bn_2.running_var.add(pre_bn_2.eps).pow(-0.5)

        extra_weight = (
            scale_invstd_1 * pre_bn_1.weight * scale_invstd_2 * pre_bn_2.weight
        )
        extra_bias = (
            scale_invstd_2
            * pre_bn_2.weight
            * (
                pre_bn_1.bias
                - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd_1
                - pre_bn_2.running_mean
            )
            + pre_bn_2.bias
        )

    if isinstance(module, nn.Linear):
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
    elif isinstance(module, nn.Conv3d):
        assert weight.shape[2] == 1 and weight.shape[3] == 1 and weight.shape[4] == 1
        weight = weight.reshape(weight.shape[0], weight.shape[1])
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
        weight = weight.reshape(weight.shape[0], weight.shape[1], 1, 1, 1)
    bias.add_(extra_bias)

    module.weight.data = weight
    module.bias.data = bias


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """Make a value divisible by a divisor, ensuring it doesn't decrease by more than 10%.

    Used to ensure channel dimensions are compatible with hardware optimization constraints.

    Args:
        v: Value to make divisible.
        divisor: The divisor to ensure divisibility.
        min_value: Optional minimum value constraint.

    Returns:
        Value rounded to nearest multiple of divisor, ensuring >= 90% of original value.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# =============================================================================
# Activation Functions
# =============================================================================


class h_sigmoid(nn.Module):
    """Hard sigmoid activation function.

    Approximation of sigmoid using ReLU6: h_sigmoid(x) = ReLU6(x + 3) / 6.
    More efficient than standard sigmoid during inference.

    Args:
        inplace: Whether to perform operation in-place.
    """

    def __init__(self, inplace: bool = True) -> None:
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply hard sigmoid activation.

        Args:
            x: Input tensor.

        Returns:
            Activated tensor with values in [0, 1].
        """
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    """Hard swish activation function.

    Approximation of swish using hard sigmoid: h_swish(x) = x * h_sigmoid(x).
    More efficient than standard swish during inference.

    Args:
        inplace: Whether to perform operation in-place.
    """

    def __init__(self, inplace: bool = True) -> None:
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply hard swish activation.

        Args:
            x: Input tensor.

        Returns:
            Activated tensor.
        """
        return x * self.sigmoid(x)


# =============================================================================
# Channel Attention Layers
# =============================================================================


class ECALayer(nn.Module):
    """Efficient Channel Attention layer.

    Implements efficient channel attention using 1D convolution on global pooled features.
    Adaptive kernel size based on channel dimension.

    Args:
        channel: Number of input channels.
        gamma: Parameter for adaptive kernel size calculation.
        b: Bias parameter for adaptive kernel size calculation.
        sigmoid_type: Type of sigmoid activation ('sigmoid' or 'h_sigmoid').

    Raises:
        NotImplementedError: If sigmoid_type is not 'sigmoid' or 'h_sigmoid'.
    """

    def __init__(
        self, channel: int, gamma: int = 2, b: int = 1, sigmoid_type: str = "sigmoid"
    ) -> None:
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        if sigmoid_type == "sigmoid":
            self.sigmoid = nn.Sigmoid()
        elif sigmoid_type == "h_sigmoid":
            self.sigmoid = h_sigmoid()
        else:
            raise NotImplementedError(
                f"Sigmoid type {sigmoid_type} not implemented for ECALayer"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply efficient channel attention.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            Channel-attention weighted tensor of shape (B, C, D, H, W).
        """
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SELayer(nn.Module):
    """Squeeze-and-Excitation layer.

    Channel attention mechanism using global pooling and two fully connected layers.

    Args:
        channel: Number of input channels.
        reduction: Reduction ratio for the bottleneck dimension.
    """

    def __init__(self, channel: int, reduction: int = 4) -> None:
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=False),
            nn.Linear(channel // reduction, channel),
            h_sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply squeeze-and-excitation attention.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            Channel-attention weighted tensor of shape (B, C, D, H, W).
        """
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


# =============================================================================
# Convolution-Based Blocks
# =============================================================================


class ConvBNReLU(nn.Module):
    """3D Convolution followed by Batch Normalization and ReLU activation.

    Basic building block combining Conv3d, BatchNorm3d, and ReLU in sequence.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel (converted to 3-tuple).
        stride: Stride of the convolution (converted to 3-tuple).
        groups: Number of blocked connections from input to output channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]],
        groups: int = 1,
    ) -> None:
        super(ConvBNReLU, self).__init__()
        kernel_size = to_3tuple(kernel_size)
        stride = to_3tuple(stride)

        padding = tuple(k // 2 for k in kernel_size)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.norm = nn.BatchNorm3d(out_channels, eps=NORM_EPS)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, batch normalization, and ReLU activation.

        Args:
            x: Input tensor of shape (B, C_in, D, H, W).

        Returns:
            Output tensor of shape (B, C_out, D', H', W').
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class PatchEmbed(nn.Module):
    """Patch embedding layer with optional downsampling.

    Performs spatial downsampling via average pooling followed by 1x1x1 convolution
    for channel adjustment, or identity mapping if no changes needed.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Downsampling stride (converted to 3-tuple).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Union[int, Tuple[int, int, int]] = 1,
    ) -> None:
        super(PatchEmbed, self).__init__()
        norm_layer = partial(nn.BatchNorm3d, eps=NORM_EPS)

        s = to_3tuple(stride)

        if s[0] > 1 or s[1] > 1 or s[2] > 1:
            pool_stride = tuple(st if st > 0 else 1 for st in s)
            self.avgpool = nn.AvgPool3d(
                kernel_size=pool_stride,
                stride=pool_stride,
                ceil_mode=True,
                count_include_pad=False,
            )
            self.conv = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=False
            )
            self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=False
            )
            self.norm = norm_layer(out_channels)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply patch embedding transformation.

        Args:
            x: Input tensor of shape (B, C_in, D, H, W).

        Returns:
            Output tensor of shape (B, C_out, D', H', W').
        """
        return self.norm(self.conv(self.avgpool(x)))


class MHCA(nn.Module):
    """Multi-Head Convolutional Attention.

    Implements attention mechanism using grouped 3D convolutions instead of
    traditional dot-product attention for improved efficiency.

    Args:
        out_channels: Number of output channels (must be divisible by head_dim).
        head_dim: Dimension of each attention head.
    """

    def __init__(self, out_channels: int, head_dim: int) -> None:
        super(MHCA, self).__init__()
        norm_layer = partial(nn.BatchNorm3d, eps=NORM_EPS)
        self.group_conv3x3 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=1,
            groups=out_channels // head_dim,
            bias=False,
        )
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=False)
        self.projection = nn.Conv3d(
            out_channels, out_channels, kernel_size=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head convolutional attention.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            Output tensor of shape (B, C, D, H, W).
        """
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        return out


# =============================================================================
# Feed-Forward Networks
# =============================================================================


class LocalityFeedForward(nn.Module):
    """Locality-enhanced Feed-Forward Network with depth-wise convolutions.

    Inverted residual block with optional depth-wise convolution and channel attention.

    Args:
        in_dim: Input dimension.
        out_dim: Output dimension (should typically equal in_dim for residual connection).
        kernel_size: Kernel size for depth-wise convolution.
        stride: Stride for depth-wise convolution.
        expand_ratio: Expansion ratio for hidden dimension.
        act: Activation function configuration.
            - 'relu': ReLU activation
            - 'hs': Hard swish
            - 'hs+se': Hard swish with SE attention
            - 'hs+eca': Hard swish with ECA attention
            - 'hs+ecah': Hard swish with ECA attention (using hard sigmoid)
        reduction: Reduction ratio for attention modules.
        wo_dp_conv: If True, omit depth-wise convolution.
        dp_first: If True, place depth-wise convolution before expansion.
    """

    def __init__(
        self,
        in_dim: int = 64,
        out_dim: int = 96,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        expand_ratio: float = 4.0,
        act: str = "hs+se",
        reduction: int = 4,
        wo_dp_conv: bool = False,
        dp_first: bool = False,
    ) -> None:
        super(LocalityFeedForward, self).__init__()
        hidden_dim = int(in_dim * expand_ratio)

        kernel_size = to_3tuple(kernel_size)
        stride = to_3tuple(stride)

        layers = []
        # the first linear layer is replaced by 1x1x1 convolution.
        layers.extend(
            [
                nn.Conv3d(
                    in_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm3d(hidden_dim),
                h_swish() if act.find("hs") >= 0 else nn.ReLU6(inplace=False),
            ]
        )

        # the depth-wise convolution between the two linear layers
        if not wo_dp_conv:
            dp = [
                nn.Conv3d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=tuple(k // 2 for k in kernel_size),
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm3d(hidden_dim),
                h_swish() if act.find("hs") >= 0 else nn.ReLU6(inplace=False),
            ]
            if dp_first:
                layers = dp + layers
            else:
                layers.extend(dp)

        if act.find("+") >= 0:
            attn_type = act.split("+")[1]
            if attn_type == "se":
                layers.append(SELayer(hidden_dim, reduction=reduction))
            elif attn_type.find("eca") >= 0:
                sigmoid_activation = "sigmoid" if attn_type == "eca" else "h_sigmoid"
                layers.append(ECALayer(hidden_dim, sigmoid_type=sigmoid_activation))
            else:
                raise NotImplementedError(
                    "Activation type {} is not implemented".format(act)
                )

        # the second linear layer is replaced by 1x1x1 convolution.
        layers.extend(
            [
                nn.Conv3d(
                    hidden_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm3d(out_dim),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply locality-enhanced feed-forward with residual connection.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            Output tensor of shape (B, C, D, H, W).
        """
        x = x + self.conv(x)
        return x


class Mlp(nn.Module):
    """Multi-Layer Perceptron implemented with 1x1x1 convolutions.

    Two-layer MLP with ReLU activation and dropout, using convolutions for efficiency.

    Args:
        in_features: Number of input features.
        out_features: Number of output features (defaults to in_features).
        mlp_ratio: Expansion ratio for hidden dimension.
        dropout: Dropout probability.
        bias: Whether to use bias in convolutions.
    """

    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        mlp_ratio: Optional[float] = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_dim = _make_divisible(in_features * mlp_ratio, 32)
        self.conv1 = nn.Conv3d(in_features, hidden_dim, kernel_size=1, bias=bias)
        self.act = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv3d(hidden_dim, out_features, kernel_size=1, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def merge_bn(self, pre_norm: nn.BatchNorm3d) -> None:
        """Merge batch normalization into first convolution layer.

        Args:
            pre_norm: Batch normalization layer to merge.
        """
        merge_pre_bn(self.conv1, pre_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two-layer MLP.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            Output tensor of shape (B, C', D, H, W).
        """
        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        return x


# =============================================================================
# Efficient Attention
# =============================================================================


class E_MHSA(nn.Module):
    """Efficient Multi-Head Self-Attention with spatial reduction.

    Memory-efficient attention mechanism that optionally reduces spatial dimensions
    of keys and values using average pooling for computational efficiency.

    Args:
        dim: Input dimension.
        out_dim: Output dimension (defaults to dim).
        head_dim: Dimension of each attention head.
        qkv_bias: Whether to use bias in Q, K, V projections.
        qk_scale: Optional scaling factor for attention scores.
        attention_dropout: Dropout probability for attention weights.
        proj_drop: Dropout probability for output projection.
        sr_ratio: Spatial reduction ratio for keys and values.
    """

    def __init__(
        self,
        dim: int,
        out_dim: Optional[int] = None,
        head_dim: int = 32,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attention_dropout: float = 0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim**-0.5
        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.out_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.N_ratio = sr_ratio**2
        if sr_ratio > 1:
            self.sr = nn.AvgPool1d(kernel_size=self.N_ratio, stride=self.N_ratio)
            self.norm = nn.BatchNorm1d(dim, eps=NORM_EPS)
        self.is_bn_merged = False

    def merge_bn(self, pre_bn: nn.BatchNorm3d) -> None:
        """Merge batch normalization into Q, K, V projections.

        Args:
            pre_bn: Batch normalization layer to merge.
        """
        merge_pre_bn(self.q, pre_bn)
        if self.sr_ratio > 1:
            merge_pre_bn(self.k, pre_bn, self.norm)
            merge_pre_bn(self.v, pre_bn, self.norm)
        else:
            merge_pre_bn(self.k, pre_bn)
            merge_pre_bn(self.v, pre_bn)
        self.is_bn_merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply efficient multi-head self-attention.

        Args:
            x: Input tensor of shape (B, N, C).

        Returns:
            Output tensor of shape (B, N, C).
        """
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2)
            x_ = self.sr(x_)
            if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
                x_ = self.norm(x_)
            x_ = x_.transpose(1, 2)
            k = (
                self.k(x_)
                .reshape(B, -1, self.num_heads, C // self.num_heads)
                .permute(0, 2, 3, 1)
            )
            v = (
                self.v(x_)
                .reshape(B, -1, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )
        else:
            k = (
                self.k(x)
                .reshape(B, -1, self.num_heads, C // self.num_heads)
                .permute(0, 2, 3, 1)
            )
            v = (
                self.v(x)
                .reshape(B, -1, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )

        attn = (q @ k) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# =============================================================================
# Weight Initialization
# =============================================================================


def initialize_weights(model: nn.Module) -> None:
    """Initialize model weights using truncated normal and constant initialization.

    Args:
        model: PyTorch model to initialize.
    """
    from timm.layers import trunc_normal_

    for _, m in model.named_modules():
        if isinstance(
            m,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.GroupNorm,
                nn.LayerNorm,
            ),
        ):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
