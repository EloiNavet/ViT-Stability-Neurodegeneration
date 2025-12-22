"""3D Swin Transformer V1 for volumetric medical imaging."""

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath, to_3tuple, trunc_normal_
from torch.utils.checkpoint import checkpoint

from models.modules.layerscale import create_layerscale
from regularization.shakedrop import ShakeDrop
from utils.stable_init import (
    apply_stable_residual,
    compute_residual_gains,
    init_stable_model,
)

_SWIN_CONFIGS: Dict[str, Dict[str, Union[int, List[int]]]] = {
    "T": {
        "patch_size": [4, 4, 4],
        "embed_dim": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "window_size": [7, 7, 7],
    },
    "S": {
        "patch_size": [4, 4, 4],
        "embed_dim": 96,
        "depths": [2, 2, 18, 2],
        "num_heads": [3, 6, 12, 24],
        "window_size": [7, 7, 7],
    },
    "B": {
        "patch_size": [4, 4, 4],
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        "num_heads": [4, 8, 16, 32],
        "window_size": [7, 7, 7],
    },
    "L": {
        "patch_size": [4, 4, 4],
        "embed_dim": 192,
        "depths": [2, 2, 18, 2],
        "num_heads": [6, 12, 24, 48],
        "window_size": [7, 7, 7],
    },
}


class MLP(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        super().__init__(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop),
        )


def window_partition(
    x: torch.Tensor, window_size: Tuple[int, int, int]
) -> torch.Tensor:
    B, D, H, W, C = x.shape
    wD, wH, wW = window_size
    x = x.view(B, D // wD, wD, H // wH, wH, W // wW, wW, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, wD * wH * wW, C)
    return windows


def window_reverse(
    windows: torch.Tensor, window_size: Tuple[int, int, int], D: int, H: int, W: int
) -> torch.Tensor:
    wD, wH, wW = window_size
    B = int(windows.shape[0] / (D * H * W / (wD * wH * wW)))
    x = windows.view(B, D // wD, H // wH, W // wW, wD, wH, wW, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size_and_shift(
    x_size: Tuple[int, int, int],
    window_size: Tuple[int, int, int],
    shift_size: Tuple[int, int, int],
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    use_window_size = list(window_size)
    use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            use_shift_size[i] = 0
    return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention3D(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1)
                * (2 * window_size[1] - 1)
                * (2 * window_size[2] - 1),
                num_heads,
            )
        )

        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(
            torch.meshgrid([coords_d, coords_h, coords_w], indexing="ij")
        )
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (
            2 * self.window_size[2] - 1
        )
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        shift_size: Tuple[int, int, int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        enable_stable: bool = False,
        stable_lam: float = 1.0,
        stable_beta: float = 0.0,
        use_shakedrop: bool = False,
        shakedrop_alpha_range: Tuple[float, float] = (-1.0, 1.0),
        layer_scale: bool = False,
        layer_scale_init_value: float = 1e-5,
        post_norm: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.enable_stable = enable_stable
        self.stable_lam = stable_lam
        self.stable_beta = stable_beta
        self.post_norm = post_norm

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # Use ShakeDrop if enabled, otherwise use DropPath
        if use_shakedrop and drop_path > 0.0:
            self.drop_path = ShakeDrop(
                p_drop=drop_path, alpha_range=shakedrop_alpha_range
            )
        elif drop_path > 0.0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # LayerScale for attention and MLP branches
        self.ls1 = create_layerscale(dim, layer_scale, layer_scale_init_value)
        self.ls2 = create_layerscale(dim, layer_scale, layer_scale_init_value)

    def forward(
        self, x: torch.Tensor, mask_matrix: Optional[torch.Tensor]
    ) -> torch.Tensor:
        B, D, H, W, C = x.shape
        shortcut = x

        if self.post_norm:
            # Post-norm: attention/MLP → norm → LayerScale → drop_path → residual
            # Attention path
            if any(s > 0 for s in self.shift_size):
                shifted_x = torch.roll(
                    x,
                    shifts=(
                        -self.shift_size[0],
                        -self.shift_size[1],
                        -self.shift_size[2],
                    ),
                    dims=(1, 2, 3),
                )
                attn_mask = mask_matrix
            else:
                shifted_x = x
                attn_mask = None

            x_windows = window_partition(shifted_x, self.window_size)
            attn_windows = self.attn(x_windows, mask=attn_mask)
            shifted_x = window_reverse(attn_windows, self.window_size, D, H, W)

            if any(s > 0 for s in self.shift_size):
                x = torch.roll(
                    shifted_x,
                    shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                    dims=(1, 2, 3),
                )
            else:
                x = shifted_x

            # Apply post-norm for attention path
            attn_out = self.norm1(x)
            attn_out = self.ls1(attn_out) if self.ls1 is not None else attn_out
            if self.enable_stable:
                x = apply_stable_residual(
                    shortcut,
                    self.drop_path(attn_out),
                    self.stable_lam,
                    self.stable_beta,
                )
            else:
                x = shortcut + self.drop_path(attn_out)

            # MLP path with post-norm
            mlp_out = self.mlp(x)
            mlp_out = self.norm2(mlp_out)
            mlp_out = self.ls2(mlp_out) if self.ls2 is not None else mlp_out
            if self.enable_stable:
                x = apply_stable_residual(
                    x, self.drop_path(mlp_out), self.stable_lam, self.stable_beta
                )
            else:
                x = x + self.drop_path(mlp_out)
        else:
            # Pre-norm (original): norm → attention/MLP → LayerScale → drop_path → residual
            x = self.norm1(x)

            if any(s > 0 for s in self.shift_size):
                shifted_x = torch.roll(
                    x,
                    shifts=(
                        -self.shift_size[0],
                        -self.shift_size[1],
                        -self.shift_size[2],
                    ),
                    dims=(1, 2, 3),
                )
                attn_mask = mask_matrix
            else:
                shifted_x = x
                attn_mask = None

            x_windows = window_partition(shifted_x, self.window_size)
            attn_windows = self.attn(x_windows, mask=attn_mask)
            shifted_x = window_reverse(attn_windows, self.window_size, D, H, W)

            if any(s > 0 for s in self.shift_size):
                x = torch.roll(
                    shifted_x,
                    shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                    dims=(1, 2, 3),
                )
            else:
                x = shifted_x

            # Apply LayerScale and stable residual for attention path
            attn_out = self.ls1(x) if self.ls1 is not None else x
            if self.enable_stable:
                x = apply_stable_residual(
                    shortcut,
                    self.drop_path(attn_out),
                    self.stable_lam,
                    self.stable_beta,
                )
            else:
                x = shortcut + self.drop_path(attn_out)

            # Apply LayerScale and stable residual for MLP path
            mlp_out = self.mlp(self.norm2(x))
            mlp_out = self.ls2(mlp_out) if self.ls2 is not None else mlp_out
            if self.enable_stable:
                x = apply_stable_residual(
                    x, self.drop_path(mlp_out), self.stable_lam, self.stable_beta
                )
            else:
                x = x + self.drop_path(mlp_out)

        return x


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        downsample: Optional[Callable[..., nn.Module]] = None,
        use_checkpoint: bool = False,
        enable_stable: bool = False,
        stable_lam: float = 1.0,
        stable_beta: float = 0.0,
        use_shakedrop: bool = False,
        shakedrop_alpha_range: Tuple[float, float] = (-1.0, 1.0),
        layer_scale: bool = False,
        layer_scale_init_value: float = 1e-5,
        post_norm: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=(
                        tuple(0 for _ in self.window_size)
                        if (i % 2 == 0)
                        else self.shift_size
                    ),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    norm_layer=norm_layer,
                    enable_stable=enable_stable,
                    stable_lam=stable_lam,
                    stable_beta=stable_beta,
                    use_shakedrop=use_shakedrop,
                    shakedrop_alpha_range=shakedrop_alpha_range,
                    layer_scale=layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    post_norm=post_norm,
                )
                for i in range(depth)
            ]
        )

        self.downsample = (
            downsample(dim=dim, norm_layer=norm_layer)
            if downsample is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        x = rearrange(x, "b c d h w -> b d h w c")

        pad_d = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        x_padded = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w

        attn_mask = None
        if any(s > 0 for s in self.shift_size):
            img_mask = torch.zeros((1, Dp, Hp, Wp, 1), device=x.device)
            d_slices = (
                slice(0, -self.window_size[0]),
                slice(-self.window_size[0], -self.shift_size[0]),
                slice(-self.shift_size[0], None),
            )
            h_slices = (
                slice(0, -self.window_size[1]),
                slice(-self.window_size[1], -self.shift_size[1]),
                slice(-self.shift_size[1], None),
            )
            w_slices = (
                slice(0, -self.window_size[2]),
                slice(-self.window_size[2], -self.shift_size[2]),
                slice(-self.shift_size[2], None),
            )
            cnt = 0
            for d in d_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, d, h, w, :] = cnt
                        cnt += 1

            mask_windows = window_partition(img_mask, self.window_size).squeeze(-1)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))

        for i, blk in enumerate(self.blocks):
            current_mask = (
                attn_mask
                if blk.shift_size[0] > 0
                or blk.shift_size[1] > 0
                or blk.shift_size[2] > 0
                else None
            )
            if self.use_checkpoint:
                x_padded = checkpoint(blk, x_padded, current_mask, use_reentrant=False)
            else:
                x_padded = blk(x_padded, current_mask)

        # Retirer le padding
        x = x_padded[:, :D, :H, :W, :].contiguous()

        if self.downsample is not None:
            x = self.downsample(x)

        x = rearrange(x, "b d h w c -> b c d h w")
        return x


class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        in_channels: int = 1,
        embed_dim: int = 96,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, D, H, W = x.shape
        pad_d = (self.patch_size[0] - D % self.patch_size[0]) % self.patch_size[0]
        pad_h = (self.patch_size[1] - H % self.patch_size[1]) % self.patch_size[1]
        pad_w = (self.patch_size[2] - W % self.patch_size[2]) % self.patch_size[2]
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

        x = self.proj(x)
        x = rearrange(x, "b c d h w -> b d h w c")
        x = self.norm(x)
        x = rearrange(x, "b d h w c -> b c d h w")
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, C = x.shape

        pad_input = (D % 2 == 1) or (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, D % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class SwinTransformer3DBackbone(nn.Module):
    def __init__(
        self,
        patch_size: List[int],
        in_channels: int,
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        use_checkpoint: bool = False,
        enable_stable: bool = False,
        stable_k: float = 2.0,
        stable_alpha: float = 1.0,
        use_shakedrop: bool = False,
        shakedrop_alpha_range: Tuple[float, float] = (-1.0, 1.0),
        layer_scale: bool = False,
        layer_scale_init_value: float = 1e-5,
        post_norm: bool = False,
    ):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_size = to_3tuple(patch_size)
        self.window_size = to_3tuple(window_size)
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

        self.patch_embed = PatchEmbed3D(
            patch_size=self.patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )
        self.pos_drop = nn.Dropout(p=dropout)

        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_prob, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=dropout,
                attn_drop=attention_dropout,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                enable_stable=enable_stable,
                stable_lam=self.stable_lam,
                stable_beta=self.stable_beta,
                use_shakedrop=use_shakedrop,
                shakedrop_alpha_range=shakedrop_alpha_range,
                layer_scale=layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                post_norm=post_norm,
            )
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        # Apply appropriate initialization
        if enable_stable:
            # Apply stable initialization
            init_stable_model(
                model=self,
                total_blocks=self.total_blocks,
                base_dim=embed_dim,
                dropout_prob=dropout,
                attention_dropout_prob=attention_dropout,
            )
        else:
            # Standard initialization
            self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = rearrange(x, "b c d h w -> b d h w c")
        x = self.norm(x)
        x = rearrange(x, "b d h w c -> b c d h w")

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class SwinTransformer(nn.Module):
    def __init__(
        self,
        patch_size: List[int],
        in_channels: int,
        num_classes: int,
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float,
        qkv_bias: bool,
        dropout: float,
        attention_dropout: float,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        use_checkpoint: bool = False,
        enable_stable: bool = False,
        stable_k: float = 2.0,
        stable_alpha: float = 1.0,
        use_shakedrop: bool = False,
        shakedrop_alpha_range: Tuple[float, float] = (-1.0, 1.0),
        layer_scale: bool = False,
        layer_scale_init_value: float = 1e-5,
        post_norm: bool = False,
    ):
        super().__init__()

        self.backbone = SwinTransformer3DBackbone(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            enable_stable=enable_stable,
            stable_k=stable_k,
            stable_alpha=stable_alpha,
            use_shakedrop=use_shakedrop,
            shakedrop_alpha_range=shakedrop_alpha_range,
            layer_scale=layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            post_norm=post_norm,
        )
        self.head = (
            nn.Linear(self.backbone.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x


class SwinTransformerT(SwinTransformer):
    def __init__(self, **kwargs):
        config = {**_SWIN_CONFIGS["T"], **kwargs}
        super().__init__(**config)


class SwinTransformerS(SwinTransformer):
    def __init__(self, **kwargs):
        config = {**_SWIN_CONFIGS["S"], **kwargs}
        super().__init__(**config)


class SwinTransformerB(SwinTransformer):
    def __init__(self, **kwargs):
        config = {**_SWIN_CONFIGS["B"], **kwargs}
        super().__init__(**config)


class SwinTransformerL(SwinTransformer):
    def __init__(self, **kwargs):
        config = {**_SWIN_CONFIGS["L"], **kwargs}
        super().__init__(**config)
