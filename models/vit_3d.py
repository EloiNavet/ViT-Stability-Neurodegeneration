"""
3D Vision Transformer adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_3d.py
"""

from typing import Tuple, Literal, Dict, Optional
import torch
from torch import nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.layers import trunc_normal_, DropPath
from einops.layers.torch import Rearrange
from models.modules.layerscale import create_layerscale
from utils.stable_init import (
    compute_residual_gains,
    apply_stable_residual,
    init_stable_model,
)

_VIT_CONFIGS: Dict[str, Dict[str, int]] = {
    "S": {
        "depth": 12,
        "num_heads": 6,
        "embed_dim": 384,
        "img_size": (96, 96, 96),
        "patch_size": (16, 16, 16),
    },
    "B": {
        "depth": 12,
        "num_heads": 12,
        "embed_dim": 768,
        "img_size": (96, 96, 96),
        "patch_size": (16, 16, 16),
    },
    "L": {
        "depth": 24,
        "num_heads": 16,
        "embed_dim": 1024,
        "img_size": (96, 96, 96),
        "patch_size": (16, 16, 16),
    },
    "H": {
        "depth": 32,
        "num_heads": 16,
        "embed_dim": 1280,
        "img_size": (96, 96, 96),
        "patch_size": (16, 16, 16),
    },
}


class FeedForward(nn.Module):
    def __init__(
        self, dim: int, hidden_dim: int, dropout: float = 0.0, post_norm: bool = False
    ) -> None:
        super().__init__()
        self.post_norm = post_norm
        if post_norm:
            # Post-norm: no LayerNorm in the sequential
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout),
            )
        else:
            # Pre-norm: LayerNorm at the beginning
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        post_norm: bool = False,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.post_norm = post_norm

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.post_norm:
            # Post-norm: compute attention first, then apply norm
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(
                lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
            )

            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            attn = self.attend(dots)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)
            out = rearrange(out, "b h n d -> b n (h d)")
            out = self.to_out(out)
            return out  # Norm will be applied in Transformer block
        else:
            # Pre-norm: apply norm first
            x = self.norm(x)
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(
                lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
            )

            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            attn = self.attend(dots)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)
            out = rearrange(out, "b h n d -> b n (h d)")
            return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        use_checkpoint: bool = False,
        enable_stable: bool = False,
        stable_lam: float = 1.0,
        stable_beta: float = 0.0,
        layer_scale: bool = False,
        layer_scale_init_value: float = 1e-5,
        post_norm: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        self.enable_stable = enable_stable
        self.stable_lam = stable_lam
        self.stable_beta = stable_beta
        self.post_norm = post_norm

        # Stochastic depth decay rule: linearly increasing drop rate
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_prob, depth)]

        for i in range(depth):
            # Create LayerScale modules for this block
            ls1 = create_layerscale(dim, layer_scale, layer_scale_init_value)
            ls2 = create_layerscale(dim, layer_scale, layer_scale_init_value)
            # Create DropPath for this block
            drop_path = DropPath(dpr[i]) if dpr[i] > 0.0 else nn.Identity()
            # Create LayerNorm modules for post-norm
            attn_norm = nn.LayerNorm(dim) if post_norm else None
            ff_norm = nn.LayerNorm(dim) if post_norm else None
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=attention_dropout,
                            post_norm=post_norm,
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout, post_norm=post_norm),
                        ls1,
                        ls2,
                        drop_path,
                        attn_norm,
                        ff_norm,
                    ]
                )
            )
        self.use_checkpoint = use_checkpoint

    def _forward_block(
        self,
        attn: Attention,
        ff: FeedForward,
        ls1,
        ls2,
        drop_path,
        attn_norm,
        ff_norm,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.post_norm:
            # Post-norm: attn/ff → norm → LayerScale → DropPath → residual
            attn_out = attn(x)
            attn_out = attn_norm(attn_out)
            attn_out = ls1(attn_out) if ls1 is not None else attn_out
            attn_out = drop_path(attn_out)
            if self.enable_stable:
                x = apply_stable_residual(
                    x, attn_out, self.stable_lam, self.stable_beta
                )
            else:
                x = x + attn_out

            ff_out = ff(x)
            ff_out = ff_norm(ff_out)
            ff_out = ls2(ff_out) if ls2 is not None else ff_out
            ff_out = drop_path(ff_out)
            if self.enable_stable:
                x = apply_stable_residual(x, ff_out, self.stable_lam, self.stable_beta)
            else:
                x = x + ff_out
        else:
            # Pre-norm: norm → attn/ff → LayerScale → DropPath → residual
            attn_out = attn(x)
            attn_out = ls1(attn_out) if ls1 is not None else attn_out
            attn_out = drop_path(attn_out)
            if self.enable_stable:
                x = apply_stable_residual(
                    x, attn_out, self.stable_lam, self.stable_beta
                )
            else:
                x = x + attn_out

            ff_out = ff(x)
            ff_out = ls2(ff_out) if ls2 is not None else ff_out
            ff_out = drop_path(ff_out)
            if self.enable_stable:
                x = apply_stable_residual(x, ff_out, self.stable_lam, self.stable_beta)
            else:
                x = x + ff_out
        return x

    def _checkpoint_block(
        self,
        attn: Attention,
        ff: FeedForward,
        ls1,
        ls2,
        drop_path,
        attn_norm,
        ff_norm,
        x: torch.Tensor,
    ) -> torch.Tensor:
        def custom_forward(tensor: torch.Tensor) -> torch.Tensor:
            return self._forward_block(
                attn, ff, ls1, ls2, drop_path, attn_norm, ff_norm, tensor
            )

        return checkpoint.checkpoint(custom_forward, x, use_reentrant=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff, ls1, ls2, drop_path, attn_norm, ff_norm in self.layers:
            if self.use_checkpoint and not torch.jit.is_scripting():
                x = self._checkpoint_block(
                    attn, ff, ls1, ls2, drop_path, attn_norm, ff_norm, x
                )
            else:
                x = self._forward_block(
                    attn, ff, ls1, ls2, drop_path, attn_norm, ff_norm, x
                )
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        img_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        pool: Literal["cls", "mean"] = "cls",
        in_channels: int = 1,
        dim_head: int = 64,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        use_checkpoint: bool = False,
        enable_stable: bool = False,
        stable_k: float = 2.0,
        stable_alpha: float = 1.0,
        layer_scale: bool = False,
        layer_scale_init_value: float = 1e-5,
        post_norm: bool = False,
    ) -> None:
        """
        Args:
            img_size: Input volume size as (depth, height, width) or single int
            patch_size: Patch dimensions as (depth, height, width) or single int
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_dim: Hidden dimension of MLP blocks
            pool: Pooling method - "cls" for class token or "mean" for mean pooling
            in_channels: Number of input channels
            dim_head: Dimension per attention head
            attention_dropout: Dropout rate in attention layers
            dropout: Dropout rate in feed-forward layers and also after positional embeddings
            stochastic_depth_prob: Stochastic depth dropout probability (DropPath)
            use_checkpoint: If True, enables gradient checkpointing in transformer blocks
                to reduce memory usage during training.
        """
        super().__init__()

        # Convert to 3D tuples
        img_d, img_h, img_w = img_size
        patch_d, patch_h, patch_w = patch_size

        assert img_d % patch_d == 0 and img_h % patch_h == 0 and img_w % patch_w == 0, (
            f"Image dimensions ({img_d}, {img_h}, {img_w}) must be divisible by "
            f"patch size ({patch_d}, {patch_h}, {patch_w})"
        )

        num_patches = (img_d // patch_d) * (img_h // patch_h) * (img_w // patch_w)
        patch_dim = in_channels * patch_d * patch_h * patch_w

        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either 'cls' (cls token) or 'mean' (mean pooling)"

        self.enable_stable = enable_stable
        self.total_blocks = depth

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

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)",
                p1=patch_d,
                p2=patch_h,
                p3=patch_w,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(
            dim=embed_dim,
            depth=depth,
            heads=num_heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            attention_dropout=attention_dropout,
            dropout=dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            use_checkpoint=use_checkpoint,
            enable_stable=enable_stable,
            stable_lam=self.stable_lam,
            stable_beta=self.stable_beta,
            layer_scale=layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            post_norm=post_norm,
        )

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes)
        )
        self.use_checkpoint = use_checkpoint

        # Apply appropriate initialization
        if enable_stable:
            # Apply stable initialization
            init_stable_model(
                model=self,
                total_blocks=self.total_blocks,
                base_dim=embed_dim,
                dropout_prob=dropout,
                attention_dropout_prob=attention_dropout,
                attention_module_types=("Attention",),
                mlp_module_types=("FeedForward",),
            )
        else:
            # Standard initialization
            self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _checkpoint_forward(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Helper to checkpoint a module forward pass"""

        def custom_forward(tensor: torch.Tensor) -> torch.Tensor:
            return module(tensor)

        return checkpoint.checkpoint(custom_forward, x, use_reentrant=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply gradient checkpointing to patch embedding if enabled
        if self.use_checkpoint and self.training and not torch.jit.is_scripting():
            x = self._checkpoint_forward(self.to_patch_embedding, x)
        else:
            x = self.to_patch_embedding(x)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class ViTX(ViT):
    def __init__(
        self,
        config_name: str,
        img_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        num_classes: int,
        mlp_ratio: Optional[float] = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        in_channels: int = 1,
        dim_head: int = 64,
        pool: Literal["cls", "mean"] = "cls",
        use_checkpoint: bool = False,
        **overrides,
    ) -> None:
        if config_name not in _VIT_CONFIGS:
            raise ValueError(
                f"Unknown config_name '{config_name}'. Available: {list(_VIT_CONFIGS.keys())}"
            )
        conf = dict(_VIT_CONFIGS[config_name])

        embed_dim = overrides.pop("embed_dim", conf["embed_dim"])
        depth = overrides.pop("depth", conf["depth"])
        num_heads = overrides.pop("num_heads", conf["num_heads"])
        default_mlp_ratio = (
            mlp_ratio if mlp_ratio is not None else overrides.pop("mlp_ratio", 4.0)
        )
        mlp_dim = overrides.pop("mlp_dim", int(embed_dim * default_mlp_ratio))

        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            pool=pool,
            in_channels=in_channels,
            dim_head=dim_head,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            use_checkpoint=use_checkpoint,
            **overrides,
        )


class ViTS(ViTX):
    def __init__(self, *args, **kwargs):
        super().__init__(config_name="S", *args, **kwargs)


class ViTB(ViTX):
    def __init__(self, *args, **kwargs):
        super().__init__(config_name="B", *args, **kwargs)


class ViTL(ViTX):
    def __init__(self, *args, **kwargs):
        super().__init__(config_name="L", *args, **kwargs)


class ViTH(ViTX):
    def __init__(self, *args, **kwargs):
        super().__init__(config_name="H", *args, **kwargs)
