"""
Restormer backbone for cloud removal.

Architecture:
  - Multi-Dconv Head Transposed Attention (MDTA)
  - Gated-Dconv Feed-Forward Network (GDFN)
  - U-Net encoder-decoder with skip connections
  - Input/Output: [B, 3, H, W]  (Green, Red, NIR)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for feature maps [B, C, H, W]."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


# ---------------------------------------------------------------------------
# MDTA – Multi-Dconv Head Transposed Attention
# ---------------------------------------------------------------------------

class MDTA(nn.Module):
    """
    Multi-Dconv Head Transposed Attention block.

    Operates attention across the channel dimension rather than spatial,
    which is O(C^2) instead of O(N^2) for large spatial resolutions.
    """

    def __init__(self, dim: int, num_heads: int, bias: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3,
            kernel_size=3, stride=1, padding=1,
            groups=dim * 3, bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=H, w=W)
        return self.project_out(out)


# ---------------------------------------------------------------------------
# GDFN – Gated-Dconv Feed-Forward Network
# ---------------------------------------------------------------------------

class GDFN(nn.Module):
    """
    Gated-Dconv Feed-Forward Network with depth-wise separable convolutions
    and a gating mechanism for selective feature propagation.
    """

    def __init__(self, dim: int, ffn_expansion_factor: float = 2.66, bias: bool = False) -> None:
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2,
            kernel_size=3, stride=1, padding=1,
            groups=hidden_features * 2, bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Single Restormer transformer block: LN → MDTA → LN → GDFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Down / Up sampling
# ---------------------------------------------------------------------------

class DownSample(nn.Module):
    """Pixel-unshuffle based downsampling (×2)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class UpSample(nn.Module):
    """Pixel-shuffle based upsampling (×2)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ---------------------------------------------------------------------------
# Restormer
# ---------------------------------------------------------------------------

class Restormer(nn.Module):
    """
    Full Restormer U-Net for cloud removal.

    Encoder: 3 scales, each with N transformer blocks.
    Bottleneck: transformer blocks at lowest resolution.
    Decoder: mirror of encoder with skip connections.

    Args:
        in_channels:         Number of input channels (3 for Green/Red/NIR).
        out_channels:        Number of output channels (3).
        dim:                 Base feature dimension.
        num_blocks:          Blocks per encoder level [L1, L2, L3, bottleneck].
        num_refinement_blocks: Blocks in the final refinement stage.
        heads:               Attention heads per level.
        ffn_expansion_factor: GDFN expansion ratio.
        bias:                Use bias in convolutions.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: list[int] | None = None,
        num_refinement_blocks: int = 4,
        heads: list[int] | None = None,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if heads is None:
            heads = [1, 2, 4, 8]

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1, bias=bias)

        # ---- Encoder ----
        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim, heads[0], ffn_expansion_factor, bias) for _ in range(num_blocks[0])]
        )
        self.down1_2 = DownSample(dim)

        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias) for _ in range(num_blocks[1])]
        )
        self.down2_3 = DownSample(dim * 2)

        self.encoder_level3 = nn.Sequential(
            *[TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias) for _ in range(num_blocks[2])]
        )
        self.down3_4 = DownSample(dim * 4)

        # ---- Bottleneck ----
        self.latent = nn.Sequential(
            *[TransformerBlock(dim * 8, heads[3], ffn_expansion_factor, bias) for _ in range(num_blocks[3])]
        )

        # ---- Decoder ----
        self.up4_3 = UpSample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *[TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias) for _ in range(num_blocks[2])]
        )

        self.up3_2 = UpSample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias) for _ in range(num_blocks[1])]
        )

        self.up2_1 = UpSample(dim * 2)
        # skip from encoder_level1 is concatenated → dim + dim = 2*dim, reduce back
        self.reduce_chan_level1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(
            *[TransformerBlock(dim, heads[0], ffn_expansion_factor, bias) for _ in range(num_blocks[0])]
        )

        # ---- Refinement ----
        self.refinement = nn.Sequential(
            *[TransformerBlock(dim, heads[0], ffn_expansion_factor, bias) for _ in range(num_refinement_blocks)]
        )

        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the full encoder-decoder and return the refined decoder feature
        map (before the final output projection).

        Args:
            x: [B, 3, H, W] cloudy input (Green, Red, NIR).

        Returns:
            [B, dim, H, W] decoder features. This is the correct input for the
            Cloud-Adaptive Residual Head — NOT the 3-channel reconstructed
            image.
        """
        inp_enc_level1 = self.patch_embed(x)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)

        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        return self.refinement(out_dec_level1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] cloudy input (Green, Red, NIR).

        Returns:
            [B, 3, H, W] restored output.
        """
        feats = self.forward_features(x)
        return self.output(feats) + x  # global residual
