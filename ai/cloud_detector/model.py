"""
Lightweight CNN cloud detector for LISS-IV 3-band imagery.

Architecture: U-Net-inspired encoder-decoder with a binary segmentation head.
Input:  [B, 3, H, W]  (Green, Red, NIR)
Output: [B, 1, H, W]  cloud probability map ∈ [0, 1]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CloudDetector(nn.Module):
    """
    U-Net cloud segmentation model.

    Args:
        in_channels: Number of input spectral bands (3).
        base_dim:    Feature width at the first encoder level.
    """

    def __init__(self, in_channels: int = 3, base_dim: int = 32) -> None:
        super().__init__()
        d = base_dim

        self.enc1 = _DoubleConv(in_channels, d)
        self.enc2 = _DoubleConv(d, d * 2)
        self.enc3 = _DoubleConv(d * 2, d * 4)
        self.bottleneck = _DoubleConv(d * 4, d * 8)

        self.up3 = nn.ConvTranspose2d(d * 8, d * 4, 2, stride=2)
        self.dec3 = _DoubleConv(d * 8, d * 4)
        self.up2 = nn.ConvTranspose2d(d * 4, d * 2, 2, stride=2)
        self.dec2 = _DoubleConv(d * 4, d * 2)
        self.up1 = nn.ConvTranspose2d(d * 2, d, 2, stride=2)
        self.dec1 = _DoubleConv(d * 2, d)

        self.out_conv = nn.Conv2d(d, 1, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))
