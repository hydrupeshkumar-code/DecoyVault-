"""
Cloud-Adaptive Residual Head.

Formulation:
    features_scaled = features * (1 + alpha * cloud_mask)
    predicted_residual = head(features_scaled)
    final_output = cloudy_input + (predicted_residual * cloud_mask)

Cloud-free pixels are guaranteed to remain unchanged because the residual
is multiplied by the binary cloud mask before being added to the input.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CloudAdaptiveResidualHead(nn.Module):
    """
    Cloud-adaptive residual prediction head.

    Takes the decoder feature map and a cloud mask, amplifies features
    in cloudy regions via a learned scalar alpha, predicts a per-pixel
    residual, then adds it only at cloud locations.

    Args:
        in_channels:  Number of feature channels from the decoder.
        out_channels: Number of output image channels (3 for Green/Red/NIR).
        mid_channels: Intermediate channel width in the residual head.
    """

    def __init__(
        self,
        in_channels: int = 48,
        out_channels: int = 3,
        mid_channels: int = 32,
    ) -> None:
        super().__init__()

        # Learnable cloud-region amplification scalar
        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        features: torch.Tensor,
        cloudy_input: torch.Tensor,
        cloud_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features:     [B, C, H, W]  decoder feature map.
            cloudy_input: [B, 3, H, W]  original cloudy image.
            cloud_mask:   [B, 1, H, W]  binary cloud mask (1 = cloud).

        Returns:
            [B, 3, H, W]  reconstructed image; clear pixels are identical
                          to ``cloudy_input``.
        """
        # Ensure cloud_mask has a channel dim and matches feature resolution
        if cloud_mask.dim() == 3:
            cloud_mask = cloud_mask.unsqueeze(1)

        mask = F.interpolate(
            cloud_mask.float(),
            size=features.shape[-2:],
            mode="nearest",
        )

        # Amplify features in cloud regions
        features_scaled = features * (1.0 + self.alpha * mask)

        # Predict residual
        residual = self.head(features_scaled)

        # Upsample residual back to input resolution if needed
        if residual.shape[-2:] != cloudy_input.shape[-2:]:
            residual = F.interpolate(
                residual, size=cloudy_input.shape[-2:], mode="bilinear", align_corners=False
            )
            mask_input = F.interpolate(
                mask, size=cloudy_input.shape[-2:], mode="nearest"
            )
        else:
            mask_input = mask

        # Apply residual only at cloud locations
        return cloudy_input + residual * mask_input
