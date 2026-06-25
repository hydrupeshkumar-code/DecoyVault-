"""Standalone SSIM metric."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """
    Structural Similarity Index Measure, averaged over the batch.

    Args:
        pred:        [B, C, H, W] predicted tensor in [0, 1].
        target:      [B, C, H, W] ground-truth tensor in [0, 1].
        window_size: Gaussian kernel size.
        sigma:       Gaussian standard deviation.
        C1:          Stability constant for luminance.
        C2:          Stability constant for contrast.

    Returns:
        Scalar SSIM in [0, 1].  Higher is better.
    """
    C = pred.shape[1]
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    kernel = kernel.expand(C, 1, window_size, window_size)
    pad = window_size // 2

    mu1 = F.conv2d(pred, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=pad, groups=C) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2).clamp(min=1e-8)
    return (numerator / denominator).mean()
