"""Standalone PSNR metric."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio in dB, averaged over the batch.

    Args:
        pred:    [B, C, H, W] predicted tensor in [0, max_val].
        target:  [B, C, H, W] ground-truth tensor in [0, max_val].
        max_val: Maximum value of the data range (default 1.0).

    Returns:
        Scalar PSNR in dB.  Higher is better.
    """
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    return (10.0 * torch.log10(max_val ** 2 / mse.clamp(min=1e-10))).mean()
