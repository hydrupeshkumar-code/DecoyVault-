"""Standalone NDVI MAE metric."""

from __future__ import annotations

import torch


def ndvi_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    NDVI Mean Absolute Error.

    Channels expected at indices: 0=Green, 1=Red, 2=NIR.

    Args:
        pred:   [B, 3, H, W] predicted image in [0, 1].
        target: [B, 3, H, W] ground-truth image in [0, 1].
        mask:   Optional [B, 1, H, W] binary mask; if provided, computes
                MAE only within masked (cloud) regions.
        eps:    Epsilon for NDVI denominator stability.

    Returns:
        Scalar NDVI MAE.  Lower is better.
    """
    def _ndvi(x: torch.Tensor) -> torch.Tensor:
        red = x[:, 1:2]
        nir = x[:, 2:3]
        return (nir - red) / (nir + red + eps)

    diff = (_ndvi(pred) - _ndvi(target)).abs()  # [B, 1, H, W]

    if mask is not None:
        diff = diff * mask
        return diff.sum() / mask.sum().clamp(min=1.0)

    return diff.mean()
