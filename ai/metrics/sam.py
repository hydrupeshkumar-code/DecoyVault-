"""Standalone Spectral Angle Mapper (SAM) metric."""

from __future__ import annotations

import torch


def sam(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    degrees: bool = False,
) -> torch.Tensor:
    """
    Spectral Angle Mapper metric averaged over all pixels in the batch.

    Args:
        pred:    [B, C, H, W] predicted tensor.
        target:  [B, C, H, W] ground-truth tensor.
        eps:     Numerical stability epsilon.
        degrees: Return result in degrees instead of radians.

    Returns:
        Scalar SAM.  Lower is better.
    """
    dot = (pred * target).sum(dim=1)
    # sqrt(sum+eps) avoids the NaN *gradient* that norm().clamp() has at zero pixels
    norm_p = torch.sqrt((pred * pred).sum(dim=1) + eps)
    norm_t = torch.sqrt((target * target).sum(dim=1) + eps)
    cos = (dot / (norm_p * norm_t)).clamp(-1 + eps, 1 - eps)
    angle = torch.acos(cos)
    if degrees:
        angle = torch.rad2deg(angle)
    return angle.mean()
