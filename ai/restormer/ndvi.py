"""
NDVI utilities for satellite imagery quality assessment.

NDVI = (NIR - RED) / (NIR + RED)

Band layout assumed throughout: channel 0 = Green, 1 = Red, 2 = NIR.
"""

from __future__ import annotations

import numpy as np
import torch


_EPS = 1e-8


def compute_ndvi(image: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """
    Compute NDVI from a 3-band image tensor.

    Args:
        image: [B, 3, H, W] or [3, H, W] with channels (Green, Red, NIR).
        eps:   Small value to avoid division by zero.

    Returns:
        NDVI map with the same leading batch dimensions, shape [B, 1, H, W]
        or [1, H, W].  Values are in [-1, 1].
    """
    if image.dim() == 3:
        red = image[1:2]   # [1, H, W]
        nir = image[2:3]   # [1, H, W]
    elif image.dim() == 4:
        red = image[:, 1:2]  # [B, 1, H, W]
        nir = image[:, 2:3]  # [B, 1, H, W]
    else:
        raise ValueError(f"Expected 3-D or 4-D tensor, got {image.dim()}-D")

    return (nir - red) / (nir + red + eps)


def compute_ndvi_numpy(image: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """
    Compute NDVI from a numpy array.

    Args:
        image: [..., 3] HWC or [3, H, W] CHW array with (Green, Red, NIR).
        eps:   Small value to avoid division by zero.

    Returns:
        NDVI array matching the spatial dimensions, values in [-1, 1].
    """
    if image.shape[0] == 3 and image.ndim == 3:
        # CHW layout
        red = image[1]
        nir = image[2]
    else:
        # HWC layout
        red = image[..., 1]
        nir = image[..., 2]

    return (nir - red) / (nir + red + eps)


def compute_ndvi_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = _EPS,
) -> torch.Tensor:
    """
    Compute NDVI Mean Absolute Error.

    Args:
        pred:   [B, 3, H, W] predicted image.
        target: [B, 3, H, W] ground-truth image.
        mask:   Optional [B, 1, H, W] binary mask restricting evaluation to
                cloud regions.  If None, computes globally.
        eps:    Epsilon for NDVI stability.

    Returns:
        Scalar NDVI MAE.
    """
    pred_ndvi = compute_ndvi(pred, eps)       # [B, 1, H, W]
    target_ndvi = compute_ndvi(target, eps)   # [B, 1, H, W]
    diff = (pred_ndvi - target_ndvi).abs()

    if mask is not None:
        diff = diff * mask
        denom = mask.sum().clamp(min=1.0)
        return diff.sum() / denom

    return diff.mean()


def ndvi_reconstruction_quality(
    cloudy: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = _EPS,
) -> dict[str, float]:
    """
    Summarise NDVI quality for a batch.

    Returns a dictionary with:
        - ndvi_mae_cloud:      MAE inside cloud regions.
        - ndvi_mae_global:     MAE over the whole image.
        - ndvi_improvement:    Reduction in MAE vs. the raw cloudy input.
    """
    mae_cloud = compute_ndvi_mae(pred, target, mask, eps).item()
    mae_global = compute_ndvi_mae(pred, target, None, eps).item()
    mae_baseline = compute_ndvi_mae(cloudy, target, mask, eps).item()

    improvement = mae_baseline - mae_cloud

    return {
        "ndvi_mae_cloud": mae_cloud,
        "ndvi_mae_global": mae_global,
        "ndvi_improvement": improvement,
    }
