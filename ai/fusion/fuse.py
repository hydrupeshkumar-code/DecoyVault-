"""
Multi-source image fusion for cloud removal.

When a partial cloud mask allows, uses a confidence-weighted blend of:
    - Restormer reconstruction (in cloud regions)
    - Original cloudy image (in clear regions)
    - Optional auxiliary clear-sky composite (temporal reference)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def confidence_weighted_fuse(
    cloudy: torch.Tensor,
    reconstructed: torch.Tensor,
    cloud_mask: torch.Tensor,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fuse cloudy and reconstructed images using the cloud mask as blending weight.

    Args:
        cloudy:        [B, 3, H, W] original cloudy image.
        reconstructed: [B, 3, H, W] Restormer output.
        cloud_mask:    [B, 1, H, W] binary mask (1 = cloud).
        confidence:    Optional [B, 1, H, W] model confidence map in [0, 1].
                       When provided, used as soft blend weight inside cloud
                       regions instead of the hard mask.

    Returns:
        [B, 3, H, W] fused image.
    """
    if confidence is not None:
        alpha = cloud_mask * confidence
    else:
        alpha = cloud_mask.float()

    return cloudy * (1.0 - alpha) + reconstructed * alpha


def temporal_cloud_fill(
    reconstructed: torch.Tensor,
    temporal_clear: torch.Tensor,
    cloud_mask: torch.Tensor,
    temporal_weight: float = 0.3,
) -> torch.Tensor:
    """
    Optionally blend the Restormer reconstruction with a temporal clear composite.

    Useful when a cloud-free image from a nearby date is available.

    Args:
        reconstructed:  [B, 3, H, W] Restormer output.
        temporal_clear: [B, 3, H, W] temporally-adjacent clear image.
        cloud_mask:     [B, 1, H, W] cloud mask.
        temporal_weight: Weight given to the temporal image inside cloud regions.

    Returns:
        [B, 3, H, W] blended image.
    """
    w = cloud_mask.float() * temporal_weight
    return reconstructed * (1.0 - w) + temporal_clear * w
