"""
Restormer cloud removal service layer.

Central service used by the /reconstruct API route.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from backend.services.model_loader import get_device, get_restormer
from ai.restormer.tiled_inference import tiled_infer

log = logging.getLogger(__name__)


def reconstruct(
    image: np.ndarray,
    cloud_mask: Optional[np.ndarray] = None,
    tile_size: int = 256,
    overlap: int = 32,
    preserve_clear: bool = True,
) -> dict[str, np.ndarray | float]:
    """
    Remove clouds from a LISS-IV image using the Restormer model.

    Args:
        image:         [H, W, 3] float32 array in [0, 1] (Green, Red, NIR).
        cloud_mask:    Optional [H, W] binary mask (1 = cloud).  When None,
                       the model processes the entire image.
        tile_size:     Tile size for tiled inference.
        overlap:       Tile overlap.
        preserve_clear: If True and a cloud_mask is provided, clear pixels
                        keep their original values in the output.

    Returns:
        Dictionary with:
            'reconstructed': [H, W, 3] cloud-removed image.
            'elapsed_s':     Inference time in seconds.
    """
    model = get_restormer()
    device = get_device()

    t0 = time.perf_counter()

    if model is None:
        # No trained checkpoint available — degrade gracefully instead of
        # crashing so the demo / API remains usable. Cloud pixels are filled
        # with the per-band mean of the clear region (a flat but plausible
        # in-painting baseline). This is clearly NOT a real reconstruction; the
        # ``fallback`` flag lets callers surface that to the user.
        log.warning(
            "Restormer checkpoint not loaded — using clear-region mean fallback "
            "(no real reconstruction performed)."
        )
        reconstructed = _fallback_reconstruct(image, cloud_mask)
        return {
            "reconstructed": reconstructed,
            "elapsed_s": time.perf_counter() - t0,
            "fallback": True,
        }

    mask_for_infer = cloud_mask if preserve_clear else None

    reconstructed = tiled_infer(
        model=model,
        image=image,
        mask=mask_for_infer,
        device=device,
        tile_size=tile_size,
        overlap=overlap,
    )

    elapsed = time.perf_counter() - t0
    return {
        "reconstructed": reconstructed,
        "elapsed_s": elapsed,
        "fallback": False,
    }


def _fallback_reconstruct(
    image: np.ndarray,
    cloud_mask: Optional[np.ndarray],
) -> np.ndarray:
    """
    Checkpoint-free in-painting baseline.

    Fills cloud-masked pixels with the per-band mean reflectance of the clear
    region. Without a mask there is nothing to in-paint, so the input is
    returned unchanged.

    Args:
        image:      [H, W, 3] float32 in [0, 1] (Green, Red, NIR).
        cloud_mask: Optional [H, W] (or [1, H, W]) binary mask (1 = cloud).

    Returns:
        [H, W, 3] float32 in [0, 1].
    """
    out = image.astype(np.float32).copy()
    if cloud_mask is None:
        return out

    mask = np.asarray(cloud_mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask.squeeze()
    cloud = mask > 0.5
    clear = ~cloud
    if cloud.sum() == 0 or clear.sum() == 0:
        return out

    for b in range(out.shape[-1]):
        band = out[..., b]
        band[cloud] = float(band[clear].mean())

    return np.clip(out, 0.0, 1.0).astype(np.float32)
