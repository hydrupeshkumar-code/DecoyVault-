"""
Restormer cloud removal service layer.

Central service used by the /reconstruct API route.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from backend.services.model_loader import get_device, get_restormer
from ai.restormer.tiled_inference import tiled_infer


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

    if model is None:
        raise RuntimeError(
            "Restormer model is not loaded.  Call model_loader.load_restormer() at startup."
        )

    t0 = time.perf_counter()

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
    }
