"""
Tiled inference with Hann-window blending.

For large satellite images that exceed GPU memory limits, the image is split
into overlapping 256×256 tiles. Each tile is processed independently, then
the outputs are blended using a Hann window to eliminate visible seam artefacts.

Parameters
----------
tile_size : int
    Width and height of each tile (default 256).
overlap   : int
    Pixel overlap between adjacent tiles (default 32).  Must be even.
window    : str
    'hann' (default) or 'gaussian'.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from ai.restormer.model import Restormer


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------

def _hann_window_2d(size: int) -> np.ndarray:
    """2-D Hann window of shape [size, size]."""
    hann_1d = np.hanning(size).astype(np.float32)
    return np.outer(hann_1d, hann_1d)


def _gaussian_window_2d(size: int, sigma_ratio: float = 0.25) -> np.ndarray:
    """2-D Gaussian window of shape [size, size]."""
    sigma = size * sigma_ratio
    c = size // 2
    y, x = np.ogrid[:size, :size]
    g = np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * sigma ** 2)).astype(np.float32)
    return g


def _make_window(window: str, size: int) -> np.ndarray:
    if window == "hann":
        return _hann_window_2d(size)
    if window == "gaussian":
        return _gaussian_window_2d(size)
    raise ValueError(f"Unknown window type: {window!r}. Use 'hann' or 'gaussian'.")


# ---------------------------------------------------------------------------
# Tiled inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def tiled_infer(
    model: "Restormer",
    image: np.ndarray,
    mask: np.ndarray | None,
    device: torch.device,
    tile_size: int = 256,
    overlap: int = 32,
    window: str = "hann",
) -> np.ndarray:
    """
    Perform tiled inference with Hann-window blending.

    Args:
        model:     Trained Restormer in eval mode.
        image:     [H, W, 3] float32 array in [0, 1].
        mask:      Optional [H, W] binary cloud mask.  When provided, only
                   cloud regions in the output are replaced; clear pixels
                   keep their original values.
        device:    Torch device.
        tile_size: Size of each square tile.
        overlap:   Overlap (pixels) between adjacent tiles.  Must be >= 0.
        window:    Blending window type ('hann' or 'gaussian').

    Returns:
        [H, W, 3] blended output float32 array in [0, 1].
    """
    H, W, C = image.shape
    step = tile_size - overlap

    # Pad image so all tiles are exactly tile_size×tile_size
    pad_h = math.ceil(max(H - tile_size, 0) / step) * step + tile_size - H
    pad_w = math.ceil(max(W - tile_size, 0) / step) * step + tile_size - W
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    PH, PW = padded.shape[:2]

    weight_map = np.zeros((PH, PW, 1), dtype=np.float32)
    output_map = np.zeros((PH, PW, C), dtype=np.float32)

    win = _make_window(window, tile_size)[..., np.newaxis]  # [T, T, 1]

    y_starts = list(range(0, PH - tile_size + 1, step))
    x_starts = list(range(0, PW - tile_size + 1, step))

    for y in y_starts:
        for x in x_starts:
            tile = padded[y : y + tile_size, x : x + tile_size]       # [T, T, 3]
            t_in = (
                torch.from_numpy(tile.transpose(2, 0, 1))
                .unsqueeze(0)
                .to(device)
            )                                                          # [1, 3, T, T]
            t_out = model(t_in)                                        # [1, 3, T, T]
            t_out_np = (
                t_out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            )                                                          # [T, T, 3]

            output_map[y : y + tile_size, x : x + tile_size] += t_out_np * win
            weight_map[y : y + tile_size, x : x + tile_size] += win

    # Normalise by accumulated weights
    blended = output_map / weight_map.clip(min=1e-8)

    # Crop back to original size
    result = blended[:H, :W]

    # Apply cloud mask: preserve clear pixels from original input
    if mask is not None:
        mask_3 = np.stack([mask, mask, mask], axis=-1)
        result = image * (1.0 - mask_3) + result * mask_3

    return np.clip(result, 0.0, 1.0).astype(np.float32)
