"""
Temporal comparison utilities.

Provides tools to compare cloud-removed images against a temporally adjacent
clear reference and to select the best reconstruction across multiple dates.
"""

from __future__ import annotations

import numpy as np
import torch

from ai.restormer.ndvi import compute_ndvi_numpy


def temporal_difference_map(
    image_t1: np.ndarray,
    image_t2: np.ndarray,
) -> np.ndarray:
    """
    Compute a per-pixel change magnitude between two aligned images.

    Args:
        image_t1: [H, W, 3] image at time 1 in [0, 1].
        image_t2: [H, W, 3] image at time 2 in [0, 1].

    Returns:
        [H, W] change magnitude map (sum of absolute band differences).
    """
    diff = np.abs(image_t1.astype(np.float32) - image_t2.astype(np.float32))
    return diff.sum(axis=-1)


def ndvi_change_map(
    image_t1: np.ndarray,
    image_t2: np.ndarray,
) -> np.ndarray:
    """
    NDVI change map between two dates.

    Args:
        image_t1: [H, W, 3] (Green, Red, NIR) at time 1.
        image_t2: [H, W, 3] (Green, Red, NIR) at time 2.

    Returns:
        [H, W] NDVI difference (t2 - t1) in [-2, 2].
    """
    ndvi1 = compute_ndvi_numpy(image_t1)
    ndvi2 = compute_ndvi_numpy(image_t2)
    return ndvi2 - ndvi1


def select_best_reconstruction(
    candidates: list[np.ndarray],
    reference: np.ndarray,
    cloud_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Select the candidate reconstruction closest to the reference NDVI in
    cloud regions.

    Args:
        candidates:  List of [H, W, 3] reconstructed images.
        reference:   [H, W, 3] ground-truth or clear temporal reference.
        cloud_mask:  [H, W] binary mask of cloud regions.

    Returns:
        (best_image, best_index) tuple.
    """
    ref_ndvi = compute_ndvi_numpy(reference)
    msk = cloud_mask > 0.5

    best_mae = float("inf")
    best_idx = 0

    for i, cand in enumerate(candidates):
        cand_ndvi = compute_ndvi_numpy(cand)
        mae = float(np.abs(cand_ndvi[msk] - ref_ndvi[msk]).mean()) if msk.any() else 0.0
        if mae < best_mae:
            best_mae = mae
            best_idx = i

    return candidates[best_idx], best_idx


def composite_median(images: list[np.ndarray]) -> np.ndarray:
    """
    Pixel-wise median composite of multiple registered images.

    Args:
        images: List of [H, W, 3] float32 arrays.

    Returns:
        [H, W, 3] median composite.
    """
    stack = np.stack(images, axis=0)  # [N, H, W, 3]
    return np.median(stack, axis=0).astype(np.float32)
