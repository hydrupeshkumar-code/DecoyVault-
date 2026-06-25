"""
Synthetic cloud mask generator for data augmentation.

Generates procedural cloud-like binary masks using Perlin-noise-inspired
random blob patterns. Useful when real annotated masks are scarce.
"""

from __future__ import annotations

import random

import numpy as np


def _smooth_random_field(H: int, W: int, scale: int) -> np.ndarray:
    """Generate a smoothed random field by upsampling a coarse grid."""
    coarse_h = max(H // scale, 4)
    coarse_w = max(W // scale, 4)
    coarse = np.random.rand(coarse_h, coarse_w).astype(np.float32)
    # Bilinear upsample via numpy repeat + interpolate
    from PIL import Image  # lazy import – optional dependency
    img = Image.fromarray((coarse * 255).astype(np.uint8))
    img = img.resize((W, H), resample=Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def generate_synthetic_mask(
    H: int,
    W: int,
    cloud_fraction_range: tuple[float, float] = (0.1, 0.6),
    n_scales: int = 3,
) -> np.ndarray:
    """
    Generate a synthetic cloud mask.

    Args:
        H, W:                  Spatial dimensions.
        cloud_fraction_range:  Desired cloud cover fraction (min, max).
        n_scales:              Number of frequency octaves to blend.

    Returns:
        [H, W] binary float32 mask (1 = cloud).
    """
    field = np.zeros((H, W), dtype=np.float32)
    for i in range(n_scales):
        scale = 2 ** (i + 2)
        amplitude = 1.0 / (i + 1)
        field += amplitude * _smooth_random_field(H, W, scale)
    field /= field.max() + 1e-8

    # Threshold to achieve desired cloud fraction
    target_frac = random.uniform(*cloud_fraction_range)
    threshold = np.percentile(field, (1.0 - target_frac) * 100)
    mask = (field >= threshold).astype(np.float32)
    return mask


def augment_with_synthetic_cloud(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    cloud_brightness: float = 0.85,
    cloud_fraction_range: tuple[float, float] = (0.1, 0.5),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Overlay a synthetic cloud on a clear image.

    Args:
        image:               [H, W, 3] clear image in [0, 1].
        mask:                Existing mask to merge with (or None).
        cloud_brightness:    Mean brightness of the synthetic cloud patch.
        cloud_fraction_range: Desired cloud cover.

    Returns:
        (cloudy_image, merged_mask) both as float32 arrays.
    """
    H, W = image.shape[:2]
    syn_mask = generate_synthetic_mask(H, W, cloud_fraction_range)
    cloud_layer = np.ones_like(image) * cloud_brightness
    cloudy = image * (1 - syn_mask[..., None]) + cloud_layer * syn_mask[..., None]
    cloudy = np.clip(cloudy, 0.0, 1.0).astype(np.float32)

    if mask is not None:
        merged_mask = np.clip(syn_mask + mask, 0.0, 1.0)
    else:
        merged_mask = syn_mask

    return cloudy, merged_mask
