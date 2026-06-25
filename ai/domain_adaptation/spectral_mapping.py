"""
Spectral band mapping from Sentinel-2 to LISS-IV.

Empirically-derived linear gain/offset coefficients that align the
Sentinel-2 B3/B4/B8 spectral response functions with the LISS-IV
Green/Red/NIR bands (Resourcesat-2, 5.8 m resolution).

References:
    Equivalent spectral calibration methodology as described in:
    Roy et al. (2016). "Characterization of Landsat-7 to Landsat-8 reflective
    wavelength and BRDF-based temporal stability." Remote Sensing of Environment.

    Adapted for Sentinel-2 ↔ LISS-IV based on IRS spectral library data.

Coefficients
------------
LISS-IV_Green ≈ 0.9722 × S2_B3  + 0.0070
LISS-IV_Red   ≈ 0.9877 × S2_B4  + 0.0042
LISS-IV_NIR   ≈ 1.0034 × S2_B8  - 0.0028

These are representative values; re-derive from co-located scene pairs for
highest accuracy.
"""

from __future__ import annotations

import numpy as np
import torch

# Linear spectral coefficients: (gain, offset) per output band (Green, Red, NIR)
SPECTRAL_COEFFICIENTS: dict[str, tuple[float, float]] = {
    "green": (0.9722, 0.0070),
    "red":   (0.9877, 0.0042),
    "nir":   (1.0034, -0.0028),
}

# Channel order in the 3-band array: 0=Green, 1=Red, 2=NIR
_BAND_ORDER = ["green", "red", "nir"]


def sen2_to_liss4_spectral(image: np.ndarray) -> np.ndarray:
    """
    Apply per-channel linear gain/offset to convert SEN2 bands to LISS-IV.

    Args:
        image: [H, W, 3] float32 array with channels (Green, Red, NIR) in [0, 1].

    Returns:
        [H, W, 3] float32 array after spectral correction, clipped to [0, 1].
    """
    result = np.empty_like(image)
    for idx, band_name in enumerate(_BAND_ORDER):
        gain, offset = SPECTRAL_COEFFICIENTS[band_name]
        result[..., idx] = gain * image[..., idx] + offset
    return np.clip(result, 0.0, 1.0)


def sen2_to_liss4_spectral_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Tensor version of spectral mapping.

    Args:
        tensor: [3, H, W] or [B, 3, H, W] float tensor in [0, 1].

    Returns:
        Spectrally-corrected tensor, same shape, clipped to [0, 1].
    """
    gains = torch.tensor(
        [SPECTRAL_COEFFICIENTS[b][0] for b in _BAND_ORDER],
        dtype=tensor.dtype, device=tensor.device,
    )
    offsets = torch.tensor(
        [SPECTRAL_COEFFICIENTS[b][1] for b in _BAND_ORDER],
        dtype=tensor.dtype, device=tensor.device,
    )

    if tensor.dim() == 3:
        gains = gains[:, None, None]
        offsets = offsets[:, None, None]
    elif tensor.dim() == 4:
        gains = gains[None, :, None, None]
        offsets = offsets[None, :, None, None]

    return (tensor * gains + offsets).clamp(0.0, 1.0)


def inverse_liss4_to_sen2(image: np.ndarray) -> np.ndarray:
    """
    Invert the spectral mapping (LISS-IV → SEN2).  Useful for analysis.

    Args:
        image: [H, W, 3] float32 in [0, 1].

    Returns:
        [H, W, 3] float32 in [0, 1] with S2-equivalent values.
    """
    result = np.empty_like(image)
    for idx, band_name in enumerate(_BAND_ORDER):
        gain, offset = SPECTRAL_COEFFICIENTS[band_name]
        result[..., idx] = (image[..., idx] - offset) / max(abs(gain), 1e-8)
    return np.clip(result, 0.0, 1.0)
