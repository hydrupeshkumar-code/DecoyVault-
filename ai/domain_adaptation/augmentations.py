"""
Augmentation pipeline that simulates LISS-IV spatial and radiometric characteristics.

LISS-IV (Resourcesat-2) operates at 5.8 m GSD.  When training on 10 m
Sentinel-2 imagery, we simulate the finer spatial detail through a combination
of:

    - Mild sharpening (unsharp mask)
    - Gaussian noise calibrated to LISS-IV SNR (~600)
    - Radiometric jitter (per-channel gain/bias offsets)
    - Random affine perturbations

All transforms operate on [C, H, W] torch tensors in [0, 1].
"""

from __future__ import annotations

import random
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual augmentations
# ---------------------------------------------------------------------------

def random_flip(
    cloudy: torch.Tensor,
    clear: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random horizontal and/or vertical flip applied consistently."""
    if random.random() > 0.5:
        cloudy = torch.flip(cloudy, dims=[-1])
        clear = torch.flip(clear, dims=[-1])
        mask = torch.flip(mask, dims=[-1])
    if random.random() > 0.5:
        cloudy = torch.flip(cloudy, dims=[-2])
        clear = torch.flip(clear, dims=[-2])
        mask = torch.flip(mask, dims=[-2])
    return cloudy, clear, mask


def random_rotation90(
    cloudy: torch.Tensor,
    clear: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random 90° rotation (k ∈ {0, 1, 2, 3})."""
    k = random.randint(0, 3)
    cloudy = torch.rot90(cloudy, k, dims=[-2, -1])
    clear = torch.rot90(clear, k, dims=[-2, -1])
    mask = torch.rot90(mask, k, dims=[-2, -1])
    return cloudy, clear, mask


def add_gaussian_noise(
    tensor: torch.Tensor,
    std: float = 0.005,
) -> torch.Tensor:
    """Add zero-mean Gaussian noise simulating LISS-IV sensor noise."""
    noise = torch.randn_like(tensor) * std
    return (tensor + noise).clamp(0.0, 1.0)


def radiometric_jitter(
    tensor: torch.Tensor,
    gain_range: tuple[float, float] = (0.95, 1.05),
    bias_range: tuple[float, float] = (-0.02, 0.02),
) -> torch.Tensor:
    """
    Per-channel random gain and bias to simulate inter-band calibration variance.

    Args:
        tensor:     [C, H, W] tensor.
        gain_range: (min, max) multiplicative gain per channel.
        bias_range: (min, max) additive bias per channel.
    """
    C = tensor.shape[0]
    gain = torch.FloatTensor(C).uniform_(*gain_range).to(tensor.device)
    bias = torch.FloatTensor(C).uniform_(*bias_range).to(tensor.device)
    return (tensor * gain[:, None, None] + bias[:, None, None]).clamp(0.0, 1.0)


def unsharp_mask_sharpen(
    tensor: torch.Tensor,
    strength: float = 0.3,
    kernel_size: int = 3,
) -> torch.Tensor:
    """
    Simulate the slightly higher spatial sharpness of LISS-IV at 5.8 m vs.
    Sentinel-2 at 10 m using an unsharp mask.

    Args:
        tensor:      [C, H, W] tensor.
        strength:    Sharpening coefficient (0 = no effect, 1 = strong).
        kernel_size: Blur kernel size.
    """
    C = tensor.shape[0]
    pad = kernel_size // 2
    # Box blur
    weight = torch.ones(C, 1, kernel_size, kernel_size, device=tensor.device) / (kernel_size ** 2)
    blurred = F.conv2d(tensor.unsqueeze(0), weight, padding=pad, groups=C).squeeze(0)
    sharpened = tensor + strength * (tensor - blurred)
    return sharpened.clamp(0.0, 1.0)


def random_cloud_texture_mix(
    cloudy: torch.Tensor,
    mask: torch.Tensor,
    mix_alpha_range: tuple[float, float] = (0.05, 0.2),
) -> torch.Tensor:
    """
    Slightly randomise cloud brightness to simulate varying cloud optical depths.

    Args:
        cloudy:          [C, H, W] cloudy image.
        mask:            [1, H, W] binary cloud mask.
        mix_alpha_range: Range for blending with a white cloud (value = 1.0).
    """
    alpha = random.uniform(*mix_alpha_range)
    cloud_whitened = cloudy.clone()
    cloud_whitened = cloud_whitened * (1 - mask * alpha) + mask * alpha
    return cloud_whitened.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Composed augmentation pipeline
# ---------------------------------------------------------------------------

class LISS4AugmentationPipeline:
    """
    Augmentation pipeline calibrated for SEN2→LISS-IV domain simulation.

    Applies augmentations consistently across the (cloudy, clear, mask) triplet.

    Args:
        noise_std:          Gaussian noise standard deviation.
        gain_range:         Radiometric gain perturbation range.
        bias_range:         Radiometric bias perturbation range.
        sharpen_strength:   Unsharp mask strength.
        cloud_mix:          Apply random cloud texture variation.
        p_noise:            Probability of applying noise.
        p_radiometric:      Probability of radiometric jitter.
        p_sharpen:          Probability of sharpening.
    """

    def __init__(
        self,
        noise_std: float = 0.005,
        gain_range: tuple[float, float] = (0.95, 1.05),
        bias_range: tuple[float, float] = (-0.02, 0.02),
        sharpen_strength: float = 0.2,
        cloud_mix: bool = True,
        p_noise: float = 0.5,
        p_radiometric: float = 0.5,
        p_sharpen: float = 0.4,
    ) -> None:
        self.noise_std = noise_std
        self.gain_range = gain_range
        self.bias_range = bias_range
        self.sharpen_strength = sharpen_strength
        self.cloud_mix = cloud_mix
        self.p_noise = p_noise
        self.p_radiometric = p_radiometric
        self.p_sharpen = p_sharpen

    def __call__(
        self,
        cloudy: torch.Tensor,
        clear: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply the full augmentation pipeline.

        Args:
            cloudy: [C, H, W] or [B, C, H, W].
            clear:  same shape as cloudy.
            mask:   [1, H, W] or [B, 1, H, W].

        Returns:
            Augmented (cloudy, clear, mask) tuple.
        """
        cloudy, clear, mask = random_flip(cloudy, clear, mask)
        cloudy, clear, mask = random_rotation90(cloudy, clear, mask)

        if random.random() < self.p_noise:
            cloudy = add_gaussian_noise(cloudy, self.noise_std)

        if random.random() < self.p_radiometric:
            cloudy = radiometric_jitter(cloudy, self.gain_range, self.bias_range)
            clear = radiometric_jitter(clear, self.gain_range, self.bias_range)

        if random.random() < self.p_sharpen:
            cloudy = unsharp_mask_sharpen(cloudy, self.sharpen_strength)
            clear = unsharp_mask_sharpen(clear, self.sharpen_strength)

        if self.cloud_mix:
            cloudy = random_cloud_texture_mix(cloudy, mask)

        return cloudy, clear, mask
