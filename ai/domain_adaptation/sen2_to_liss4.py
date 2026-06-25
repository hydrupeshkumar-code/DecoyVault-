"""
SEN2-CR → LISS-IV domain adaptation pipeline.

Sentinel-2 bands to LISS-IV channels:
    B3  (Green, ~560 nm)  → LISS-IV Green
    B4  (Red,   ~665 nm)  → LISS-IV Red
    B8  (NIR,   ~842 nm)  → LISS-IV NIR

Spatial resolution simulation:
    Sentinel-2 native resolution: 10 m
    LISS-IV target resolution:    5.8 m
    Upscale factor ≈ 1.72  (super-resolution simulation via augmentation)
"""

from __future__ import annotations

import numpy as np
import torch

from ai.domain_adaptation.histogram_match import histogram_match_tensor
from ai.domain_adaptation.spectral_mapping import sen2_to_liss4_spectral


# SEN2-CR band ordering (0-indexed):
#   0=B1, 1=B2, 2=B3, 3=B4, 4=B5, 5=B6, 6=B7, 7=B8, 8=B8A, 9=B9, 10=B10, 11=B11, 12=B12
SEN2_GREEN_IDX: int = 2
SEN2_RED_IDX: int = 3
SEN2_NIR_IDX: int = 7

# LISS-IV dynamic range (typical TOA reflectance, 0–1 normalised)
LISS4_NORM_MIN: float = 0.0
LISS4_NORM_MAX: float = 1.0


def sen2_to_liss4(
    sen2_image: np.ndarray | torch.Tensor,
    reference_liss4: np.ndarray | torch.Tensor | None = None,
    apply_histogram_match: bool = True,
    apply_spectral_map: bool = True,
) -> torch.Tensor:
    """
    Convert a Sentinel-2 multi-band image to a LISS-IV compatible 3-band tensor.

    Steps:
        1. Select B3, B4, B8 from the 13-band SEN2-CR array.
        2. Apply per-channel spectral gain/offset mapping.
        3. Optionally match histograms to a LISS-IV reference image.
        4. Normalise to [0, 1].

    Args:
        sen2_image:             [H, W, 13] or [13, H, W] SEN2-CR image.
        reference_liss4:        Optional LISS-IV reference for histogram matching.
                                Shape [H, W, 3] or [3, H, W] or None.
        apply_histogram_match:  Match histogram distribution of output to LISS-IV reference.
        apply_spectral_map:     Apply linear spectral gain/offset correction.

    Returns:
        [3, H, W] float32 tensor in [0, 1] representing (Green, Red, NIR).
    """
    if isinstance(sen2_image, torch.Tensor):
        arr = sen2_image.cpu().numpy()
    else:
        arr = np.array(sen2_image, dtype=np.float32)

    # Normalise CHW / HWC
    if arr.ndim == 3 and arr.shape[0] in (3, 13):
        arr = arr.transpose(1, 2, 0)  # CHW → HWC

    # Select Green / Red / NIR
    if arr.shape[-1] >= 13:
        bands = arr[..., [SEN2_GREEN_IDX, SEN2_RED_IDX, SEN2_NIR_IDX]]
    elif arr.shape[-1] == 3:
        bands = arr  # already 3-band
    else:
        raise ValueError(
            f"Expected 3 or 13 bands, got {arr.shape[-1]}."
        )

    bands = bands.astype(np.float32)

    # Clip to valid reflectance range and normalise
    bands = np.clip(bands, LISS4_NORM_MIN, LISS4_NORM_MAX)

    # Spectral gain/offset mapping
    if apply_spectral_map:
        bands = sen2_to_liss4_spectral(bands)

    # Convert to torch
    tensor = torch.from_numpy(bands.transpose(2, 0, 1))  # [3, H, W]

    # Histogram matching
    if apply_histogram_match and reference_liss4 is not None:
        if isinstance(reference_liss4, np.ndarray):
            ref = torch.from_numpy(
                reference_liss4.transpose(2, 0, 1)
                if reference_liss4.ndim == 3 and reference_liss4.shape[-1] == 3
                else reference_liss4
            ).float()
        else:
            ref = reference_liss4.float()
        tensor = histogram_match_tensor(tensor, ref)

    return tensor.float().clamp(0.0, 1.0)


def batch_sen2_to_liss4(
    sen2_batch: torch.Tensor,
    reference_liss4: torch.Tensor | None = None,
    apply_histogram_match: bool = True,
    apply_spectral_map: bool = True,
) -> torch.Tensor:
    """
    Batch-level SEN2-CR → LISS-IV conversion.

    Args:
        sen2_batch:             [B, 13, H, W] or [B, 3, H, W] batch tensor.
        reference_liss4:        Optional [B, 3, H, W] reference batch.
        apply_histogram_match:  Apply histogram matching per sample.
        apply_spectral_map:     Apply spectral correction.

    Returns:
        [B, 3, H, W] tensor in [0, 1].
    """
    results: list[torch.Tensor] = []
    B = sen2_batch.shape[0]

    for i in range(B):
        ref = reference_liss4[i] if reference_liss4 is not None else None
        adapted = sen2_to_liss4(
            sen2_batch[i],
            reference_liss4=ref,
            apply_histogram_match=apply_histogram_match,
            apply_spectral_map=apply_spectral_map,
        )
        results.append(adapted)

    return torch.stack(results, dim=0)
