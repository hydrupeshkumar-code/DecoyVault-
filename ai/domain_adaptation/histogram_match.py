"""
Histogram matching for domain adaptation.

Matches the pixel-value distribution of a source image/tensor to a target
reference, independently per channel.  Uses the classical CDF-based look-up
approach (Gonzalez & Woods, Digital Image Processing).
"""

from __future__ import annotations

import numpy as np
import torch


def histogram_match_numpy(
    source: np.ndarray,
    reference: np.ndarray,
    n_bins: int = 256,
) -> np.ndarray:
    """
    Match the histogram of *source* to *reference* channel-by-channel.

    Args:
        source:    [H, W, C] or [C, H, W] float32 array in [0, 1].
        reference: Same shape/layout as source.
        n_bins:    Number of histogram bins.

    Returns:
        Histogram-matched array with same shape as source, in [0, 1].
    """
    if source.ndim == 3 and source.shape[0] <= 4 and source.shape[-1] > 4:
        # CHW
        src = source.transpose(1, 2, 0)
        ref = reference.transpose(1, 2, 0)
        chw = True
    else:
        src = source
        ref = reference
        chw = False

    H, W, C = src.shape
    matched = np.empty_like(src)

    for c in range(C):
        s_ch = src[..., c].ravel()
        r_ch = ref[..., c].ravel()

        s_hist, s_edges = np.histogram(s_ch, bins=n_bins, range=(0.0, 1.0))
        r_hist, r_edges = np.histogram(r_ch, bins=n_bins, range=(0.0, 1.0))

        s_cdf = np.cumsum(s_hist).astype(np.float64)
        r_cdf = np.cumsum(r_hist).astype(np.float64)
        s_cdf /= s_cdf[-1]
        r_cdf /= r_cdf[-1]

        # Build look-up table: for each source bin, find nearest reference bin
        lut = np.interp(s_cdf, r_cdf, r_edges[:-1])

        # Map source pixel values through LUT
        bin_idx = np.digitize(s_ch, s_edges[:-1]) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        matched_ch = lut[bin_idx].reshape(H, W).astype(np.float32)
        matched[..., c] = matched_ch

    return matched.transpose(2, 0, 1) if chw else matched


def histogram_match_tensor(
    source: torch.Tensor,
    reference: torch.Tensor,
    n_bins: int = 256,
) -> torch.Tensor:
    """
    Tensor wrapper around histogram_match_numpy.

    Args:
        source:    [C, H, W] or [B, C, H, W] float tensor in [0, 1].
        reference: Same shape as source.
        n_bins:    Number of histogram bins.

    Returns:
        Histogram-matched tensor with same shape/dtype/device as source.
    """
    device = source.device
    dtype = source.dtype
    squeezed = False

    if source.dim() == 4:
        # Process each sample in batch independently
        results = [
            histogram_match_tensor(source[i], reference[i], n_bins)
            for i in range(source.shape[0])
        ]
        return torch.stack(results, dim=0)

    src_np = source.cpu().float().numpy()
    ref_np = reference.cpu().float().numpy()

    matched_np = histogram_match_numpy(src_np, ref_np, n_bins)
    return torch.from_numpy(matched_np).to(device=device, dtype=dtype)
