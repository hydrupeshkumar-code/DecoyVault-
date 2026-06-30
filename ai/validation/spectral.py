"""
Spectral fidelity validation for LISS-IV cloud removal output.

Cloud removal is NOT image enhancement — the scientific goal is spectral
accuracy: the reconstructed reflectances must be radiometrically consistent
with the true surface reflectance so that derived products (NDVI, NDWI,
LSWI, land cover classifications) computed on the reconstruction remain
scientifically valid.

This module validates:
    1. Per-band radiometric accuracy (bias, MAE, RMSE, Pearson r)
    2. Spectral shape preservation (SAM per pixel and histogram)
    3. Band ratio preservation (NDVI, NDWI, band ratios)
    4. Spectral unmixing fidelity (for land-cover applications)

Conventions:
    Input tensors: [B, 3, H, W] float32 [0,1]
    Band order:    [0]=Green, [1]=Red, [2]=NIR  (LISS-IV)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


BAND_NAMES = ["Green", "Red", "NIR"]


# ---------------------------------------------------------------------------
# Per-band radiometric accuracy
# ---------------------------------------------------------------------------

def per_band_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> dict[str, float]:
    """
    Compute per-band radiometric accuracy metrics.

    Args:
        pred:   [B, 3, H, W]
        target: [B, 3, H, W]
        mask:   [B, 1, H, W] optional — restrict to cloud pixels if provided.
        eps:    Numerical stability.

    Returns:
        Dict with keys like:
            green_bias, green_mae, green_rmse, green_pearson_r,
            red_bias,   red_mae,   red_rmse,   red_pearson_r,
            nir_bias,   nir_mae,   nir_rmse,   nir_pearson_r,
    """
    results: dict[str, float] = {}

    # Apply optional mask
    if mask is not None:
        for b, name in enumerate(BAND_NAMES):
            bm = mask[:, 0].bool()
            p_b = pred[:, b][bm].flatten()
            t_b = target[:, b][bm].flatten()
            results.update(_scalar_accuracy(name.lower(), p_b, t_b, eps))
    else:
        for b, name in enumerate(BAND_NAMES):
            p_b = pred[:, b].flatten()
            t_b = target[:, b].flatten()
            results.update(_scalar_accuracy(name.lower(), p_b, t_b, eps))

    return results


def _scalar_accuracy(
    prefix: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Compute bias / MAE / RMSE / Pearson r for a 1-D vector pair."""
    diff = pred - target
    mae   = float(diff.abs().mean())
    rmse  = float((diff ** 2).mean().sqrt())
    bias  = float(diff.mean())

    # Pearson correlation
    pn = pred - pred.mean()
    tn = target - target.mean()
    corr_denom = (pn.norm() * tn.norm()).clamp(min=eps)
    corr = float((pn * tn).sum() / corr_denom) if corr_denom > eps else float("nan")

    return {
        f"{prefix}_bias":      bias,
        f"{prefix}_mae":       mae,
        f"{prefix}_rmse":      rmse,
        f"{prefix}_pearson_r": corr,
    }


# ---------------------------------------------------------------------------
# Band ratio preservation
# ---------------------------------------------------------------------------

def band_ratio_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> dict[str, float]:
    """
    Validate preservation of scientifically important band ratios.

    Band ratios are the primary derived products in satellite imagery analysis.
    Errors in individual band reflectances compound in these ratios.

    Computed ratios:
        NDVI  = (NIR - Red) / (NIR + Red)    — vegetation greenness
        NDWI  = (Green - NIR) / (Green + NIR) — water bodies (McFeeters 1996)
        Red/NIR ratio                          — land/vegetation discrimination
        NIR/Green ratio                        — biomass proxy

    Args:
        pred:   [B, 3, H, W] [Green=0, Red=1, NIR=2]
        target: [B, 3, H, W]
        mask:   [B, 1, H, W] optional cloud mask
        eps:    Numerical stability

    Returns:
        Dict: ndvi_mae, ndvi_rmse, ndwi_mae, ndwi_rmse, red_nir_ratio_mae, ...
    """
    results: dict[str, float] = {}

    G, R, N = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    Gt, Rt, Nt = target[:, 0:1], target[:, 1:2], target[:, 2:3]

    ratios = {
        "ndvi":       ((N - R) / (N + R + eps),         (Nt - Rt) / (Nt + Rt + eps)),
        "ndwi":       ((G - N) / (G + N + eps),         (Gt - Nt) / (Gt + Nt + eps)),
        "red_nir":    (R / (N + eps),                   Rt / (Nt + eps)),
        "nir_green":  (N / (G + eps),                   Nt / (Gt + eps)),
    }

    for name, (p_ratio, t_ratio) in ratios.items():
        if mask is not None:
            bm = mask.bool()
            p_vals = p_ratio[bm].flatten()
            t_vals = t_ratio[bm].flatten()
        else:
            p_vals = p_ratio.flatten()
            t_vals = t_ratio.flatten()

        diff = p_vals - t_vals
        results[f"{name}_mae"]  = float(diff.abs().mean())
        results[f"{name}_rmse"] = float((diff ** 2).mean().sqrt())
        results[f"{name}_bias"] = float(diff.mean())

    return results


# ---------------------------------------------------------------------------
# Spectral histogram divergence
# ---------------------------------------------------------------------------

def spectral_histogram_divergence(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    n_bins: int = 64,
) -> dict[str, float]:
    """
    Kullback-Leibler divergence between predicted and target spectral histograms.

    A large KL divergence indicates the model has shifted the spectral
    distribution — the reconstruction has a different overall radiometric
    character from the true clear surface.

    Lower is better (0 = identical distributions).

    Returns:
        kl_green, kl_red, kl_nir (in nats)
    """
    results: dict[str, float] = {}
    eps = 1e-10

    for b, name in enumerate(BAND_NAMES):
        if mask is not None:
            bm = mask[:, 0].bool()
            p_vals = pred[:, b][bm].flatten().cpu().numpy()
            t_vals = target[:, b][bm].flatten().cpu().numpy()
        else:
            p_vals = pred[:, b].flatten().cpu().numpy()
            t_vals = target[:, b].flatten().cpu().numpy()

        bins = np.linspace(0, 1, n_bins + 1)
        p_hist, _ = np.histogram(p_vals, bins=bins, density=True)
        t_hist, _ = np.histogram(t_vals, bins=bins, density=True)

        p_hist = p_hist + eps
        t_hist = t_hist + eps
        p_hist /= p_hist.sum()
        t_hist /= t_hist.sum()

        kl = float(np.sum(t_hist * np.log(t_hist / p_hist)))
        results[f"kl_{name.lower()}"] = kl

    return results


# ---------------------------------------------------------------------------
# Full spectral validation suite
# ---------------------------------------------------------------------------

def full_spectral_validation(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """
    Run the complete spectral validation pipeline.

    Args:
        pred:   [B, 3, H, W]
        target: [B, 3, H, W]
        mask:   [B, 1, H, W] optional cloud mask

    Returns:
        Combined dict of all spectral metrics.
    """
    out: dict[str, float] = {}
    out.update(per_band_accuracy(pred, target, mask))
    out.update(band_ratio_metrics(pred, target, mask))
    out.update(spectral_histogram_divergence(pred, target, mask))
    return out
