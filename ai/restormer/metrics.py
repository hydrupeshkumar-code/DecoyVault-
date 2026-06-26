"""
Per-image and batch metrics for cloud removal evaluation.

Implemented:
    PSNR, SSIM, SAM, RMSE, NDVI-MAE

All functions accept PyTorch tensors [B, C, H, W] ∈ [0, 1].
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from ai.restormer.ndvi import compute_ndvi_mae


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio (dB), averaged over the batch."""
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])  # [B]
    return (10.0 * torch.log10(max_val ** 2 / mse.clamp(min=1e-10))).mean()


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def _ssim_single(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Full-image SSIM scalar for [B, C, H, W] inputs."""
    C = pred.shape[1]

    # Build separable Gaussian window
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    kernel_2d = kernel_2d.expand(C, 1, window_size, window_size)

    pad = window_size // 2
    mu1 = F.conv2d(pred, kernel_2d, padding=pad, groups=C)
    mu2 = F.conv2d(target, kernel_2d, padding=pad, groups=C)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Clamp variances non-negative: the conv-based E[x²]-E[x]² estimator can
    # yield tiny negative values from rounding (especially under bfloat16).
    # A negative variance makes SSIM > 1 and breaks the reported metric.
    sigma1_sq = (F.conv2d(pred * pred, kernel_2d, padding=pad, groups=C) - mu1_sq).clamp(min=0.0)
    sigma2_sq = (F.conv2d(target * target, kernel_2d, padding=pad, groups=C) - mu2_sq).clamp(min=0.0)
    sigma12 = F.conv2d(pred * target, kernel_2d, padding=pad, groups=C) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.clamp(0.0, 1.0).mean()


def ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """SSIM averaged over the batch."""
    return _ssim_single(pred, target)


# ---------------------------------------------------------------------------
# SAM
# ---------------------------------------------------------------------------

def sam(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Spectral Angle Mapper (radians), averaged over pixels and batch.

    Lower is better.
    """
    dot = (pred * target).sum(dim=1)
    # sqrt(sum+eps) instead of norm().clamp(): the latter has a 0/0 NaN
    # *gradient* at all-zero pixels (black borders, nodata). Under no_grad
    # this only produces NaN metric values; the eps-inside-sqrt makes both
    # the forward value and its gradient finite everywhere.
    norm_p = torch.sqrt((pred * pred).sum(dim=1) + eps)
    norm_t = torch.sqrt((target * target).sum(dim=1) + eps)
    cos = (dot / (norm_p * norm_t)).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos).mean()


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------

def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root-Mean-Square Error, averaged over the batch."""
    return torch.sqrt(F.mse_loss(pred, target))


# ---------------------------------------------------------------------------
# Batch metrics runner
# ---------------------------------------------------------------------------

def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    cloudy: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> Dict[str, float]:
    """
    Compute all metrics for a batch.

    Args:
        pred:   [B, 3, H, W] model predictions.
        target: [B, 3, H, W] ground-truth clear images.
        cloudy: Optional [B, 3, H, W] input (used for NDVI baseline).
        mask:   Optional [B, 1, H, W] cloud mask for NDVI-MAE.

    Returns:
        Dictionary of metric_name → float value.
    """
    with torch.no_grad():
        results: Dict[str, float] = {
            "psnr": psnr(pred, target).item(),
            "ssim": ssim(pred, target).item(),
            "sam_rad": sam(pred, target).item(),
            "rmse": rmse(pred, target).item(),
            "ndvi_mae_global": compute_ndvi_mae(pred, target, mask=None).item(),
        }
        if mask is not None:
            results["ndvi_mae_cloud"] = compute_ndvi_mae(pred, target, mask=mask).item()
        if cloudy is not None and mask is not None:
            baseline_ndvi_mae = compute_ndvi_mae(cloudy, target, mask=mask).item()
            results["ndvi_improvement"] = baseline_ndvi_mae - results["ndvi_mae_cloud"]

    return results


def aggregate_metrics(
    metric_list: Sequence[Dict[str, float]],
) -> Dict[str, float]:
    """
    Average a sequence of per-batch metric dicts.

    Args:
        metric_list: List of dicts from ``compute_metrics``.

    Returns:
        Aggregated dict with mean values.
    """
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    return {k: sum(d[k] for d in metric_list) / len(metric_list) for k in keys}


def save_report(metrics: Dict[str, float], output_path: str | Path) -> None:
    """
    Save a metrics dict to a JSON file.

    Args:
        metrics:     Dictionary of metric_name → float.
        output_path: Destination .json path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
