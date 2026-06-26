"""
Cloud-region and vegetation-region scientific metrics.

Standard metrics (PSNR, SSIM, SAM, RMSE) computed on the FULL image are
insufficient for cloud removal evaluation. A model can achieve high global
PSNR by perfectly preserving cloud-free regions while failing to reconstruct
the actual cloud-covered content.

This module provides:
    1. Cloud-region PSNR / SSIM / SAM / RMSE
       — Evaluated only within cloud-masked pixels (the actual task)
    2. Clear-region consistency PSNR
       — The model should NOT modify clear pixels (quality preservation check)
    3. Vegetation-region NDVI fidelity
       — Critical for agriculture, forestry, and land-cover applications
    4. Edge preservation score
       — Structural continuity at cloud boundaries

Reference:
    Meraner et al. (2020) "Cloud removal in Sentinel-2 imagery using a deep
    residual neural network and SAR-optical data fusion", ISPRS J.

    Ebel et al. (2020) "Cloud removal in satellite image time series through
    multiscale bottleneck convolutional neural networks", Remote Sensing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def cloud_region_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, float]:
    """
    Compute all metrics restricted to cloud-covered pixels.

    Args:
        pred:   [B, 3, H, W] model output in [0,1].
        target: [B, 3, H, W] ground-truth clear image in [0,1].
        mask:   [B, 1, H, W] binary cloud mask (1 = cloud).
        eps:    Small value for numerical stability.

    Returns:
        Dictionary:
            cloud_psnr_db       — PSNR within cloud region
            cloud_ssim          — SSIM within cloud region (approx.)
            cloud_sam_rad       — SAM within cloud region (radians)
            cloud_sam_deg       — SAM within cloud region (degrees)
            cloud_rmse          — RMSE within cloud region
            cloud_pixel_count   — Number of cloud pixels evaluated
    """
    cloud_bool = mask.bool().expand_as(pred)   # [B, 3, H, W]

    if cloud_bool.sum() == 0:
        return {
            "cloud_psnr_db": float("nan"),
            "cloud_ssim": float("nan"),
            "cloud_sam_rad": float("nan"),
            "cloud_sam_deg": float("nan"),
            "cloud_rmse": float("nan"),
            "cloud_pixel_count": 0,
        }

    p_cloud  = pred[cloud_bool]
    t_cloud  = target[cloud_bool]

    # MSE / RMSE / PSNR
    mse = ((p_cloud - t_cloud) ** 2).mean()
    rmse_val = float(mse.sqrt())
    psnr_val = float(-10.0 * torch.log10(mse + eps))

    # Cloud-region SSIM (approximate — pixel-pair comparison)
    # Full sliding-window SSIM in masked region requires spatial logic;
    # we use the simplified mean-luminance SSIM approximation here.
    mu_p = p_cloud.mean()
    mu_t = t_cloud.mean()
    var_p = ((p_cloud - mu_p) ** 2).mean()
    var_t = ((t_cloud - mu_t) ** 2).mean()
    cov   = ((p_cloud - mu_p) * (t_cloud - mu_t)).mean()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_val = float(
        ((2 * mu_p * mu_t + C1) * (2 * cov + C2))
        / ((mu_p ** 2 + mu_t ** 2 + C1) * (var_p + var_t + C2))
    )

    # SAM in cloud region — reshape to [N, C] for per-pixel angle
    B, C, H, W = pred.shape
    m2d = mask[:, 0].bool()  # [B, H, W]
    p_px  = pred.permute(0, 2, 3, 1)[m2d]   # [N, C]
    t_px  = target.permute(0, 2, 3, 1)[m2d]  # [N, C]

    dot      = (p_px * t_px).sum(dim=1)
    norm_p   = torch.sqrt((p_px * p_px).sum(dim=1) + eps)
    norm_t   = torch.sqrt((t_px * t_px).sum(dim=1) + eps)
    cos_a    = (dot / (norm_p * norm_t)).clamp(-1 + eps, 1 - eps)
    sam_rad  = float(torch.acos(cos_a).mean())

    n_pixels = int(m2d.sum().item())

    return {
        "cloud_psnr_db":     psnr_val,
        "cloud_ssim":        ssim_val,
        "cloud_sam_rad":     sam_rad,
        "cloud_sam_deg":     float(np.degrees(sam_rad)),
        "cloud_rmse":        rmse_val,
        "cloud_pixel_count": n_pixels,
    }


def clear_region_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, float]:
    """
    Measure how well the model preserves clear (non-cloud) pixels.

    A cloud removal model should ideally be an identity function on clear
    pixels. This metric measures degradation introduced to cloud-free areas.

    Args:
        pred:   [B, 3, H, W]
        target: [B, 3, H, W]
        mask:   [B, 1, H, W] cloud mask (1 = cloud → we evaluate 0 regions)

    Returns:
        clear_psnr_db     — should be > 40 dB for a well-behaved model
        clear_rmse
        clear_pixel_count
    """
    clear_bool = (1.0 - mask).bool().expand_as(pred)

    if clear_bool.sum() == 0:
        return {"clear_psnr_db": float("nan"), "clear_rmse": float("nan"), "clear_pixel_count": 0}

    p_clear = pred[clear_bool]
    t_clear = target[clear_bool]

    mse = ((p_clear - t_clear) ** 2).mean()
    return {
        "clear_psnr_db":     float(-10.0 * torch.log10(mse + eps)),
        "clear_rmse":        float(mse.sqrt()),
        "clear_pixel_count": int(clear_bool.sum().item()),
    }


def vegetation_region_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    ndvi_threshold: float = 0.2,
    eps: float = 1e-8,
) -> dict[str, float]:
    """
    NDVI fidelity metrics in vegetated areas within cloud regions.

    Critical for agriculture and environmental monitoring applications.
    LISS-IV's NIR band makes this the primary land-cover discriminator.

    Args:
        pred:            [B, 3, H, W] with bands [Green, Red, NIR].
        target:          [B, 3, H, W]
        mask:            [B, 1, H, W] cloud mask.
        ndvi_threshold:  Pixels with target NDVI > this are "vegetated".
        eps:             Numerical stability.

    Returns:
        veg_ndvi_mae     — MAE of NDVI in vegetated cloud regions
        veg_ndvi_rmse    — RMSE of NDVI in vegetated cloud regions
        veg_ndvi_bias    — Signed bias (+ = over-estimated, − = under-estimated)
        veg_pixel_count  — Pixels evaluated
        ndvi_correlation — Pearson r between pred and target NDVI
    """
    # NDVI = (NIR - Red) / (NIR + Red) using bands [Green=0, Red=1, NIR=2]
    def ndvi(x: torch.Tensor) -> torch.Tensor:
        red, nir = x[:, 1:2], x[:, 2:3]
        return (nir - red) / (nir + red + eps)

    ndvi_pred   = ndvi(pred)    # [B, 1, H, W]
    ndvi_target = ndvi(target)

    # Vegetation mask: target NDVI > threshold AND within cloud region
    veg_mask = ((ndvi_target > ndvi_threshold) & mask.bool())  # [B, 1, H, W]

    if veg_mask.sum() == 0:
        return {
            "veg_ndvi_mae": float("nan"),
            "veg_ndvi_rmse": float("nan"),
            "veg_ndvi_bias": float("nan"),
            "veg_pixel_count": 0,
            "ndvi_correlation": float("nan"),
        }

    p_ndvi = ndvi_pred[veg_mask]
    t_ndvi = ndvi_target[veg_mask]

    diff = p_ndvi - t_ndvi
    mae  = float(diff.abs().mean())
    rmse = float((diff ** 2).mean().sqrt())
    bias = float(diff.mean())

    # Pearson correlation
    if len(p_ndvi) > 1:
        p_norm = p_ndvi - p_ndvi.mean()
        t_norm = t_ndvi - t_ndvi.mean()
        corr_denom = (p_norm.norm() * t_norm.norm()).clamp(min=eps)
        corr = float((p_norm * t_norm).sum() / corr_denom)
    else:
        corr = float("nan")

    return {
        "veg_ndvi_mae":    mae,
        "veg_ndvi_rmse":   rmse,
        "veg_ndvi_bias":   bias,
        "veg_pixel_count": int(veg_mask.sum().item()),
        "ndvi_correlation": corr,
    }


def edge_preservation_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, float]:
    """
    Evaluate structural continuity at cloud boundaries.

    Poor cloud removal often leaves visible halos or abrupt transitions at
    mask boundaries. This metric uses Sobel gradients to measure edge
    preservation in the boundary zone (±8px from the cloud edge).

    Args:
        pred:   [B, 3, H, W]
        target: [B, 3, H, W]
        mask:   [B, 1, H, W]

    Returns:
        edge_psnr_db     — PSNR in the boundary zone
        boundary_sam_rad — SAM in boundary pixels
    """
    # Dilate mask to get boundary zone
    boundary = _mask_boundary(mask, dilation=8)
    boundary_bool = boundary.bool().expand_as(pred)

    if boundary_bool.sum() == 0:
        return {"edge_psnr_db": float("nan"), "boundary_sam_rad": float("nan")}

    p_b = pred[boundary_bool]
    t_b = target[boundary_bool]

    mse = ((p_b - t_b) ** 2).mean()
    psnr = float(-10.0 * torch.log10(mse + eps))

    # SAM in boundary zone
    B, C, H, W = pred.shape
    bm = boundary[:, 0].bool()
    p_px = pred.permute(0, 2, 3, 1)[bm]
    t_px = target.permute(0, 2, 3, 1)[bm]
    dot   = (p_px * t_px).sum(dim=1)
    n_p   = torch.sqrt((p_px * p_px).sum(dim=1) + eps)
    n_t   = torch.sqrt((t_px * t_px).sum(dim=1) + eps)
    sam   = float(torch.acos((dot / (n_p * n_t)).clamp(-1 + eps, 1 - eps)).mean())

    return {"edge_psnr_db": psnr, "boundary_sam_rad": sam}


def full_scientific_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    cloudy: Optional[torch.Tensor],
    mask: torch.Tensor,
) -> dict[str, float]:
    """
    Run the complete scientific validation suite.

    Combines cloud-region, clear-region, vegetation, and edge metrics into
    a single dict suitable for JSON export and ISRO presentation.

    Args:
        pred:   [B, 3, H, W]
        target: [B, 3, H, W]
        cloudy: [B, 3, H, W] optional — used for baseline NDVI comparison
        mask:   [B, 1, H, W]

    Returns:
        Unified metrics dict.
    """
    from ai.metrics.psnr import psnr
    from ai.metrics.ssim import ssim
    from ai.metrics.sam import sam
    from ai.metrics.ndvi_mae import ndvi_mae

    out: dict[str, float] = {}

    # ── Global metrics ────────────────────────────────────────────────
    out["psnr_db"]          = float(psnr(pred, target))
    out["ssim"]             = float(ssim(pred, target))
    out["sam_rad"]          = float(sam(pred, target))
    out["sam_deg"]          = float(sam(pred, target, degrees=True))
    out["rmse"]             = float(((pred - target) ** 2).mean().sqrt())
    out["ndvi_mae_global"]  = float(ndvi_mae(pred, target))

    # ── Cloud-region metrics ──────────────────────────────────────────
    out.update(cloud_region_metrics(pred, target, mask))

    # ── Clear-region preservation ─────────────────────────────────────
    out.update(clear_region_psnr(pred, target, mask))

    # ── Vegetation metrics ────────────────────────────────────────────
    out.update(vegetation_region_metrics(pred, target, mask))

    # ── Edge preservation ─────────────────────────────────────────────
    out.update(edge_preservation_score(pred, target, mask))

    # ── Baseline comparison ───────────────────────────────────────────
    if cloudy is not None:
        baseline_cloud_mse = ((cloudy - target)[mask.bool().expand_as(cloudy)] ** 2).mean()
        pred_cloud_mse     = ((pred   - target)[mask.bool().expand_as(pred)] ** 2).mean()
        eps = 1e-8
        out["improvement_db"] = float(
            -10.0 * torch.log10(pred_cloud_mse + eps)
            - (-10.0 * torch.log10(baseline_cloud_mse + eps))
        )

    return out


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _mask_boundary(mask: torch.Tensor, dilation: int = 8) -> torch.Tensor:
    """Compute the boundary zone (±dilation pixels) around the cloud mask edge."""
    k = 2 * dilation + 1
    kernel = torch.ones(1, 1, k, k, device=mask.device)
    dilated  = F.conv2d(mask.float(), kernel, padding=dilation).clamp(0, 1)
    eroded   = 1.0 - F.conv2d(1.0 - mask.float(), kernel, padding=dilation).clamp(0, 1)
    return (dilated - eroded).clamp(0, 1)
