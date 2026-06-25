"""
Scientific metrics route.

Evaluates reconstructions against ground-truth clear references using the
full scientific validation suite including cloud-region, clear-region,
vegetation, and spectral accuracy metrics.

This endpoint is designed for ISRO scientific review — it goes beyond
simple global PSNR and provides the per-region metrics that matter for
remote sensing applications.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.schemas.metrics_schema import MetricsRequest, MetricsResponse
from backend.services.file_resolver import resolve_npy
from backend.services.metrics_service import compute_image_metrics

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.post("/", response_model=MetricsResponse)
async def compute_metrics_route(request: MetricsRequest) -> MetricsResponse:
    """
    Compute full scientific metrics for a cloud removal reconstruction.

    Requires:
        pred_id:    UUID from /reconstruct
        target_id:  UUID of the ground-truth clear image (uploaded via /upload)

    Optional:
        mask_id:    UUID from /detect (enables cloud-region metrics)
        cloudy_id:  UUID of the original cloudy input (enables improvement_db)

    Returns comprehensive per-region metrics suitable for scientific review.
    """
    pred_path   = resolve_npy(request.pred_id, "pred")
    target_path = resolve_npy(request.target_id, "clear")

    if pred_path is None:
        raise HTTPException(status_code=404, detail=f"Prediction '{request.pred_id}' not found.")
    if target_path is None:
        raise HTTPException(status_code=404, detail=f"Target '{request.target_id}' not found.")

    pred   = np.load(str(pred_path)).astype(np.float32)
    target = np.load(str(target_path)).astype(np.float32)

    # Normalize to [C, H, W] if needed
    pred   = _ensure_chw(pred)
    target = _ensure_chw(target)

    if pred.shape != target.shape:
        raise HTTPException(
            status_code=422,
            detail=f"Shape mismatch: prediction {pred.shape} vs target {target.shape}.",
        )

    cloudy: Optional[np.ndarray] = None
    if request.cloudy_id:
        cloudy_path = resolve_npy(request.cloudy_id, "cloudy")
        if cloudy_path is not None:
            cloudy = _ensure_chw(np.load(str(cloudy_path)).astype(np.float32))

    mask: Optional[np.ndarray] = None
    if request.mask_id:
        mask_path = resolve_npy(request.mask_id, "mask")
        if mask_path is not None:
            mask = np.load(str(mask_path)).astype(np.float32)
            if mask.ndim == 2:
                mask = mask[np.newaxis, np.newaxis]   # [1,1,H,W]
            elif mask.ndim == 3:
                mask = mask[np.newaxis]

    m = compute_image_metrics(pred, target, cloudy=cloudy, mask=mask)

    return MetricsResponse(
        pred_id=request.pred_id,
        # Global
        psnr_db=m["psnr_db"],
        ssim=m["ssim"],
        sam_rad=m["sam_rad"],
        sam_deg=m["sam_deg"],
        rmse=m["rmse"],
        ndvi_mae_global=m["ndvi_mae_global"],
        # Cloud-region
        cloud_psnr_db=m.get("cloud_psnr_db"),
        cloud_ssim=m.get("cloud_ssim"),
        cloud_sam_rad=m.get("cloud_sam_rad"),
        cloud_sam_deg=m.get("cloud_sam_deg"),
        cloud_rmse=m.get("cloud_rmse"),
        cloud_pixel_count=m.get("cloud_pixel_count"),
        # Clear-region preservation
        clear_psnr_db=m.get("clear_psnr_db"),
        # Vegetation
        veg_ndvi_mae=m.get("veg_ndvi_mae"),
        veg_ndvi_rmse=m.get("veg_ndvi_rmse"),
        veg_ndvi_bias=m.get("veg_ndvi_bias"),
        ndvi_correlation=m.get("ndvi_correlation"),
        # Convenience
        ndvi_mae_cloud=m.get("ndvi_mae_cloud"),
        ndvi_improvement=m.get("ndvi_improvement"),
        improvement_db=m.get("improvement_db"),
    )


def _ensure_chw(arr: np.ndarray) -> np.ndarray:
    """Normalize to [C, H, W] regardless of input layout."""
    if arr.ndim == 3 and arr.shape[2] <= 13:
        return arr.transpose(2, 0, 1)
    return arr
