"""Pydantic schemas for metrics endpoints."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class MetricsRequest(BaseModel):
    pred_id: str = Field(..., description="Reconstruction result ID")
    target_id: str = Field(..., description="Ground-truth clear image file ID")
    mask_id: Optional[str] = Field(None, description="Cloud mask ID")
    cloudy_id: Optional[str] = Field(None, description="Original cloudy image ID (for NDVI baseline)")


class MetricsResponse(BaseModel):
    pred_id: str

    # ── Global metrics ────────────────────────────────────────────────
    psnr_db: float
    ssim: float
    sam_rad: float
    sam_deg: float
    rmse: float
    ndvi_mae_global: float

    # ── Cloud-region metrics (primary scientific evaluation) ──────────
    cloud_psnr_db: Optional[float] = None
    cloud_ssim: Optional[float] = None
    cloud_sam_rad: Optional[float] = None
    cloud_sam_deg: Optional[float] = None
    cloud_rmse: Optional[float] = None
    cloud_pixel_count: Optional[int] = None

    # ── Clear-region preservation ─────────────────────────────────────
    clear_psnr_db: Optional[float] = None

    # ── Vegetation metrics ────────────────────────────────────────────
    veg_ndvi_mae: Optional[float] = None
    veg_ndvi_rmse: Optional[float] = None
    veg_ndvi_bias: Optional[float] = None
    ndvi_correlation: Optional[float] = None

    # ── Legacy / convenience ──────────────────────────────────────────
    ndvi_mae_cloud: Optional[float] = None
    ndvi_improvement: Optional[float] = None
    improvement_db: Optional[float] = None
