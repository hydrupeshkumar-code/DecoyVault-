"""Pydantic schemas for reconstruction endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ReconstructRequest(BaseModel):
    file_id: str = Field(..., description="Cloudy image file ID")
    mask_id: str | None = Field(None, description="Cloud mask ID (optional)")
    tile_size: int = Field(256, ge=64, le=1024)
    overlap: int = Field(32, ge=0, le=128)
    preserve_clear: bool = Field(True, description="Keep clear pixels unchanged")


class ReconstructResponse(BaseModel):
    file_id: str
    result_id: str    = Field(..., description="UUID of the saved reconstruction")
    elapsed_s: float  = Field(..., description="Inference time in seconds")
    tiff_saved: bool  = Field(False, description="True if a GeoTIFF was saved alongside .npy")
    fallback: bool    = Field(False, description="True if the checkpoint-free clear-region-mean fallback was used (no trained model)")
    message: str      = "Reconstruction complete"
