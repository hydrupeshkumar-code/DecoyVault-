"""Pydantic schemas for cloud detection endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    file_id: str = Field(..., description="File ID from the upload response")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Cloud binarisation threshold")


class DetectResponse(BaseModel):
    file_id: str
    cloud_fraction: float = Field(..., description="Fraction of pixels classified as cloud")
    mask_id: str = Field(..., description="UUID of the saved cloud mask")
    message: str = "Detection complete"
