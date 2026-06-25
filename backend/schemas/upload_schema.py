"""Pydantic schemas for file upload endpoints."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    file_id: str           = Field(..., description="UUID identifier for the uploaded file")
    filename: str          = Field(..., description="Original filename")
    format: str            = Field(..., description="File format: tif or npy")
    shape: list[int]       = Field(..., description="Array shape [C, H, W]")
    size_bytes: int        = Field(..., description="File size in bytes")
    is_georeferenced: bool = Field(False, description="True if CRS and geotransform are present")
    crs: Optional[str]     = Field(None, description="Coordinate reference system (EPSG string)")
    message: str           = "Upload successful"
