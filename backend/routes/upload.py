"""
Upload route — accepts GeoTIFF and NumPy satellite imagery.

Supported formats:
    .tif / .tiff  — GeoTIFF (preferred, preserves CRS and geotransform)
    .npy          — Legacy NumPy array (float32 [H,W,3] or [C,H,W])

The upload stores both the raw file and a normalized [3,H,W] float32 NumPy
array alongside the original, so downstream services can use either.

LISS-IV band convention:
    Band 0 → Green
    Band 1 → Red
    Band 2 → NIR
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.config.settings import settings
from backend.schemas.upload_schema import UploadResponse

router = APIRouter(prefix="/upload", tags=["upload"])

_SUPPORTED_SUFFIXES = {".tif", ".tiff", ".npy"}


@router.post("/", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload a LISS-IV satellite image for cloud removal.

    Accepts GeoTIFF (recommended) or legacy .npy arrays.
    Returns a ``file_id`` used by /detect and /reconstruct.

    The file must represent a 3-band image (Green, Red, NIR).
    For GeoTIFF inputs: any number of bands ≥ 3 is accepted; pass
    ``band_indices`` in subsequent requests to select specific bands.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Accepted: .tif, .tiff, .npy",
        )

    contents = await file.read()
    file_id = str(uuid.uuid4())

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save raw file (preserves all metadata for GeoTIFF)
    raw_path = upload_dir / f"{file_id}{suffix}"
    raw_path.write_bytes(contents)

    # Parse and validate the array
    try:
        arr, meta, shape, is_georef = _parse_upload(raw_path, suffix)
    except Exception as exc:
        raw_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not parse file: {exc}")

    # Save normalized [3, H, W] float32 array alongside raw file
    npy_path = upload_dir / f"{file_id}.npy"
    np.save(str(npy_path), arr)

    return UploadResponse(
        file_id=file_id,
        filename=file.filename or "",
        format=suffix.lstrip("."),
        shape=list(shape),
        size_bytes=len(contents),
        is_georeferenced=is_georef,
        crs=str(meta.crs) if (meta and meta.crs) else None,
    )


def _parse_upload(
    path: Path,
    suffix: str,
) -> tuple[np.ndarray, Optional[object], tuple[int, ...], bool]:
    """
    Parse an uploaded file into a normalized [3, H, W] float32 array.

    Returns (arr, tiff_meta, original_shape, is_georeferenced).
    """
    if suffix == ".npy":
        arr = np.load(str(path)).astype(np.float32)
        # Normalize HWC → CHW
        if arr.ndim == 3 and arr.shape[2] <= 13:
            arr = arr.transpose(2, 0, 1)
        if arr.ndim != 3 or arr.shape[0] < 3:
            raise ValueError(
                f"Expected [H,W,3] or [3,H,W] array, got shape {arr.shape}. "
                "Array must have exactly 3 bands (Green, Red, NIR)."
            )
        arr = arr[:3]  # take first 3 bands
        arr = np.clip(arr, 0.0, 1.0)
        return arr, None, arr.shape, False

    # GeoTIFF
    from ai.geospatial.tiff_io import read_tiff
    arr, meta = read_tiff(str(path), normalize=True)

    if arr.shape[0] < 3:
        raise ValueError(
            f"GeoTIFF has {arr.shape[0]} band(s). Need at least 3 (Green, Red, NIR). "
            "Re-export with all spectral bands included."
        )

    # For multi-band files (>3), take first 3 bands (assumes LISS-IV ordering)
    # The user can override band selection via /reconstruct band_indices param.
    arr = arr[:3]

    return arr, meta, arr.shape, meta.is_georeferenced() if meta else False
