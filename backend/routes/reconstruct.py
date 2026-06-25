"""
Cloud removal reconstruction route.

Outputs:
    1. predictions/<result_id>.npy   — float32 [3,H,W] array (always)
    2. predictions/<result_id>.tif   — GeoTIFF with original CRS/transform (if source was georeferenced)

The GeoTIFF output is critical for scientific use: downstream tools
(QGIS, GDAL, rasterio scripts) need the CRS and geotransform to
align the reconstruction with other geospatial data layers.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.config.settings import settings
from backend.schemas.reconstruct_schema import ReconstructRequest, ReconstructResponse
from backend.services.file_resolver import resolve_npy
from backend.services.restormer_service import reconstruct

router = APIRouter(prefix="/reconstruct", tags=["reconstruct"])


@router.post("/", response_model=ReconstructResponse)
async def run_reconstruction(request: ReconstructRequest) -> ReconstructResponse:
    """
    Remove clouds from an uploaded LISS-IV image using the Restormer model.

    Workflow:
        1. Load the uploaded image (.npy or source .tif).
        2. Optionally load a pre-computed cloud mask from /detect.
        3. Run tiled Restormer inference with Hann-window blending.
        4. Save the result as .npy (and .tif if source was georeferenced).
        5. Return the result_id for /metrics and /report endpoints.
    """
    upload_dir = Path(settings.UPLOAD_DIR)

    # ── Load image (real upload or bundled demo scene) ────────────────
    npy_path = resolve_npy(request.file_id, "cloudy")
    if npy_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"File ID '{request.file_id}' not found. Upload the image first via /upload.",
        )

    image = np.load(str(npy_path)).astype(np.float32)  # [3, H, W]
    # Restormer service expects [H, W, 3]
    if image.ndim == 3 and image.shape[0] == 3:
        image_hwc = image.transpose(1, 2, 0)
    else:
        image_hwc = image

    # ── Load cloud mask (optional) ────────────────────────────────────
    cloud_mask: Optional[np.ndarray] = None
    if request.mask_id:
        mask_path = resolve_npy(request.mask_id, "mask")
        if mask_path is None:
            raise HTTPException(
                status_code=404,
                detail=f"Mask ID '{request.mask_id}' not found. Run /detect first.",
            )
        cloud_mask = np.load(str(mask_path)).astype(np.float32)

    # ── Run reconstruction ────────────────────────────────────────────
    result = reconstruct(
        image=image_hwc,
        cloud_mask=cloud_mask,
        tile_size=request.tile_size,
        overlap=request.overlap,
        preserve_clear=request.preserve_clear,
    )

    result_id = str(uuid.uuid4())
    out_dir = Path(settings.OUTPUT_DIR) / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always save as .npy [3, H, W]
    reconstructed_hwc: np.ndarray = result["reconstructed"]
    reconstructed_chw = reconstructed_hwc.transpose(2, 0, 1)
    np.save(str(out_dir / f"{result_id}.npy"), reconstructed_chw)

    # ── Save GeoTIFF if source was georeferenced ──────────────────────
    tiff_saved = False
    raw_source = _find_raw_tiff(upload_dir, request.file_id)
    if raw_source is not None:
        try:
            from ai.geospatial.tiff_io import read_tiff, write_tiff
            _, meta = read_tiff(str(raw_source), normalize=False)
            if meta and meta.is_georeferenced():
                meta.band_descriptions = ["Green_reconstructed", "Red_reconstructed", "NIR_reconstructed"]
                write_tiff(str(out_dir / f"{result_id}.tif"), reconstructed_chw, meta)
                tiff_saved = True
        except Exception:
            pass  # GeoTIFF output is best-effort; .npy always succeeds

    return ReconstructResponse(
        file_id=request.file_id,
        result_id=result_id,
        elapsed_s=float(result["elapsed_s"]),
        tiff_saved=tiff_saved,
        fallback=bool(result.get("fallback", False)),
    )


def _find_raw_tiff(upload_dir: Path, file_id: str) -> Optional[Path]:
    """Return the original .tif source file if it exists."""
    for ext in (".tif", ".tiff"):
        p = upload_dir / f"{file_id}{ext}"
        if p.exists():
            return p
    return None
