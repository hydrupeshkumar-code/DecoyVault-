"""Cloud detection route."""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.config.settings import settings
from backend.schemas.detect_schema import DetectRequest, DetectResponse
from backend.services.cloud_detector_service import detect_clouds

router = APIRouter(prefix="/detect", tags=["detect"])


@router.post("/", response_model=DetectResponse)
async def detect(request: DetectRequest) -> DetectResponse:
    """
    Run cloud detection on a previously uploaded image.

    Returns the cloud fraction and a ``mask_id`` pointing to the saved
    binary cloud mask.
    """
    img_path = Path(settings.UPLOAD_DIR) / f"{request.file_id}.npy"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"File ID '{request.file_id}' not found.")

    image = np.load(str(img_path)).astype(np.float32)
    # Saved as [3,H,W] CHW by upload route; detect_clouds expects [H,W,3] HWC.
    if image.ndim == 3 and image.shape[0] <= 13:
        image = image.transpose(1, 2, 0)

    result = detect_clouds(image, threshold=request.threshold)

    mask_id = str(uuid.uuid4())
    mask_path = Path(settings.OUTPUT_DIR) / "predictions" / f"{mask_id}_mask.npy"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(mask_path), result["binary_mask"])

    return DetectResponse(
        file_id=request.file_id,
        cloud_fraction=float(result["cloud_fraction"]),
        mask_id=mask_id,
    )
