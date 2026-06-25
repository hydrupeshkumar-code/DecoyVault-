"""Report generation route."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.config.settings import settings
from backend.schemas.report_schema import ReportRequest, ReportResponse
from backend.services.file_resolver import resolve_npy
from backend.services.metrics_service import compute_image_metrics
from backend.services.pdf_service import generate_pdf_report

router = APIRouter(prefix="/report", tags=["report"])


@router.post("/generate", response_model=ReportResponse)
async def generate_report(request: ReportRequest) -> ReportResponse:
    """
    Generate a PDF (or JSON fallback) quality report for a reconstruction.
    """
    pred_path = resolve_npy(request.pred_id, "pred")
    if pred_path is None:
        raise HTTPException(status_code=404, detail=f"Result ID '{request.pred_id}' not found.")

    pred = np.load(str(pred_path)).astype(np.float32)
    # Normalize to HWC for visualization
    if pred.ndim == 3 and pred.shape[0] <= 13:
        pred = pred.transpose(1, 2, 0)

    metrics: dict[str, float] = {}
    cloudy_img: Optional[np.ndarray] = None
    cloud_mask: Optional[np.ndarray] = None

    # Load cloudy source (upload or demo scene) for before/after PDF comparison
    if request.file_id:
        cloudy_path = resolve_npy(request.file_id, "cloudy")
        if cloudy_path is not None:
            cloudy_raw = np.load(str(cloudy_path)).astype(np.float32)
            cloudy_img = cloudy_raw.transpose(1, 2, 0) if cloudy_raw.shape[0] <= 13 else cloudy_raw

    if request.mask_id:
        msk_path = resolve_npy(request.mask_id, "mask")
        if msk_path is not None:
            cloud_mask = np.load(str(msk_path)).astype(np.float32)

    target: Optional[np.ndarray] = None
    if request.target_id:
        tgt_path = resolve_npy(request.target_id, "clear")
        if tgt_path is not None:
            target_raw = np.load(str(tgt_path)).astype(np.float32)
            target = target_raw.transpose(1, 2, 0) if target_raw.shape[0] <= 13 else target_raw
            metrics = compute_image_metrics(pred, target, mask=cloud_mask)

    report_path = (
        Path(settings.OUTPUT_DIR)
        / "reports"
        / f"{request.pred_id}_report.pdf"
    )
    final_path = generate_pdf_report(
        output_path=report_path,
        metrics=metrics,
        cloudy_image=cloudy_img,
        reconstructed_image=pred,
        cloud_mask=cloud_mask,
        scene_id=request.scene_id,
    )

    # Structured, machine-readable report (verdict + grouped metrics) saved
    # alongside the PDF for downstream tooling and the frontend.
    if metrics:
        from ai.validation.report_generator import build_report
        report_dict = build_report(scene_id=request.scene_id, metrics=metrics)
        json_sidecar = Path(settings.OUTPUT_DIR) / "reports" / f"{request.pred_id}_report.json"
        json_sidecar.parent.mkdir(parents=True, exist_ok=True)
        with open(json_sidecar, "w") as f:
            json.dump(report_dict, f, indent=2)

    fmt = "pdf" if final_path.endswith(".pdf") else "json"
    return ReportResponse(report_path=final_path, format=fmt)


@router.get("/download/{report_filename}")
async def download_report(report_filename: str) -> FileResponse:
    """Serve a previously generated report file."""
    report_path = Path(settings.OUTPUT_DIR) / "reports" / report_filename
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(str(report_path), media_type="application/octet-stream")
