"""Pydantic schemas for report generation endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ReportRequest(BaseModel):
    pred_id: str = Field(..., description="Reconstruction result ID")
    file_id: str | None = Field(None, description="Original upload file ID (for before/after PDF comparison)")
    target_id: str | None = Field(None, description="Ground-truth image ID")
    mask_id: str | None = Field(None, description="Cloud mask ID")
    scene_id: str = Field("unknown", description="Human-readable scene identifier")


class ReportResponse(BaseModel):
    report_path: str = Field(..., description="Server-side path to the generated report")
    format: str = Field("pdf", description="Report format: 'pdf' or 'json'")
    message: str = "Report generated"
