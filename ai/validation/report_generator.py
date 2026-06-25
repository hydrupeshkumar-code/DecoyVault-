"""
Structured scientific report builder.

Assembles a single, JSON-serialisable report dict from a metrics dict plus
scene/model provenance. The same structure backs both the JSON report and the
PDF (backend/services/pdf_service.py), and is returned to the frontend so the
verdict and metric tables can be rendered consistently everywhere.

Usage
-----
    from ai.validation.report_generator import build_report
    report = build_report(
        scene_id="ROI_0042_Karnataka_2024",
        metrics=metrics_dict,
        fallback=False,
    )
    json.dump(report, fp, indent=2)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from ai.validation.scientific_summary import summarize

# Fixed project provenance — LISS-IV band convention is load-bearing, so it is
# recorded in every report rather than left implicit.
SENSOR = "LISS-IV (Resourcesat-2/2A)"
BANDS = ["Green", "Red", "NIR"]
MODEL_NAME = "Restormer (cloud-adaptive)"
REPORT_VERSION = "2.0"

# Human-readable grouping of metric keys for tabular display.
_METRIC_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Global Image Quality", [
        ("psnr_db", "PSNR (dB)"),
        ("ssim", "SSIM"),
        ("sam_deg", "SAM (deg)"),
        ("rmse", "RMSE"),
        ("ndvi_mae_global", "NDVI MAE"),
    ]),
    ("Cloud-Region Reconstruction", [
        ("cloud_psnr_db", "Cloud PSNR (dB)"),
        ("cloud_ssim", "Cloud SSIM"),
        ("cloud_sam_deg", "Cloud SAM (deg)"),
        ("cloud_rmse", "Cloud RMSE"),
        ("cloud_pixel_count", "Cloud pixels"),
        ("improvement_db", "Improvement vs cloudy (dB)"),
    ]),
    ("Clear-Region Preservation", [
        ("clear_psnr_db", "Clear PSNR (dB)"),
    ]),
    ("Vegetation Fidelity (NDVI)", [
        ("veg_ndvi_mae", "NDVI MAE"),
        ("veg_ndvi_rmse", "NDVI RMSE"),
        ("veg_ndvi_bias", "NDVI bias"),
        ("ndvi_correlation", "NDVI correlation"),
    ]),
]


def _clean(value: object) -> object:
    """Make a metric value JSON-safe (NaN/Inf → None, round floats)."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 4)
    return value


def build_report(
    scene_id: str,
    metrics: dict[str, float],
    *,
    fallback: bool = False,
    elapsed_s: Optional[float] = None,
    dataset: Optional[str] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Build the structured report dict.

    Args:
        scene_id:  Human-readable scene identifier.
        metrics:   Metrics dict (from full_scientific_metrics / compute_image_metrics).
        fallback:  True if the reconstruction used the checkpoint-free fallback.
        elapsed_s: Optional inference time in seconds.
        dataset:   Optional dataset name (e.g. "SEN12MS-CR test split").
        extra:     Optional additional provenance to embed.

    Returns:
        JSON-serialisable report dict.
    """
    verdict = summarize(metrics)

    # Grouped, display-ready metric tables (skip groups with no data).
    groups: list[dict] = []
    for group_name, fields in _METRIC_GROUPS:
        rows = [
            {"label": label, "key": key, "value": _clean(metrics[key])}
            for key, label in fields
            if key in metrics and _clean(metrics[key]) is not None
        ]
        if rows:
            groups.append({"group": group_name, "rows": rows})

    report = {
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scene": {
            "scene_id": scene_id,
            "sensor": SENSOR,
            "bands": BANDS,
        },
        "model": {
            "name": MODEL_NAME,
            "dataset": dataset,
            "fallback_used": fallback,
            "elapsed_s": _clean(elapsed_s) if elapsed_s is not None else None,
        },
        "verdict": verdict.to_dict(),
        "metric_groups": groups,
        "metrics": {k: _clean(v) for k, v in metrics.items()},
    }

    if fallback:
        report["model"]["fallback_note"] = (
            "No trained checkpoint was loaded. Cloud pixels were filled with the "
            "clear-region per-band mean. Metrics reflect this baseline, NOT a "
            "trained Restormer reconstruction."
        )

    if extra:
        report["extra"] = extra

    return report
