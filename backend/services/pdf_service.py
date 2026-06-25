"""
PDF report generation service.

Produces a structured PDF summary of a cloud removal run, including
input/output visualisations and metric tables.

Requires: reportlab (pip install reportlab)
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


def _try_import_reportlab():
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            Image as RLImage,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
        return True, (colors, A4, getSampleStyleSheet, cm, RLImage, Paragraph,
                      SimpleDocTemplate, Spacer, Table, TableStyle)
    except ImportError:
        return False, None


def generate_pdf_report(
    output_path: str | Path,
    metrics: dict[str, float],
    cloudy_image: Optional[np.ndarray] = None,
    reconstructed_image: Optional[np.ndarray] = None,
    cloud_mask: Optional[np.ndarray] = None,
    scene_id: str = "unknown",
) -> str:
    """
    Generate a PDF report for a single cloud removal result.

    Args:
        output_path:         Destination .pdf path.
        metrics:             Metric dict from metrics_service.
        cloudy_image:        Optional [H, W, 3] cloudy input (for thumbnail).
        reconstructed_image: Optional [H, W, 3] output (for thumbnail).
        cloud_mask:          Optional [H, W] binary cloud mask.
        scene_id:            Scene identifier string.

    Returns:
        Absolute path to the generated PDF.
    """
    # Scientific verdict (Excellent / Acceptable / Needs Improvement) derived
    # from the metric suite. Computed up-front so both the PDF and the JSON
    # fallback carry it.
    verdict: Optional[dict] = None
    if metrics:
        try:
            from ai.validation.scientific_summary import summarize
            verdict = summarize(metrics).to_dict()
        except Exception:
            verdict = None

    available, libs = _try_import_reportlab()
    if not available:
        # Fallback: write a plain JSON report
        json_path = str(output_path).replace(".pdf", ".json")
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump({"scene_id": scene_id, "metrics": metrics, "verdict": verdict}, f, indent=2)
        return json_path

    (colors, A4, getSampleStyleSheet, cm, RLImage, Paragraph,
     SimpleDocTemplate, Spacer, Table, TableStyle) = libs

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(output_path), pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("ChaturVyuha CloudVision AI – Reconstruction Report", styles["Title"]))
    story.append(Spacer(1, 0.5 * cm))

    # Scene info
    story.append(Paragraph(f"Scene ID: {scene_id}", styles["Normal"]))
    story.append(Paragraph(f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]))
    story.append(Paragraph("Sensor: LISS-IV (Resourcesat-2)", styles["Normal"]))
    story.append(Spacer(1, 0.5 * cm))

    # Scientific verdict
    if verdict is not None:
        story.append(Paragraph("Scientific Verdict", styles["Heading2"]))
        story.append(Paragraph(
            f"<b>{verdict['verdict']}</b> &nbsp; (score {verdict['score']:.2f})",
            styles["Normal"],
        ))
        story.append(Paragraph(verdict.get("headline", ""), styles["Normal"]))
        for note in verdict.get("notes", []):
            story.append(Paragraph(f"• {note}", styles["Normal"]))
        story.append(Spacer(1, 0.5 * cm))

    # Metrics table
    story.append(Paragraph("Quality Metrics", styles["Heading2"]))
    table_data = [["Metric", "Value"]] + [
        [k, f"{v:.4f}"] for k, v in metrics.items()
    ]
    tbl = Table(table_data, colWidths=[8 * cm, 4 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    story.append(tbl)

    doc.build(story)
    return str(output_path.resolve())
