"""
Central file-ID → path resolver for the CloudVision pipeline.

Every route used to build ``.npy`` paths inline, which meant the bundled demo
scenes (``demo_agriculture`` / ``demo_urban`` / ``demo_water``) could not be fed
through /detect, /reconstruct, /metrics or /report — they 404'd because no such
upload existed. This module centralises resolution so a file ID is looked up in
the right place regardless of whether it is a real upload, a pipeline result, or
a demo scene.

Resolution rules
----------------
Demo IDs have the form ``demo_<scene>`` and map to ``datasets/demo/<scene>/``:

    role="cloudy"  → demo/<scene>/cloudy.npy
    role="clear"   → demo/<scene>/clear.npy
    role="mask"    → demo/<scene>/mask.npy

Regular (UUID) IDs map to their conventional locations:

    role="cloudy" / "clear"  → UPLOAD_DIR/<id>.npy
    role="pred"              → OUTPUT_DIR/predictions/<id>.npy
    role="mask"              → OUTPUT_DIR/predictions/<id>_mask.npy

``resolve_npy`` returns ``None`` when nothing matches so callers keep raising
their own 404s with a meaningful message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np

from backend.config.settings import settings

Role = Literal["cloudy", "clear", "pred", "mask"]

DEMO_PREFIX = "demo_"


def is_demo_id(file_id: str) -> bool:
    """True if ``file_id`` refers to a bundled demo scene (``demo_<scene>``)."""
    return file_id.startswith(DEMO_PREFIX)


def demo_scene(file_id: str) -> str:
    """Extract the scene name from a demo ID (``demo_agriculture`` → ``agriculture``)."""
    return file_id[len(DEMO_PREFIX):]


def resolve_npy(file_id: str, role: Role = "cloudy") -> Optional[Path]:
    """
    Resolve a file ID + role to a concrete ``.npy`` path on disk.

    Args:
        file_id: Upload UUID, pipeline result UUID, or ``demo_<scene>``.
        role:    Which array the caller wants for this ID.

    Returns:
        Path to an existing ``.npy`` file, or ``None`` if not found.
    """
    if is_demo_id(file_id):
        scene_dir = Path(settings.DEMO_DIR) / demo_scene(file_id)
        # Demo scenes only carry source arrays; a "pred" of a demo is treated as
        # its clear reference so /report and /metrics still have something sane.
        name = {"cloudy": "cloudy", "clear": "clear", "mask": "mask", "pred": "clear"}[role]
        candidate = scene_dir / f"{name}.npy"
        return candidate if candidate.exists() else None

    if role == "pred":
        candidate = Path(settings.OUTPUT_DIR) / "predictions" / f"{file_id}.npy"
    elif role == "mask":
        candidate = Path(settings.OUTPUT_DIR) / "predictions" / f"{file_id}_mask.npy"
    else:  # cloudy / clear are uploads
        candidate = Path(settings.UPLOAD_DIR) / f"{file_id}.npy"

    return candidate if candidate.exists() else None


def load_chw(path: Path) -> np.ndarray:
    """
    Load a ``.npy`` and normalise to ``[C, H, W]`` float32.

    Uses the same channels-last heuristic as the rest of the backend: an array
    whose last axis is <= 13 is treated as HWC and transposed to CHW.
    """
    arr = np.load(str(path)).astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] <= 13:
        arr = arr.transpose(2, 0, 1)
    return arr
