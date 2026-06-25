"""
Singleton model loader for all AI components.

Loads models once at startup and caches them in memory.  All backend
services import from this module rather than loading weights themselves.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

log = logging.getLogger(__name__)

_RESTORMER: Optional["Restormer"] = None  # type: ignore[name-defined]
_DETECTOR: Optional["CloudDetector"] = None  # type: ignore[name-defined]
_DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device() -> torch.device:
    """Return the active inference device."""
    return _DEVICE


def load_restormer(
    checkpoint_path: str | Path,
    config_path: Optional[str | Path] = None,
    force_reload: bool = False,
) -> "Restormer":  # type: ignore[name-defined]
    """
    Load (or return cached) Restormer model.

    Args:
        checkpoint_path: Path to .pt checkpoint.
        config_path:     Optional path to train_config.yaml.
        force_reload:    If True, reload even if already cached.

    Returns:
        Restormer model in eval mode on the active device.
    """
    global _RESTORMER

    if _RESTORMER is not None and not force_reload:
        return _RESTORMER

    from ai.restormer.infer import load_model
    _RESTORMER = load_model(checkpoint_path, config_path, _DEVICE)
    log.info("Restormer loaded from %s", checkpoint_path)
    return _RESTORMER


def load_cloud_detector(
    checkpoint_path: str | Path,
    force_reload: bool = False,
) -> "CloudDetector":  # type: ignore[name-defined]
    """
    Load (or return cached) CloudDetector model.

    Args:
        checkpoint_path: Path to .pt checkpoint.
        force_reload:    If True, reload even if already cached.

    Returns:
        CloudDetector model in eval mode on the active device.
    """
    global _DETECTOR

    if _DETECTOR is not None and not force_reload:
        return _DETECTOR

    from ai.cloud_detector.model import CloudDetector
    _DETECTOR = CloudDetector(in_channels=3).to(_DEVICE)
    state = torch.load(checkpoint_path, map_location=_DEVICE)
    _DETECTOR.load_state_dict(state)
    _DETECTOR.eval()
    log.info("CloudDetector loaded from %s", checkpoint_path)
    return _DETECTOR


def get_restormer() -> Optional["Restormer"]:  # type: ignore[name-defined]
    """Return the cached Restormer model, or None if not loaded."""
    return _RESTORMER


def get_cloud_detector() -> Optional["CloudDetector"]:  # type: ignore[name-defined]
    """Return the cached CloudDetector model, or None if not loaded."""
    return _DETECTOR
