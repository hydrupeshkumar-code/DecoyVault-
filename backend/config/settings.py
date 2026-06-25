"""
Application settings loaded from environment variables with sane defaults.
"""

from __future__ import annotations

import os
from pathlib import Path


class Settings:
    # Paths
    RESTORMER_CHECKPOINT: str = os.getenv(
        "RESTORMER_CHECKPOINT",
        "ai/restormer/checkpoints/phase2_final.pt",
    )
    DETECTOR_CHECKPOINT: str = os.getenv(
        "DETECTOR_CHECKPOINT",
        "ai/cloud_detector/checkpoints/best.pt",
    )
    TRAIN_CONFIG: str = os.getenv(
        "TRAIN_CONFIG",
        "ai/restormer/train_config.yaml",
    )
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "outputs")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "outputs/uploads")
    # Bundled synthetic demo scenes (agriculture / urban / water).
    # Resolved for file IDs of the form ``demo_<scene>`` so the pipeline can be
    # exercised end-to-end without uploading data — see services/file_resolver.py.
    DEMO_DIR: str = os.getenv("DEMO_DIR", "datasets/demo")

    # Inference parameters
    TILE_SIZE: int = int(os.getenv("TILE_SIZE", "256"))
    TILE_OVERLAP: int = int(os.getenv("TILE_OVERLAP", "32"))
    CLOUD_THRESHOLD: float = float(os.getenv("CLOUD_THRESHOLD", "0.5"))

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # CORS
    ALLOWED_ORIGINS: list[str] = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000"
    ).split(",")

    def ensure_dirs(self) -> None:
        for d in (self.OUTPUT_DIR, self.UPLOAD_DIR):
            Path(d).mkdir(parents=True, exist_ok=True)


settings = Settings()
