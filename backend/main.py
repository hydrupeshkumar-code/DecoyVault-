"""
ChaturVyuha CloudVision AI – FastAPI Backend

Startup:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000

Or via Docker:
    docker compose up backend
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config.settings import settings
from backend.routes import detect, metrics, reconstruct, report, upload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load AI models once at startup; release nothing on shutdown (stateless)."""
    settings.ensure_dirs()

    restormer_ckpt = Path(settings.RESTORMER_CHECKPOINT)
    if restormer_ckpt.exists():
        from backend.services.model_loader import load_restormer
        load_restormer(
            checkpoint_path=restormer_ckpt,
            config_path=settings.TRAIN_CONFIG if Path(settings.TRAIN_CONFIG).exists() else None,
        )
    else:
        log.warning(
            "Restormer checkpoint not found at %s – reconstruction endpoint will fail.",
            restormer_ckpt,
        )

    detector_ckpt = Path(settings.DETECTOR_CHECKPOINT)
    if detector_ckpt.exists():
        from backend.services.model_loader import load_cloud_detector
        load_cloud_detector(checkpoint_path=detector_ckpt)
    else:
        log.warning(
            "Cloud detector checkpoint not found at %s – heuristic fallback will be used.",
            detector_ckpt,
        )

    yield


app = FastAPI(
    title="ChaturVyuha CloudVision AI",
    description=(
        "Generative AI-Based Cloud Removal and Reconstruction for LISS-IV "
        "Satellite Imagery using the Restormer architecture."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(detect.router)
app.include_router(reconstruct.router)
app.include_router(metrics.router)
app.include_router(report.router)


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """Liveness check endpoint."""
    return {"status": "ok", "version": app.version}


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
