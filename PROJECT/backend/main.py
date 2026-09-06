"""
SatQuery AI — FastAPI Application Entrypoint.

Problem Statement: SIH26167 (ISRO / Space Applications Centre)
Title: SatQuery AI — Interactive Vision-Language Assistant for Multimodal
       Remote Sensing Image Analysis through Text Queries.

This server exposes high-performance REST endpoints for:
- Vision-Language inference over satellite imagery (GeoChat-7B)
- 12-stage automated bi-temporal change detection with STSF-Net pseudo-change filter
- Spectral index extraction (NDVI, NDWI, NDBI, EVI, RVI)
- DOTA-calibrated object detection and SAM land-cover segmentation
- Built-in ISRO demo scenario execution for zero-GPU hackathon presentations
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from backend.api.routes import router as api_router
    from backend.config import settings
except ImportError:
    from api.routes import router as api_router
    from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("satquery.main")

BANNER = r"""
================================================================================
   SatQuery AI — ISRO Remote Sensing Vision-Language Assistant
   Smart India Hackathon 2026 | Problem Statement: SIH26167
   Space Applications Centre (SAC), Ahmedabad
================================================================================
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown lifecycle context."""
    # Startup
    try:
        print(BANNER)
    except Exception:
        print(BANNER.encode("ascii", errors="replace").decode("ascii"))
    demo_mode = getattr(settings, "DEMO_MODE", True)
    mode_str = "DEMO (Zero-GPU Canned Inference)" if demo_mode else "PRODUCTION (CUDA GeoChat-7B)"
    logger.info("=" * 60)
    logger.info("SatQuery AI is ready | ISRO SIH26167 | Mode: %s", mode_str)
    logger.info("Calibrated sensors: Cartosat-2S/3, RISAT-1C, ResourceSat-2A, EOS-04/05")
    logger.info("API Documentation available at: http://localhost:8000/docs")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("🛰️ SatQuery AI shutting down gracefully.")


app = FastAPI(
    title="SatQuery AI — ISRO Remote Sensing Assistant",
    description=(
        "An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis "
        "through Text Queries. Designed for ISRO / Space Applications Centre (SAC), SIH 2026 (PS SIH26167)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS for local frontend development and production hosting
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_and_log(request: Request, call_next):
    """Timing and access log middleware."""
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time_ms = (time.time() - start_time) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
        if request.url.path != "/health":
            logger.info(
                "%s %s -> %d (%.1f ms)",
                request.method,
                request.url.path,
                response.status_code,
                process_time_ms,
            )
        return response
    except Exception as exc:
        process_time_ms = (time.time() - start_time) * 1000.0
        logger.error("Unhandled error processing %s: %s", request.url.path, exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": f"Internal Server Error: {str(exc)}",
                "path": request.url.path,
                "process_time_ms": round(process_time_ms, 2),
            },
        )


# Mount all API routes
app.include_router(api_router)


@app.get("/", tags=["System"])
async def root():
    """Root landing endpoint with system status overview."""
    demo_mode = getattr(settings, "DEMO_MODE", True)
    return {
        "title": "SatQuery AI — ISRO Remote Sensing Assistant",
        "problem_statement": "SIH26167",
        "status": "online",
        "demo_mode": demo_mode,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "endpoints": {
            "health": "GET /health",
            "analyze": "POST /api/analyze",
            "change_detection": "POST /api/change-detection",
            "spectral_indices": "POST /api/spectral-indices",
            "demo_scenarios": "GET /api/demo/scenarios",
            "run_demo": "POST /api/demo/run/{scenario_id}",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
