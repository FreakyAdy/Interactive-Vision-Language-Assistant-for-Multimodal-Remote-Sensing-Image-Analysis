"""
SatQuery AI — Configuration Module.

Centralized configuration for the SatQuery AI backend.
Loads from environment variables with sensible defaults.
Defines model paths, device selection, sensor band mappings,
and the critical DEMO_MODE flag for GPU-free demonstrations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DEMO_DATA_DIR = PROJECT_DIR / "demo_data"
ML_MODELS_DIR = PROJECT_DIR / "ml_models"

# ---------------------------------------------------------------------------
# ISRO Sensor Band Mappings
# ---------------------------------------------------------------------------
# Maps sensor name -> { band_name: (wavelength_min_nm, wavelength_max_nm) }
SENSOR_BAND_MAPPINGS: dict[str, dict[str, tuple[float, float]]] = {
    "Cartosat-2S": {
        "PAN": (500.0, 850.0),
        "Blue": (450.0, 520.0),
        "Green": (520.0, 590.0),
        "Red": (620.0, 690.0),
        "NIR": (770.0, 860.0),
    },
    "Cartosat-3": {
        "PAN": (450.0, 900.0),
        "Blue": (450.0, 520.0),
        "Green": (520.0, 590.0),
        "Red": (620.0, 680.0),
        "NIR": (770.0, 880.0),
    },
    "RISAT-1C": {
        # SAR — single frequency, polarimetric
        "C-band": (5350.0, 5350.0),  # MHz stored for consistency; interpreted differently
    },
    "ResourceSat-2A_LISS3": {
        "Green": (520.0, 590.0),
        "Red": (620.0, 680.0),
        "NIR": (770.0, 860.0),
        "SWIR": (1550.0, 1700.0),
    },
    "ResourceSat-2A_LISS4": {
        "Green": (520.0, 590.0),
        "Red": (620.0, 680.0),
        "NIR": (770.0, 860.0),
    },
    "ResourceSat-2A_AWiFS": {
        "Green": (520.0, 590.0),
        "Red": (620.0, 680.0),
        "NIR": (770.0, 860.0),
        "SWIR": (1550.0, 1700.0),
    },
    "EOS-04": {
        # SAR L-band
        "L-band": (1270.0, 1270.0),  # MHz
    },
    "EOS-05": {
        # Hyperspectral — simplified representative bands
        "VNIR": (450.0, 875.0),
        "SWIR": (900.0, 2500.0),
    },
}

# Sensor metadata for display
SENSOR_METADATA: dict[str, dict[str, Any]] = {
    "Cartosat-2S": {
        "resolution_m": 0.65,
        "type": "Optical",
        "orbit_km": 505,
        "swath_km": 9.6,
        "radiometric_bits": 10,
        "revisit_days": 4,
        "badge": "ISRO Cartosat-2S | 0.65m | Panchromatic + MS",
    },
    "Cartosat-3": {
        "resolution_m": 0.25,
        "type": "Optical",
        "orbit_km": 509,
        "swath_km": 16.0,
        "radiometric_bits": 10,
        "revisit_days": 5,
        "badge": "ISRO Cartosat-3 | 0.25m | Highest-res Indian Civilian",
    },
    "RISAT-1C": {
        "resolution_m": 3.0,
        "type": "SAR",
        "frequency_ghz": 5.35,
        "polarizations": ["VV", "VH"],
        "incidence_range_deg": (20, 55),
        "badge": "ISRO RISAT-1C | 3m | SAR C-band VV+VH",
    },
    "ResourceSat-2A": {
        "resolution_m": 23.5,
        "type": "Optical",
        "swath_km": 141,
        "badge": "ISRO ResourceSat-2A | LISS-III 23.5m | 4-band",
    },
    "EOS-04": {
        "resolution_m": 1.0,
        "type": "SAR",
        "frequency_ghz": 1.27,
        "polarizations": ["HH", "HV", "VV", "VH"],
        "badge": "ISRO EOS-04 (RISAT-1A) | 1m | SAR L-band",
    },
    "EOS-05": {
        "resolution_m": 42.0,
        "type": "Hyperspectral",
        "vnir_bands": 158,
        "swir_bands": 256,
        "badge": "ISRO EOS-05 (GISAT-1A) | GEO | Hyperspectral 158+256 bands",
    },
}

# Supported image file formats
SUPPORTED_FORMATS: list[str] = [".tif", ".tiff", ".jpg", ".jpeg", ".png"]

# Maximum upload size
MAX_IMAGE_SIZE_MB: int = 50


# ---------------------------------------------------------------------------
# Pydantic Settings
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_title: str = "SatQuery AI — ISRO Remote Sensing Assistant"
    app_version: str = "1.0.0"
    app_description: str = (
        "An Interactive Vision-Language Assistant for Multimodal Remote Sensing "
        "Image Analysis through Text Queries. SIH26167 — ISRO / Space Applications Centre."
    )

    # Model configuration
    model_name: str = Field(
        default="MBZUAI/geochat-7B",
        description="HuggingFace model identifier for the VLM backbone",
    )
    device: str = Field(default=DEVICE, description="Compute device (cuda / cpu)")
    use_4bit: bool = Field(default=True, description="Use 4-bit quantization via bitsandbytes")
    max_new_tokens: int = Field(default=512, description="Maximum tokens for VLM generation")

    # Demo mode — when True, use synthetic data and canned responses (no GPU needed)
    demo_mode: bool = Field(
        default=True,
        description=(
            "When True, the app uses synthetic data and mock inference. "
            "This ensures the demo works without GPU or actual model weights."
        ),
    )

    # Image constraints
    max_image_size_mb: int = MAX_IMAGE_SIZE_MB
    supported_formats: list[str] = SUPPORTED_FORMATS

    # Paths
    base_dir: Path = BASE_DIR
    demo_data_dir: Path = DEMO_DATA_DIR
    ml_models_dir: Path = ML_MODELS_DIR

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]

    # Embedding model for query routing
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    model_config = {"env_prefix": "SATQUERY_", "env_file": ".env", "extra": "ignore"}


# Singleton
settings = Settings()


if __name__ == "__main__":
    print("SatQuery AI — Configuration")
    print(f"  Demo Mode : {settings.demo_mode}")
    print(f"  Device    : {settings.device}")
    print(f"  Model     : {settings.model_name}")
    print(f"  Base Dir  : {settings.base_dir}")
    print(f"  Supported : {settings.supported_formats}")
