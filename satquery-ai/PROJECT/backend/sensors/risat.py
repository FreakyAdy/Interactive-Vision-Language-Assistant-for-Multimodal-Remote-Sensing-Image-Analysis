"""
SatQuery AI — ISRO RISAT & EOS SAR Sensor Module.

Handles Synthetic Aperture Radar (SAR) data from:
- RISAT-1C: C-band (5.35 GHz), VV+VH dual-pol, 3m (FRS-1) / 25m (MRS)
- EOS-04 (RISAT-1A): C-band (5.405 GHz), High-resolution dual-pol, 1m (HRS)

Features:
- SAR amplitude and intensity calibration to sigma-nought (σ°) in decibels (dB)
- Lee speckle noise reduction filtering
- Radar Vegetation Index (RVI) computation for biomass / crop structure
- Water surface inundation segmentation (specular reflection / low backscatter)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy.ndimage import uniform_filter

RISAT_SPECS = {
    "RISAT-1C": {
        "band": "C-band (5.35 GHz)",
        "polarizations": ["VV", "VH", "HH", "HV"],
        "modes": {
            "FRS-1": {"resolution_m": 3.0, "swath_km": 25},
            "MRS": {"resolution_m": 25.0, "swath_km": 115},
            "CRS": {"resolution_m": 50.0, "swath_km": 223},
        },
        "water_threshold_db": -18.0,
        "forest_threshold_db": -9.0,
    },
    "EOS-04": {
        "band": "C-band (5.405 GHz)",
        "polarizations": ["VV", "VH"],
        "modes": {
            "HRS": {"resolution_m": 1.0, "swath_km": 10},
            "FRS-2": {"resolution_m": 2.0, "swath_km": 25},
        },
        "water_threshold_db": -17.5,
        "forest_threshold_db": -8.5,
    },
}


class RISATCalibrator:
    """
    Calibrator and processor for ISRO SAR payloads.
    """

    def __init__(self, satellite_model: str = "RISAT-1C"):
        self.model = satellite_model if satellite_model in RISAT_SPECS else "RISAT-1C"
        self.specs = RISAT_SPECS[self.model]

    def dn_to_sigma_nought_db(self, dn_array: np.ndarray, cal_factor: float = -40.0) -> np.ndarray:
        """
        Converts raw SAR digital numbers (DN) to backscatter coefficient sigma0 in dB.
        σ° (dB) = 20 * log10(DN) + K
        """
        eps = 1e-7
        dn_float = np.maximum(dn_array.astype(np.float32), eps)
        sigma_db = 20.0 * np.log10(dn_float) + cal_factor
        return np.clip(sigma_db, -35.0, 10.0)

    def lee_filter(self, img_array: np.ndarray, window_size: int = 7) -> np.ndarray:
        """
        Applies Lee speckle filter to SAR imagery while preserving sharp edges.
        """
        img = img_array.astype(np.float32)
        mean = uniform_filter(img, (window_size, window_size))
        sqr_mean = uniform_filter(img**2, (window_size, window_size))
        variance = np.maximum(sqr_mean - mean**2, 1e-6)

        overall_variance = np.var(img) + 1e-6
        weights = variance / (variance + overall_variance)
        filtered = mean + weights * (img - mean)
        return filtered

    def compute_rvi(self, vv_linear: np.ndarray, vh_linear: np.ndarray) -> np.ndarray:
        """
        Computes Radar Vegetation Index:
        RVI = 4 * σ_VH / (σ_VV + σ_VH)
        Values range from 0 (smooth bare soil) to 1 (dense vegetation canopy).
        """
        numerator = 4.0 * vh_linear
        denominator = vv_linear + vh_linear + 1e-6
        rvi = numerator / denominator
        return np.clip(rvi, 0.0, 1.0)

    def segment_water_inundation(self, sigma_vv_db: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Segments water / flood areas based on low radar backscatter (specular reflection).
        Returns binary mask (1=water, 0=non-water) and inundated area ratio.
        """
        threshold = self.specs["water_threshold_db"]
        filtered = self.lee_filter(sigma_vv_db)
        water_mask = (filtered < threshold).astype(np.uint8)
        fraction = float(np.mean(water_mask))
        return water_mask, fraction

    def get_metadata_badge(self) -> str:
        """Returns standard ISRO UI metadata badge."""
        band = self.specs["band"]
        return f"ISRO {self.model} | {band} SAR | All-Weather"
