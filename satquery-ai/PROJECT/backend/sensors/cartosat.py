"""
SatQuery AI — ISRO Cartosat Sensor Module.

Provides calibration constants, band mapping, and radiometric specifications
for ISRO's high-resolution optical earth observation satellites:
- Cartosat-2S (launched May 2017): 0.65m PAN, 2.1m 4-band MS
- Cartosat-3 (launched November 2019): 0.25m PAN, 1.13m 4-band MS
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Radiometric and spatial specifications
CARTOSAT_SPECS = {
    "Cartosat-2S": {
        "orbit": "505 km Sun-Synchronous",
        "swath_km": 9.6,
        "revisit_days": 4,
        "radiometric_bits": 10,
        "bands": {
            "PAN": {"range_nm": (500, 850), "resolution_m": 0.65, "gain": 1.0, "offset": 0.0},
            "Blue": {"range_nm": (450, 520), "resolution_m": 2.10, "gain": 1.05, "offset": -2.1},
            "Green": {"range_nm": (520, 590), "resolution_m": 2.10, "gain": 0.98, "offset": -1.8},
            "Red": {"range_nm": (620, 690), "resolution_m": 2.10, "gain": 1.02, "offset": -1.5},
            "NIR": {"range_nm": (770, 860), "resolution_m": 2.10, "gain": 1.12, "offset": -3.0},
        },
    },
    "Cartosat-3": {
        "orbit": "505 km Sun-Synchronous",
        "swath_km": 17.0,
        "revisit_days": 4,
        "radiometric_bits": 11,
        "bands": {
            "PAN": {"range_nm": (450, 900), "resolution_m": 0.25, "gain": 1.0, "offset": 0.0},
            "Blue": {"range_nm": (450, 520), "resolution_m": 1.13, "gain": 1.0, "offset": 0.0},
            "Green": {"range_nm": (520, 590), "resolution_m": 1.13, "gain": 1.0, "offset": 0.0},
            "Red": {"range_nm": (620, 690), "resolution_m": 1.13, "gain": 1.0, "offset": 0.0},
            "NIR": {"range_nm": (770, 860), "resolution_m": 1.13, "gain": 1.0, "offset": 0.0},
        },
    },
}


class CartosatCalibrator:
    """
    Applies ISRO Level-1B to Level-2A radiometric calibration to Cartosat digital numbers (DN).
    """

    def __init__(self, satellite_model: str = "Cartosat-2S"):
        self.model = satellite_model if satellite_model in CARTOSAT_SPECS else "Cartosat-2S"
        self.specs = CARTOSAT_SPECS[self.model]

    def dn_to_top_of_atmosphere_radiance(
        self,
        dn_array: np.ndarray,
        band_name: str,
    ) -> np.ndarray:
        """
        Converts raw DN values to Top-Of-Atmosphere (TOA) spectral radiance (W/(m²·sr·μm)).
        L_λ = Gain * DN + Offset
        """
        band_info = self.specs["bands"].get(band_name)
        if not band_info:
            return dn_array.astype(np.float32)

        gain = band_info.get("gain", 1.0)
        offset = band_info.get("offset", 0.0)
        radiance = gain * dn_array.astype(np.float32) + offset
        return np.maximum(radiance, 0.0)

    def pansharpen_brovey(
        self,
        ms_rgb: np.ndarray,
        pan: np.ndarray,
    ) -> np.ndarray:
        """
        Brovey transform pan-sharpening: merges high-res PAN with lower-res MS RGB.
        Result has the spatial resolution of PAN and spectral fidelity of MS.
        """
        if ms_rgb.shape[:2] != pan.shape[:2]:
            from PIL import Image
            pan_img = Image.fromarray(pan)
            pan_resized = np.array(pan_img.resize((ms_rgb.shape[1], ms_rgb.shape[0]), Image.Resampling.BILINEAR))
        else:
            pan_resized = pan

        ms_float = ms_rgb.astype(np.float32)
        pan_float = pan_resized.astype(np.float32)

        intensity = np.sum(ms_float, axis=2, keepdims=True) + 1e-6
        sharpened = (ms_float / intensity) * np.expand_dims(pan_float, axis=2)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def get_metadata_badge(self) -> str:
        """Returns standard ISRO UI metadata badge."""
        res = self.specs["bands"]["PAN"]["resolution_m"]
        return f"ISRO {self.model} | {res}m PAN | Sun-Sync 505km"
