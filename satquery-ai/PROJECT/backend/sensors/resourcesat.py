"""
SatQuery AI — ISRO ResourceSat Sensor Module.

Handles multispectral optical imagery from ResourceSat-2A (launched December 2016):
- LISS-IV: High resolution (5.8m), 3-band (Green, Red, NIR) or mono PAN
- LISS-III: Medium resolution (23.5m), 4-band (Green, Red, NIR, SWIR), 141km swath
- AWiFS (Advanced Wide Field Sensor): Synoptic resolution (56m), 4-band, 740km swath

Critical for nationwide agricultural crop discrimination, forest canopy inventory,
and drought assessment.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

RESOURCESAT_SPECS = {
    "LISS-IV": {
        "resolution_m": 5.8,
        "swath_km": 23.0,
        "radiometric_bits": 10,
        "bands": {
            "Green": {"range_nm": (520, 590), "gain": 0.95, "offset": -1.2},
            "Red": {"range_nm": (620, 680), "gain": 1.02, "offset": -1.4},
            "NIR": {"range_nm": (770, 860), "gain": 1.10, "offset": -2.0},
        },
    },
    "LISS-III": {
        "resolution_m": 23.5,
        "swath_km": 141.0,
        "radiometric_bits": 10,
        "bands": {
            "Green": {"range_nm": (520, 590), "gain": 0.94, "offset": -1.0},
            "Red": {"range_nm": (620, 680), "gain": 1.05, "offset": -1.3},
            "NIR": {"range_nm": (770, 860), "gain": 1.15, "offset": -2.5},
            "SWIR": {"range_nm": (1550, 1700), "gain": 0.88, "offset": -0.8},
        },
    },
    "AWiFS": {
        "resolution_m": 56.0,
        "swath_km": 740.0,
        "radiometric_bits": 10,
        "bands": {
            "Green": {"range_nm": (520, 590), "gain": 0.90, "offset": -0.9},
            "Red": {"range_nm": (620, 680), "gain": 1.00, "offset": -1.1},
            "NIR": {"range_nm": (770, 860), "gain": 1.18, "offset": -2.8},
            "SWIR": {"range_nm": (1550, 1700), "gain": 0.85, "offset": -0.7},
        },
    },
}


class ResourceSatCalibrator:
    """
    Radiometric and spectral calibration engine for ISRO ResourceSat-2A sensors.
    """

    def __init__(self, sensor_payload: str = "LISS-III"):
        self.payload = sensor_payload if sensor_payload in RESOURCESAT_SPECS else "LISS-III"
        self.specs = RESOURCESAT_SPECS[self.payload]

    def compute_agricultural_vigor(
        self,
        red_dn: np.ndarray,
        nir_dn: np.ndarray,
        swir_dn: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Computes agricultural crop vigor index and moisture stress proxy.
        Utilizes LISS-III / AWiFS NIR and SWIR bands.
        """
        red = red_dn.astype(np.float32)
        nir = nir_dn.astype(np.float32)

        # NDVI
        ndvi = (nir - red) / (nir + red + 1e-6)
        mean_ndvi = float(np.mean(ndvi))

        metrics: Dict[str, Any] = {
            "sensor": f"ResourceSat-2A ({self.payload})",
            "resolution_m": self.specs["resolution_m"],
            "mean_ndvi": round(mean_ndvi, 3),
            "healthy_crop_fraction": round(float(np.mean(ndvi > 0.5)), 3),
            "stressed_crop_fraction": round(float(np.mean((ndvi >= 0.2) & (ndvi <= 0.4))), 3),
        }

        # Normalized Difference Water/Moisture Index (NDWI/NDMI) if SWIR present
        if swir_dn is not None:
            swir = swir_dn.astype(np.float32)
            ndmi = (nir - swir) / (nir + swir + 1e-6)
            metrics["mean_ndmi"] = round(float(np.mean(ndmi)), 3)
            metrics["canopy_water_stress"] = "LOW" if np.mean(ndmi) > 0.2 else "MODERATE_TO_HIGH"

        return metrics

    def get_metadata_badge(self) -> str:
        """Returns standard ISRO UI metadata badge."""
        res = self.specs["resolution_m"]
        swath = self.specs["swath_km"]
        return f"ISRO ResourceSat-2A | {self.payload} ({res}m) | Swath {swath}km"
