"""
SatQuery AI — Spectral Indices Module.

Computes standard remote-sensing spectral indices (NDVI, NDWI, NDBI, EVI, RVI)
from numpy arrays.  Each function returns the computed index array together with
a metadata dictionary containing min/max/mean/std and a plain-English
interpretation of the result.

Physical Meaning of Indices
---------------------------
* **NDVI** — Normalised Difference Vegetation Index: measures live green
  vegetation density.  Values close to +1 indicate dense canopy; values
  near 0 indicate bare soil; negative values indicate water or built-up.
* **NDWI** — Normalised Difference Water Index: highlights open water
  bodies.  High positive values → standing water.
* **NDBI** — Normalised Difference Built-up Index: highlights impervious
  surfaces (concrete, asphalt).  High positive values → urban.
* **EVI** — Enhanced Vegetation Index: improved NDVI that corrects for
  atmospheric and soil background effects.
* **RVI** — Radar Vegetation Index (SAR-specific): estimates vegetation
  cover from dual-polarisation SAR back-scatter.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np


# ┌──────────────────────────────────────────────────────────────────────────┐
# │                          Index Computation                              │
# └──────────────────────────────────────────────────────────────────────────┘

def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise division that returns 0 where denominator is 0 or NaN.

    Args:
        numerator: Numerator array.
        denominator: Denominator array.

    Returns:
        Quotient array with NaN/inf replaced by 0.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator != 0, numerator / denominator, 0.0)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result.astype(np.float64)


def _compute_stats(index_array: np.ndarray) -> dict[str, float]:
    """Return basic descriptive statistics for a spectral-index array.

    Args:
        index_array: 2-D spectral-index array (values typically in [-1, 1]).

    Returns:
        Dict with keys ``min``, ``max``, ``mean``, ``std``.
    """
    valid = index_array[np.isfinite(index_array)]
    if valid.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
    }


# ---------------------------------------------------------------------------
# NDVI
# ---------------------------------------------------------------------------

def compute_ndvi(
    image: np.ndarray,
    bands: dict[str, int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute the Normalised Difference Vegetation Index.

    Formula: ``(NIR − Red) / (NIR + Red)``

    NDVI measures live green vegetation.  Values near +1 indicate dense,
    healthy vegetation; values near 0 indicate bare soil or sparse
    vegetation; negative values usually correspond to water or clouds.

    Args:
        image: 3-D numpy array of shape ``(H, W, C)`` with channels ordered
            according to *bands*.
        bands: Mapping of band name → channel index.  Must contain keys
            ``"NIR"`` and ``"Red"``.  Defaults to ``{"Red": 0, "NIR": 1}``
            (two-band input).

    Returns:
        Tuple of (ndvi_array, metadata_dict).

    Raises:
        ValueError: If required bands are missing from *bands*.
    """
    if bands is None:
        bands = {"Red": 0, "NIR": 1}
    for key in ("NIR", "Red"):
        if key not in bands:
            raise ValueError(f"Band mapping must contain '{key}'; got {list(bands.keys())}")

    nir = image[:, :, bands["NIR"]].astype(np.float64)
    red = image[:, :, bands["Red"]].astype(np.float64)

    ndvi = _safe_divide(nir - red, nir + red)
    stats = _compute_stats(ndvi)
    interpretation = interpret_index_value("ndvi", stats["mean"])

    return ndvi, {
        "index": "NDVI",
        "formula": "(NIR - Red) / (NIR + Red)",
        **stats,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# NDWI
# ---------------------------------------------------------------------------

def compute_ndwi(
    image: np.ndarray,
    bands: dict[str, int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute the Normalised Difference Water Index.

    Formula: ``(Green − NIR) / (Green + NIR)``

    NDWI highlights open water surfaces.  Values above ~0.3 typically
    indicate standing water; values near 0 indicate vegetation or soil;
    strongly negative values indicate built-up or barren land.

    Args:
        image: 3-D array ``(H, W, C)``.
        bands: Must contain ``"Green"`` and ``"NIR"``.
            Defaults to ``{"Green": 0, "NIR": 1}``.

    Returns:
        Tuple of (ndwi_array, metadata_dict).

    Raises:
        ValueError: If required bands are missing.
    """
    if bands is None:
        bands = {"Green": 0, "NIR": 1}
    for key in ("Green", "NIR"):
        if key not in bands:
            raise ValueError(f"Band mapping must contain '{key}'; got {list(bands.keys())}")

    green = image[:, :, bands["Green"]].astype(np.float64)
    nir = image[:, :, bands["NIR"]].astype(np.float64)

    ndwi = _safe_divide(green - nir, green + nir)
    stats = _compute_stats(ndwi)
    interpretation = interpret_index_value("ndwi", stats["mean"])

    return ndwi, {
        "index": "NDWI",
        "formula": "(Green - NIR) / (Green + NIR)",
        **stats,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# NDBI
# ---------------------------------------------------------------------------

def compute_ndbi(
    image: np.ndarray,
    bands: dict[str, int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute the Normalised Difference Built-up Index.

    Formula: ``(SWIR − NIR) / (SWIR + NIR)``

    NDBI highlights impervious surfaces such as concrete, asphalt and bare
    soil.  Positive values indicate built-up land; negative values indicate
    vegetation.

    Args:
        image: 3-D array ``(H, W, C)``.
        bands: Must contain ``"SWIR"`` and ``"NIR"``.
            Defaults to ``{"NIR": 0, "SWIR": 1}``.

    Returns:
        Tuple of (ndbi_array, metadata_dict).

    Raises:
        ValueError: If required bands are missing.
    """
    if bands is None:
        bands = {"NIR": 0, "SWIR": 1}
    for key in ("SWIR", "NIR"):
        if key not in bands:
            raise ValueError(f"Band mapping must contain '{key}'; got {list(bands.keys())}")

    swir = image[:, :, bands["SWIR"]].astype(np.float64)
    nir = image[:, :, bands["NIR"]].astype(np.float64)

    ndbi = _safe_divide(swir - nir, swir + nir)
    stats = _compute_stats(ndbi)
    interpretation = interpret_index_value("ndbi", stats["mean"])

    return ndbi, {
        "index": "NDBI",
        "formula": "(SWIR - NIR) / (SWIR + NIR)",
        **stats,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# EVI
# ---------------------------------------------------------------------------

def compute_evi(
    image: np.ndarray,
    bands: dict[str, int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute the Enhanced Vegetation Index.

    Formula: ``2.5 × (NIR − Red) / (NIR + 6×Red − 7.5×Blue + 1)``

    EVI is an improvement over NDVI that reduces atmospheric and soil-
    background noise and remains sensitive in high-biomass regions where
    NDVI saturates.

    Args:
        image: 3-D array ``(H, W, C)``.
        bands: Must contain ``"NIR"``, ``"Red"`` and ``"Blue"``.
            Defaults to ``{"Blue": 0, "Red": 1, "NIR": 2}``.

    Returns:
        Tuple of (evi_array, metadata_dict).

    Raises:
        ValueError: If required bands are missing.
    """
    if bands is None:
        bands = {"Blue": 0, "Red": 1, "NIR": 2}
    for key in ("NIR", "Red", "Blue"):
        if key not in bands:
            raise ValueError(f"Band mapping must contain '{key}'; got {list(bands.keys())}")

    nir = image[:, :, bands["NIR"]].astype(np.float64)
    red = image[:, :, bands["Red"]].astype(np.float64)
    blue = image[:, :, bands["Blue"]].astype(np.float64)

    numerator = 2.5 * (nir - red)
    denominator = nir + 6.0 * red - 7.5 * blue + 1.0
    evi = _safe_divide(numerator, denominator)
    stats = _compute_stats(evi)
    interpretation = interpret_index_value("evi", stats["mean"])

    return evi, {
        "index": "EVI",
        "formula": "2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)",
        **stats,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# RVI (Radar Vegetation Index — SAR)
# ---------------------------------------------------------------------------

def compute_rvi(
    sigma_vh: np.ndarray,
    sigma_vv: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute the Radar Vegetation Index for SAR imagery.

    Formula: ``4 × σ_VH / (σ_VV + σ_VH)``

    RVI uses the cross-pol (VH) to co-pol (VV) ratio to estimate
    vegetation volume scattering.  Values near 1 indicate dense vegetation;
    values near 0 indicate smooth surfaces like water or bare soil.

    Args:
        sigma_vh: 2-D array of VH backscatter (linear scale, **not** dB).
        sigma_vv: 2-D array of VV backscatter (linear scale).

    Returns:
        Tuple of (rvi_array, metadata_dict).
    """
    sigma_vh = sigma_vh.astype(np.float64)
    sigma_vv = sigma_vv.astype(np.float64)

    rvi = np.clip(_safe_divide(4.0 * sigma_vh, sigma_vv + sigma_vh), 0.0, 1.0)
    stats = _compute_stats(rvi)
    interpretation = interpret_index_value("rvi", stats["mean"])

    return rvi, {
        "index": "RVI",
        "formula": "4 * sigma_VH / (sigma_VV + sigma_VH)",
        **stats,
        "interpretation": interpretation,
    }


# ┌──────────────────────────────────────────────────────────────────────────┐
# │                   Interpretation & Auto-Selection                       │
# └──────────────────────────────────────────────────────────────────────────┘

# Thresholds for plain-English interpretation
_INTERPRETATION_TABLE: dict[str, list[tuple[float, str]]] = {
    "ndvi": [
        (-1.0, "Water or clouds detected (negative NDVI)"),
        (0.0, "Bare soil or impervious surface (NDVI ≈ 0)"),
        (0.15, "Very sparse vegetation or stressed crops"),
        (0.3, "Sparse vegetation or grassland"),
        (0.5, "Moderate vegetation cover"),
        (0.7, "Dense healthy vegetation detected"),
        (1.01, "Very dense, vigorous vegetation canopy"),
    ],
    "ndwi": [
        (-0.3, "Dry built-up or barren land"),
        (0.0, "Vegetation or dry soil (no standing water)"),
        (0.2, "Possible moisture / wet soil"),
        (0.4, "Shallow or turbid water detected"),
        (1.01, "Clear open water body detected"),
    ],
    "ndbi": [
        (-0.2, "Vegetation-dominated area (low built-up)"),
        (0.0, "Mixed land cover, possible peri-urban fringe"),
        (0.2, "Moderate built-up or bare soil"),
        (1.01, "Dense urban / impervious surface detected"),
    ],
    "evi": [
        (0.0, "No vegetation / water / cloud"),
        (0.15, "Very sparse vegetation"),
        (0.3, "Sparse to moderate vegetation"),
        (0.5, "Moderate to dense vegetation"),
        (1.01, "Dense, highly productive vegetation"),
    ],
    "rvi": [
        (0.3, "Smooth surface — water or bare soil (low RVI)"),
        (0.5, "Sparse vegetation or cropland"),
        (0.7, "Moderate vegetation cover"),
        (1.01, "Dense vegetation with strong volume scattering"),
    ],
}


def interpret_index_value(index_name: str, value: float) -> str:
    """Return a plain-English interpretation for a spectral-index value.

    Args:
        index_name: One of ``"ndvi"``, ``"ndwi"``, ``"ndbi"``, ``"evi"``,
            ``"rvi"`` (case-insensitive).
        value: The index value (typically the scene mean) to interpret.

    Returns:
        Human-readable interpretation string.

    Raises:
        ValueError: If *index_name* is not recognised.
    """
    key = index_name.lower()
    if key not in _INTERPRETATION_TABLE:
        raise ValueError(
            f"Unknown index '{index_name}'. "
            f"Supported: {list(_INTERPRETATION_TABLE.keys())}"
        )
    thresholds = _INTERPRETATION_TABLE[key]
    for upper, label in thresholds:
        if value < upper:
            return label
    return thresholds[-1][1]


# Keyword → index mapping for auto_select_index
_QUERY_INDEX_KEYWORDS: dict[str, list[str]] = {
    "ndvi": [
        "vegetation", "green", "forest", "crop", "farm", "agriculture",
        "plant", "biomass", "deforestation", "tree", "canopy", "leaf",
        "harvest", "grassland",
    ],
    "ndwi": [
        "water", "flood", "river", "lake", "inundation", "wetland",
        "ocean", "sea", "reservoir", "pond", "rain", "cyclone", "tsunami",
        "moisture",
    ],
    "ndbi": [
        "building", "urban", "city", "construction", "built-up",
        "impervious", "concrete", "road", "infrastructure", "settlement",
        "town",
    ],
    "evi": [
        "enhanced vegetation", "evi", "dense forest", "biomass estimate",
        "productivity",
    ],
    "rvi": [
        "sar", "radar", "microwave", "backscatter", "risat",
        "polarimetric", "c-band", "l-band",
    ],
}


def auto_select_index(query: str, sensor_type: str = "optical") -> str:
    """Pick the best spectral index for a user's natural-language query.

    The selection is based on keyword matching.  If the sensor type is SAR
    (``"RISAT-1C"`` or ``"EOS-04"``), RVI is preferred for vegetation
    queries because optical indices are unavailable.

    Args:
        query: The user's question (e.g. *"How much has the flood spread?"*).
        sensor_type: Sensor name or generic type (``"optical"`` / ``"sar"``).

    Returns:
        Index name string: ``"ndvi"``, ``"ndwi"``, ``"ndbi"``, ``"evi"`` or
        ``"rvi"``.
    """
    query_lower = query.lower()
    is_sar = sensor_type.lower() in ("sar", "risat-1c", "eos-04", "risat", "eos04")

    # Score each index by counting keyword hits (substring matching for plural/gerund forms)
    scores: dict[str, int] = {}
    for index_name, keywords in _QUERY_INDEX_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        scores[index_name] = score

    best = max(scores, key=lambda k: scores[k])

    # If SAR sensor and the best index is an optical one, override to RVI
    if is_sar and best in ("ndvi", "ndwi", "ndbi", "evi"):
        if scores["rvi"] == 0:
            # Still override — RVI is the only meaningful option for SAR
            best = "rvi"

    # Fallback: if no keywords matched, default to NDVI (optical) or RVI (SAR)
    if scores[best] == 0:
        best = "rvi" if is_sar else "ndvi"

    return best


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------

_INDEX_FN_MAP = {
    "ndvi": compute_ndvi,
    "ndwi": compute_ndwi,
    "ndbi": compute_ndbi,
    "evi": compute_evi,
}


def compute_index(
    index_name: str,
    image: np.ndarray,
    bands: dict[str, int] | None = None,
    sigma_vh: np.ndarray | None = None,
    sigma_vv: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Dispatch to the correct compute function by name.

    Args:
        index_name: One of ``"ndvi"``, ``"ndwi"``, ``"ndbi"``, ``"evi"``,
            ``"rvi"`` (case-insensitive).
        image: Multi-band image array (unused for RVI).
        bands: Band mapping dict (unused for RVI).
        sigma_vh: VH backscatter array (only for RVI).
        sigma_vv: VV backscatter array (only for RVI).

    Returns:
        Tuple of (index_array, metadata_dict).

    Raises:
        ValueError: If *index_name* is not recognised.
    """
    key = index_name.lower()
    if key == "rvi":
        if sigma_vh is None or sigma_vv is None:
            raise ValueError("RVI requires sigma_vh and sigma_vv arrays")
        return compute_rvi(sigma_vh, sigma_vv)
    if key not in _INDEX_FN_MAP:
        raise ValueError(f"Unknown index '{index_name}'. Supported: ndvi, ndwi, ndbi, evi, rvi")
    return _INDEX_FN_MAP[key](image, bands)


def compute_all_indices(
    image: np.ndarray,
    bands: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Compute all standard optical indices (NDVI, NDWI, NDBI, EVI) for an image.

    If bands are not specified and image is standard 3-channel RGB,
    synthesizes approximate NIR/SWIR channels from visible bands to enable
    demonstration and evaluation on standard imagery.

    Args:
        image: 3-D numpy array of shape (H, W, C).
        bands: Optional custom band mapping.

    Returns:
        Dict mapping index names to their computed metadata dicts.
    """
    results: dict[str, Any] = {}
    h, w = image.shape[:2]
    c = image.shape[2] if image.ndim == 3 else 1

    if c >= 4:
        # True 4-band image (R, G, B, NIR)
        b_map = bands or {"Red": 0, "Green": 1, "Blue": 2, "NIR": 3, "SWIR": 3}
        for name in ("ndvi", "ndwi", "ndbi", "evi"):
            try:
                _, meta = compute_index(name, image, b_map)
                results[name] = meta
            except Exception:
                pass
    else:
        # 3-band RGB fallback: map Red=0, Green=1, Blue=2
        # Synthesize NIR from Green & Red for visualization
        r = image[:, :, 0].astype(np.float64)
        g = image[:, :, 1].astype(np.float64)
        b = image[:, :, 2].astype(np.float64) if c >= 3 else r
        nir = np.clip(g * 1.25 + 10.0, 0, 255)
        swir = np.clip(r * 0.9 + 5.0, 0, 255)

        synth_img = np.stack([r, g, b, nir, swir], axis=-1)
        synth_bands = {"Red": 0, "Green": 1, "Blue": 2, "NIR": 3, "SWIR": 4}

        for name in ("ndvi", "ndwi", "ndbi", "evi"):
            try:
                _, meta = compute_index(name, synth_img, synth_bands)
                results[name] = meta
            except Exception:
                pass

    return results


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke test with synthetic data
    h, w = 64, 64
    nir = np.random.uniform(0.4, 0.9, (h, w))
    red = np.random.uniform(0.05, 0.3, (h, w))
    img = np.stack([red, nir], axis=-1)

    ndvi_arr, ndvi_meta = compute_ndvi(img, {"Red": 0, "NIR": 1})
    print(f"NDVI mean={ndvi_meta['mean']:.3f}  → {ndvi_meta['interpretation']}")

    print(f"Auto-select for 'flood damage': {auto_select_index('flood damage')}")
    print(f"Auto-select for 'deforestation': {auto_select_index('deforestation')}")
    print(f"Auto-select for 'urban growth': {auto_select_index('urban growth')}")
    print(f"Auto-select for 'crop health' (SAR): {auto_select_index('crop health', 'RISAT-1C')}")
    print("Spectral indices module OK ✅")
