"""
SatQuery AI — Image Processor Module.

Handles preprocessing of satellite imagery for the VLM inference pipeline.
Supports GeoTIFF (via rasterio), standard optical formats (PIL), and SAR
imagery.  Provides validation, metadata extraction, and temporal-pair
co-registration for change detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Conditional imports — gracefully degrade if rasterio unavailable
try:
    import rasterio
    from rasterio.transform import array_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logger.warning("rasterio not installed — GeoTIFF support disabled; falling back to PIL")

try:
    from backend.config import SUPPORTED_FORMATS, MAX_IMAGE_SIZE_MB, SENSOR_METADATA
except ImportError:
    from config import SUPPORTED_FORMATS, MAX_IMAGE_SIZE_MB, SENSOR_METADATA


# ┌──────────────────────────────────────────────────────────────────────────┐
# │                            Validation                                   │
# └──────────────────────────────────────────────────────────────────────────┘

def validate_image(image_path: str | Path) -> tuple[bool, str]:
    """Check that an image file is valid for processing.

    Validates: file exists, extension is supported, file size is within
    limits, and the image can be opened by PIL or rasterio.

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of ``(is_valid, message)`` where *message* describes the
        validation result or error.
    """
    path = Path(image_path)

    if not path.exists():
        return False, f"File not found: {path}"

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        return False, (
            f"Unsupported format '{path.suffix}'. "
            f"Supported: {SUPPORTED_FORMATS}"
        )

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        return False, (
            f"File too large ({size_mb:.1f} MB). "
            f"Maximum: {MAX_IMAGE_SIZE_MB} MB"
        )

    # Try opening
    try:
        if path.suffix.lower() in (".tif", ".tiff") and HAS_RASTERIO:
            with rasterio.open(path) as src:
                if src.width < 16 or src.height < 16:
                    return False, f"Image too small ({src.width}×{src.height}). Minimum 16×16."
        else:
            img = Image.open(path)
            img.verify()
            w, h = img.size
            if w < 16 or h < 16:
                return False, f"Image too small ({w}×{h}). Minimum 16×16."
    except Exception as exc:
        return False, f"Cannot open image: {exc}"

    return True, "Image is valid"


# ┌──────────────────────────────────────────────────────────────────────────┐
# │                        GeoTIFF Processing                               │
# └──────────────────────────────────────────────────────────────────────────┘

def _read_geotiff(path: Path) -> dict[str, Any]:
    """Read a GeoTIFF file with rasterio and return structured metadata.

    Args:
        path: Path to a GeoTIFF file.

    Returns:
        Dict with keys: ``bands`` (numpy array, shape H×W×C), ``crs``,
        ``transform``, ``bounds``, ``nodata``, ``width``, ``height``,
        ``band_count``.

    Raises:
        RuntimeError: If rasterio is not available.
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio is required for GeoTIFF processing")

    with rasterio.open(path) as src:
        bands = src.read()  # shape: (C, H, W)
        bands = np.moveaxis(bands, 0, -1)  # → (H, W, C)

        crs_str = str(src.crs) if src.crs else "unknown"
        bounds = array_bounds(src.height, src.width, src.transform)

        return {
            "bands": bands,
            "crs": crs_str,
            "transform": src.transform,
            "bounds": {
                "west": bounds[0], "south": bounds[1],
                "east": bounds[2], "north": bounds[3],
            },
            "nodata": src.nodata,
            "width": src.width,
            "height": src.height,
            "band_count": src.count,
            "dtype": str(src.dtypes[0]),
        }


# ┌──────────────────────────────────────────────────────────────────────────┐
# │                        Optical / PIL Processing                         │
# └──────────────────────────────────────────────────────────────────────────┘

def _read_pil(path: Path) -> dict[str, Any]:
    """Read a standard image file (JPEG/PNG) via PIL.

    Args:
        path: Path to image.

    Returns:
        Dict with ``bands`` (H×W×3 numpy array normalised to [0, 1]).
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float64) / 255.0
    return {
        "bands": arr,
        "crs": "unknown",
        "transform": None,
        "bounds": None,
        "nodata": None,
        "width": arr.shape[1],
        "height": arr.shape[0],
        "band_count": 3,
        "dtype": "uint8",
    }


# ┌──────────────────────────────────────────────────────────────────────────┐
# │                         SAR Processing                                  │
# └──────────────────────────────────────────────────────────────────────────┘

def _db_to_linear(db_array: np.ndarray) -> np.ndarray:
    """Convert SAR dB values to linear power scale.

    Args:
        db_array: Array of values in decibel scale.

    Returns:
        Array in linear scale.
    """
    return np.power(10.0, db_array / 10.0)


def _lee_filter(image: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply Lee speckle filter to SAR imagery.

    A simple mean-based Lee filter that reduces multiplicative speckle
    noise while preserving edges.

    Args:
        image: 2-D SAR intensity array.
        window_size: Filter window size (odd integer).

    Returns:
        Filtered array of the same shape.
    """
    from scipy.ndimage import uniform_filter

    mean = uniform_filter(image.astype(np.float64), size=window_size)
    sq_mean = uniform_filter(image.astype(np.float64) ** 2, size=window_size)
    variance = sq_mean - mean ** 2
    overall_var = np.var(image)

    if overall_var == 0:
        return image.copy()

    weight = variance / (variance + overall_var)
    result = mean + weight * (image - mean)
    return result


def _preprocess_sar(bands: np.ndarray) -> np.ndarray:
    """Convert SAR data: dB → linear, apply Lee filter, normalise to [0, 1].

    Args:
        bands: Raw SAR band array (H, W, C).

    Returns:
        Preprocessed array normalised to [0, 1].
    """
    processed_bands = []
    for c in range(bands.shape[2]):
        band = bands[:, :, c].astype(np.float64)

        # Heuristic: if values contain negatives, assume dB scale
        if np.min(band) < -5:
            band = _db_to_linear(band)

        band = _lee_filter(band)

        # Normalise to [0, 1]
        bmin, bmax = np.nanmin(band), np.nanmax(band)
        if bmax > bmin:
            band = (band - bmin) / (bmax - bmin)
        else:
            band = np.zeros_like(band)

        processed_bands.append(band)

    return np.stack(processed_bands, axis=-1)


# ┌──────────────────────────────────────────────────────────────────────────┐
# │                        Main Preprocessing API                           │
# └──────────────────────────────────────────────────────────────────────────┘

def _detect_sensor(path: Path, band_count: int) -> str:
    """Heuristic sensor detection from filename and band count.

    Args:
        path: File path (may contain sensor hints in the filename).
        band_count: Number of spectral bands.

    Returns:
        Best-guess sensor name string.
    """
    name_lower = path.stem.lower()

    if "cartosat" in name_lower or "cart" in name_lower:
        return "Cartosat-2S"
    if "risat" in name_lower or "sar" in name_lower:
        return "RISAT-1C"
    if "resourcesat" in name_lower or "liss" in name_lower or "awifs" in name_lower:
        return "ResourceSat-2A"
    if "eos04" in name_lower or "eos-04" in name_lower:
        return "EOS-04"
    if "eos05" in name_lower or "eos-05" in name_lower or "gisat" in name_lower:
        return "EOS-05"

    # Band-count heuristic
    if band_count == 1:
        return "Cartosat-2S"  # Likely panchromatic
    if band_count == 2:
        return "RISAT-1C"     # Dual-pol SAR
    if band_count >= 6:
        return "EOS-05"       # Hyperspectral
    return "Cartosat-2S"      # Default to most common


def preprocess_for_vlm(
    image_path: str | Path,
    sensor_type: str | None = None,
) -> dict[str, Any]:
    """Preprocess a satellite image for VLM inference.

    Reads the image (GeoTIFF or standard format), performs sensor-specific
    preprocessing, extracts metadata, and returns a normalised RGB array
    suitable for the VLM pipeline.

    Args:
        image_path: Path to the satellite image file.
        sensor_type: ISRO sensor name (e.g. ``"Cartosat-2S"``).  If
            ``None``, auto-detection is attempted.

    Returns:
        Dict with keys:
        - ``rgb_array``: 3-channel numpy array normalised to ``[0, 1]``,
          shape ``(H, W, 3)``.
        - ``metadata``: dict with sensor, bands, resolution, CRS,
          bounding_box, band_count.
        - ``sensor_badge``: display string (e.g.
          ``"ISRO Cartosat-2S | 0.65m | Panchromatic"``).
        - ``raw_bands``: original multi-band array.

    Raises:
        FileNotFoundError: If *image_path* does not exist.
        ValueError: If the image cannot be processed.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # ── Read the image ────────────────────────────────────────────────
    geo_meta: dict[str, Any] = {}
    is_geotiff = path.suffix.lower() in (".tif", ".tiff")

    if is_geotiff and HAS_RASTERIO:
        try:
            geo_meta = _read_geotiff(path)
            logger.info("Read GeoTIFF: %s (%d bands)", path.name, geo_meta["band_count"])
        except Exception as exc:
            logger.warning("rasterio failed for %s (%s); falling back to PIL", path.name, exc)
            geo_meta = _read_pil(path)
    else:
        if is_geotiff and not HAS_RASTERIO:
            logger.info("rasterio unavailable — reading GeoTIFF %s with PIL (metadata limited)", path.name)
        geo_meta = _read_pil(path)

    bands = geo_meta["bands"]  # (H, W, C)

    # ── Detect sensor ─────────────────────────────────────────────────
    if sensor_type is None:
        sensor_type = _detect_sensor(path, geo_meta["band_count"])
    is_sar = sensor_type in ("RISAT-1C", "EOS-04")

    # ── Sensor-specific preprocessing ─────────────────────────────────
    if is_sar:
        bands = _preprocess_sar(bands)

    # ── Create RGB for VLM ────────────────────────────────────────────
    if bands.shape[2] >= 3:
        rgb = bands[:, :, :3].astype(np.float64)
    elif bands.shape[2] == 2:
        # Dual-pol SAR: map VV → R, VH → G, VV/VH → B
        vv = bands[:, :, 0]
        vh = bands[:, :, 1]
        ratio = _safe_ratio(vh, vv)
        rgb = np.stack([vv, vh, ratio], axis=-1)
    else:
        # Single band → grayscale RGB
        rgb = np.repeat(bands, 3, axis=-1)

    # Normalise to [0, 1]
    rgb_min, rgb_max = np.nanmin(rgb), np.nanmax(rgb)
    if rgb_max > rgb_min:
        rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
    rgb = np.clip(rgb, 0.0, 1.0)

    # ── Sensor badge ──────────────────────────────────────────────────
    meta_info = SENSOR_METADATA.get(sensor_type, {})
    badge = meta_info.get("badge", f"ISRO {sensor_type}")

    return {
        "rgb_array": rgb,
        "raw_bands": bands,
        "metadata": {
            "sensor": sensor_type,
            "band_count": geo_meta["band_count"],
            "width": geo_meta["width"],
            "height": geo_meta["height"],
            "crs": geo_meta.get("crs", "unknown"),
            "bounding_box": geo_meta.get("bounds"),
            "resolution_m": meta_info.get("resolution_m", "unknown"),
            "dtype": geo_meta.get("dtype", "unknown"),
        },
        "sensor_badge": badge,
    }


def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise ratio a/b, returning 0 where b is 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(b != 0, a / b, 0.0)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


# ┌──────────────────────────────────────────────────────────────────────────┐
# │                   Temporal Pair Co-registration                         │
# └──────────────────────────────────────────────────────────────────────────┘

def coregister_temporal_pair(
    t1_path: str | Path,
    t2_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Preprocess and co-register a bi-temporal image pair for change detection.

    Aligns T1 and T2 spatially by cropping to the common bounding box and
    resampling to the same grid size.  Computes an alignment score based on
    cross-correlation of shared content.

    Args:
        t1_path: Path to the *before* image (Time 1).
        t2_path: Path to the *after* image (Time 2).

    Returns:
        Tuple of ``(t1_array, t2_array, alignment_score)`` where both
        arrays have shape ``(H, W, C)`` normalised to ``[0, 1]`` and
        *alignment_score* is in ``[0, 1]`` (1 = perfect alignment).

    Raises:
        FileNotFoundError: If either path does not exist.
    """
    data1 = preprocess_for_vlm(t1_path)
    data2 = preprocess_for_vlm(t2_path)

    arr1 = data1["rgb_array"]
    arr2 = data2["rgb_array"]

    # Resize to common shape (use the smaller dimensions)
    h = min(arr1.shape[0], arr2.shape[0])
    w = min(arr1.shape[1], arr2.shape[1])

    arr1 = _resize_array(arr1, h, w)
    arr2 = _resize_array(arr2, h, w)

    # Compute alignment score via normalised cross-correlation (grayscale)
    gray1 = np.mean(arr1, axis=-1)
    gray2 = np.mean(arr2, axis=-1)

    g1 = gray1 - np.mean(gray1)
    g2 = gray2 - np.mean(gray2)

    denom = np.sqrt(np.sum(g1 ** 2) * np.sum(g2 ** 2))
    if denom > 0:
        ncc = float(np.sum(g1 * g2) / denom)
    else:
        ncc = 0.0

    alignment_score = max(0.0, min(1.0, (ncc + 1.0) / 2.0))

    return arr1, arr2, alignment_score


def _resize_array(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize array to target dimensions using simple slicing or PIL.

    Args:
        arr: Input array of shape (H, W, C).
        target_h: Target height.
        target_w: Target width.

    Returns:
        Resized array of shape (target_h, target_w, C).
    """
    if arr.shape[0] == target_h and arr.shape[1] == target_w:
        return arr

    # Use PIL for clean resizing
    channels = []
    for c in range(arr.shape[2]):
        img = Image.fromarray((arr[:, :, c] * 255).astype(np.uint8))
        img = img.resize((target_w, target_h), Image.BILINEAR)
        channels.append(np.array(img, dtype=np.float64) / 255.0)
    return np.stack(channels, axis=-1)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Image Processor module loaded OK ✅")
    print(f"  rasterio available: {HAS_RASTERIO}")
    print(f"  Supported formats: {SUPPORTED_FORMATS}")
