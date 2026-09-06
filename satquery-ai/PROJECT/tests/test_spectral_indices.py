"""
Comprehensive unit tests for SatQuery AI Spectral Indices Module.
Requires at least 15 tests covering mathematical correctness, edge cases,
auto-selection, and ecological interpretation.
"""

import numpy as np
import pytest

from core.spectral_indices import (
    auto_select_index,
    compute_all_indices,
    compute_evi,
    compute_index,
    compute_ndbi,
    compute_ndvi,
    compute_ndwi,
    compute_rvi,
    interpret_index_value,
)


def test_ndvi_known_values():
    """NDVI for NIR=0.8, Red=0.2 should be (0.8 - 0.2) / (0.8 + 0.2) = 0.6."""
    img = np.zeros((4, 4, 2), dtype=np.float64)
    img[:, :, 0] = 0.2  # Red
    img[:, :, 1] = 0.8  # NIR
    arr, meta = compute_ndvi(img, {"Red": 0, "NIR": 1})
    assert np.allclose(arr, 0.6)
    assert meta["mean"] == pytest.approx(0.6, abs=1e-3)
    assert "vegetation" in meta["interpretation"].lower()


def test_ndwi_known_water_pixels():
    """NDWI for Green=0.7, NIR=0.1 should be (0.7 - 0.1) / (0.7 + 0.1) = 0.75."""
    img = np.zeros((4, 4, 2), dtype=np.float64)
    img[:, :, 0] = 0.7  # Green
    img[:, :, 1] = 0.1  # NIR
    arr, meta = compute_ndwi(img, {"Green": 0, "NIR": 1})
    assert np.allclose(arr, 0.75)
    assert meta["mean"] == pytest.approx(0.75, abs=1e-3)
    assert "water" in meta["interpretation"].lower()


def test_ndbi_known_values():
    """NDBI for SWIR=0.6, NIR=0.2 should be (0.6 - 0.2) / (0.6 + 0.2) = 0.5."""
    img = np.zeros((4, 4, 2), dtype=np.float64)
    img[:, :, 0] = 0.2  # NIR
    img[:, :, 1] = 0.6  # SWIR
    arr, meta = compute_ndbi(img, {"NIR": 0, "SWIR": 1})
    assert np.allclose(arr, 0.5)
    assert meta["mean"] == pytest.approx(0.5, abs=1e-3)
    assert "built-up" in meta["interpretation"].lower() or "impervious" in meta["interpretation"].lower()


def test_evi_calculation():
    """EVI returns non-empty array within plausible bounds."""
    img = np.zeros((4, 4, 3), dtype=np.float64)
    img[:, :, 0] = 0.1  # Blue
    img[:, :, 1] = 0.2  # Red
    img[:, :, 2] = 0.7  # NIR
    arr, meta = compute_evi(img, {"Blue": 0, "Red": 1, "NIR": 2})
    assert arr.shape == (4, 4)
    assert "interpretation" in meta


def test_rvi_sar_calculation():
    """RVI for VH=0.2, VV=0.4 should be 4*0.2 / (0.4 + 0.2) = 0.8/0.6 = 1.33 -> clipped to 1.0."""
    vh = np.ones((4, 4), dtype=np.float32) * 0.2
    vv = np.ones((4, 4), dtype=np.float32) * 0.4
    arr, meta = compute_rvi(vh, vv)
    assert np.all(arr <= 1.0)
    assert np.all(arr >= 0.0)


def test_ndvi_all_zeros():
    """Zero division edge case returns 0.0 and does not crash."""
    img = np.zeros((8, 8, 2), dtype=np.float64)
    arr, meta = compute_ndvi(img, {"Red": 0, "NIR": 1})
    assert np.all(arr == 0.0)
    assert meta["mean"] == 0.0


def test_ndvi_nan_handling():
    """NaN in input returns valid 0.0 and finite stats."""
    img = np.full((4, 4, 2), np.nan)
    arr, meta = compute_ndvi(img, {"Red": 0, "NIR": 1})
    assert not np.isnan(arr).any()
    assert np.isfinite(meta["mean"])


def test_ndvi_single_pixel():
    """Single pixel array (1, 1, 2) computes cleanly."""
    img = np.array([[[0.1, 0.9]]], dtype=np.float64)
    arr, meta = compute_ndvi(img, {"Red": 0, "NIR": 1})
    assert arr.shape == (1, 1)
    assert arr[0, 0] == pytest.approx(0.8, abs=1e-3)


def test_auto_select_flood():
    assert auto_select_index("How much has the flood expanded?") == "ndwi"
    assert auto_select_index("Where is the inundated water?") == "ndwi"


def test_auto_select_vegetation():
    assert auto_select_index("Check forest canopy health") == "ndvi"
    assert auto_select_index("Evaluate crop vegetation loss") == "ndvi"


def test_auto_select_urban():
    assert auto_select_index("Detect new building construction") == "ndbi"
    assert auto_select_index("Urban sprawl and concrete expansion") == "ndbi"


def test_auto_select_sar_sensor():
    assert auto_select_index("Crop biomass vigor", sensor_type="RISAT-1C") == "rvi"


def test_interpret_ndvi_dense():
    msg = interpret_index_value("ndvi", 0.72)
    assert "dense" in msg.lower() or "healthy" in msg.lower()


def test_interpret_ndvi_sparse():
    msg = interpret_index_value("ndvi", 0.15)
    assert "sparse" in msg.lower() or "soil" in msg.lower()


def test_interpret_ndwi_water():
    msg = interpret_index_value("ndwi", 0.45)
    assert "water" in msg.lower() or "flood" in msg.lower()


def test_compute_all_indices_rgb():
    """RGB image synthesizes all 4 optical indices."""
    rgb = np.ones((32, 32, 3), dtype=np.uint8) * 120
    indices = compute_all_indices(rgb)
    assert "ndvi" in indices
    assert "ndwi" in indices
    assert "ndbi" in indices
    assert "evi" in indices


def test_compute_index_dispatch():
    """compute_index dispatches correctly by name."""
    img = np.zeros((4, 4, 2), dtype=np.float64)
    img[:, :, 0] = 0.2
    img[:, :, 1] = 0.8
    arr, meta = compute_index("ndvi", img, {"Red": 0, "NIR": 1})
    assert meta["index"] == "NDVI"


def test_compute_index_invalid():
    """Raises ValueError on unknown index name."""
    with pytest.raises(ValueError):
        compute_index("invalid_index", np.zeros((2, 2, 2)))
