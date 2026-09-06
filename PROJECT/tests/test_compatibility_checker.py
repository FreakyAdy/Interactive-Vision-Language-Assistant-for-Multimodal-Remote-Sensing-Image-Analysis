"""
Test Suite for InputCompatibilityChecker
========================================
Validates input scopes, formats, and sensor modalities according to SIH26167.
"""

from pathlib import Path
import pytest
from backend.core.image_processor import InputCompatibilityChecker

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo_data"


def test_empty_images():
    res = InputCompatibilityChecker.check_compatibility([])
    assert res["is_compatible"] is False
    assert res["scope"] == "UNKNOWN"


def test_single_image_compatibility():
    p1 = DEMO_DIR / "flood_t1.tif"
    if p1.exists():
        res = InputCompatibilityChecker.check_compatibility([p1])
        assert res["is_compatible"] is True
        assert res["scope"] == "SINGLE_IMAGE"
        assert len(res["image_metas"]) == 1


def test_bitemporal_pair_compatibility():
    p1 = DEMO_DIR / "flood_t1.tif"
    p2 = DEMO_DIR / "flood_t2.tif"
    if p1.exists() and p2.exists():
        res = InputCompatibilityChecker.check_compatibility([p1, p2])
        assert res["is_compatible"] is True
        assert res["scope"] == "BITEMPORAL_PAIR"


def test_cross_modal_pair_compatibility():
    opt = DEMO_DIR / "cartosat_optical_sample.tif"
    sar = DEMO_DIR / "risat_sar_sample.tif"
    if opt.exists() and sar.exists():
        res = InputCompatibilityChecker.check_compatibility([opt, sar])
        assert res["is_compatible"] is True
        assert res["scope"] == "CROSS_MODAL_PAIR"
        modalities = [m["modality"] for m in res["image_metas"]]
        assert "OPTICAL" in modalities
        assert "SAR" in modalities
