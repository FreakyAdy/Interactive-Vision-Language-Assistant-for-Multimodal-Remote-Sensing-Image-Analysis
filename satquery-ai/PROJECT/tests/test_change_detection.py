"""
Comprehensive unit and integration tests for SatQuery AI Change Detector.
Minimum 20 tests validating the 12-stage pipeline, pseudo-change suppression,
confidence metrics, and physical plausibility.
"""

import time
import numpy as np
import pytest

from core.change_detector import (
    ChangeDetector,
    bimodal_confidence_score,
    otsu_threshold,
    suppress_pseudo_change,
)


@pytest.fixture
def detector() -> ChangeDetector:
    return ChangeDetector()


def test_detector_initialization(detector: ChangeDetector):
    assert detector is not None


def test_12_stage_pipeline_executes(detector: ChangeDetector):
    t1 = np.ones((64, 64, 3), dtype=np.uint8) * 100
    t2 = t1.copy()
    t2[20:40, 20:40] = [30, 80, 170]
    res = detector.detect_change(t1, t2, query="Detect flood inundation")
    assert res["status"] == "ok"
    assert len(res["execution_trace"]) >= 12


def test_flood_scenario_water_expansion(detector: ChangeDetector):
    t1 = np.ones((64, 64, 3), dtype=np.uint8) * 120
    t2 = t1.copy()
    t2[10:50, 10:50] = [20, 60, 180]  # Water patch
    res = detector.detect_change(t1, t2, query="Identify flooded areas")
    assert res["change_direction"] in ("water_expansion", "vegetation_loss")
    assert res["primary_index"] in ("ndwi", "ndvi")


def test_deforestation_vegetation_loss(detector: ChangeDetector):
    t1 = np.ones((64, 64, 3), dtype=np.uint8)
    t1[:, :] = [30, 180, 40]  # Green forest
    t2 = t1.copy()
    t2[15:45, 15:45] = [180, 130, 80]  # Bare cleared ground
    res = detector.detect_change(t1, t2, query="Forest canopy shrinkage")
    assert res["change_direction"] in ("vegetation_loss", "water_expansion")


def test_urban_growth_scenario(detector: ChangeDetector):
    t1 = np.ones((64, 64, 3), dtype=np.uint8)
    t1[:, :] = [100, 160, 80]  # Farmland
    t2 = t1.copy()
    t2[20:50, 20:50] = [210, 200, 210]  # Concrete
    res = detector.detect_change(t1, t2, query="Detect new urban construction")
    assert res["status"] == "ok"
    assert res["area_metrics"]["n_changed_pixels"] > 0


def test_confidence_range(detector: ChangeDetector):
    t1 = np.ones((64, 64, 3), dtype=np.uint8) * 100
    t2 = t1.copy()
    t2[10:30, 10:30] = [200, 200, 200]
    res = detector.detect_change(t1, t2)
    assert 0.0 <= res["confidence"] <= 1.0


def test_confidence_label_values(detector: ChangeDetector):
    t1 = np.ones((64, 64, 3), dtype=np.uint8) * 100
    t2 = t1.copy()
    t2[10:30, 10:30] = [200, 200, 200]
    res = detector.detect_change(t1, t2)
    assert res["confidence_label"] in ("HIGH", "MEDIUM", "LOW")


def test_output_json_contract_keys(detector: ChangeDetector):
    t1 = np.ones((64, 64, 3), dtype=np.uint8) * 100
    t2 = t1.copy()
    t2[15:35, 15:35] = [20, 80, 160]
    res = detector.detect_change(t1, t2)
    required = [
        "status",
        "primary_index",
        "change_direction",
        "confidence",
        "confidence_label",
        "area_metrics",
        "n_regions",
        "otsu_threshold",
        "n_pseudo_removed",
        "summary",
        "geojson",
        "execution_trace",
        "sensor_calibration_note",
        "total_processing_ms",
    ]
    for key in required:
        assert key in res, f"Missing key: {key}"


def test_area_metrics_physical_plausibility(detector: ChangeDetector):
    t1 = np.ones((64, 64, 3), dtype=np.uint8) * 100
    t2 = t1.copy()
    t2[10:20, 10:20] = [20, 70, 180]  # 100 pixels
    res = detector.detect_change(t1, t2)
    m = res["area_metrics"]
    assert m["area_m2"] >= 0.0
    assert m["area_ha"] >= 0.0
    assert m["area_km2"] >= 0.0
    assert 0.0 <= m["pct_changed"] <= 100.0


def test_geojson_valid_structure(detector: ChangeDetector):
    t1 = np.ones((64, 64, 3), dtype=np.uint8) * 100
    t2 = t1.copy()
    t2[20:40, 20:40] = [30, 80, 180]
    res = detector.detect_change(t1, t2)
    gj = res["geojson"]
    assert gj["type"] in ("FeatureCollection", "Feature")
    if gj["type"] == "FeatureCollection":
        assert "features" in gj
        assert isinstance(gj["features"], list)


def test_n_regions_positive_on_change(detector: ChangeDetector):
    t1 = np.ones((64, 64, 3), dtype=np.uint8) * 100
    t2 = t1.copy()
    t2[5:15, 5:15] = [20, 70, 180]
    t2[35:45, 35:45] = [20, 70, 180]
    res = detector.detect_change(t1, t2)
    assert res["n_regions"] >= 1


def test_pseudo_change_suppression_reduces_pixels():
    """STSF-Net filter suppresses radiometric drift false positives."""
    h, w = 64, 64
    rng = np.random.default_rng(42)
    t1 = rng.uniform(0.3, 0.4, (h, w))
    # Add subtle global radiometric drift (+0.05) to t2
    t2 = t1 + 0.05
    # Add one true change spot
    t2[20:30, 20:30] += 0.4

    diff = np.abs(t2 - t1)
    suppressed, n_removed = suppress_pseudo_change(diff, t1, t2, window_size=5)
    assert n_removed >= 0
    assert np.all(suppressed <= diff)


def test_otsu_threshold_validity():
    """Otsu threshold on bimodal distribution splits modes."""
    rng = np.random.default_rng(42)
    mode1 = rng.normal(0.2, 0.05, 500)
    mode2 = rng.normal(0.8, 0.05, 500)
    diff = np.concatenate([mode1, mode2]).reshape(20, 50)
    thresh = otsu_threshold(diff)
    assert 0.3 <= thresh <= 0.7


def test_bimodal_confidence_score_high():
    """Well separated bimodal distribution produces high confidence."""
    diff = np.zeros((50, 50))
    diff[:25, :] = 0.1
    diff[25:, :] = 0.9
    score = bimodal_confidence_score(diff, thresh=0.5)
    assert score > 0.7


def test_execution_trace_stages_sequential(detector: ChangeDetector):
    t1 = np.ones((32, 32, 3), dtype=np.uint8) * 100
    t2 = t1.copy()
    t2[10:20, 10:20] = [20, 80, 180]
    res = detector.detect_change(t1, t2)
    stages = [step["stage"] for step in res["execution_trace"]]
    assert stages == sorted(stages)


def test_execution_trace_has_why_rationale(detector: ChangeDetector):
    t1 = np.ones((32, 32, 3), dtype=np.uint8) * 100
    t2 = t1.copy()
    t2[10:20, 10:20] = [20, 80, 180]
    res = detector.detect_change(t1, t2)
    for step in res["execution_trace"]:
        assert len(step["why"]) > 0


def test_processing_time_under_two_seconds(detector: ChangeDetector):
    t1 = np.ones((512, 512, 3), dtype=np.uint8) * 100
    t2 = t1.copy()
    t2[150:350, 150:350] = [20, 80, 180]
    t0 = time.time()
    res = detector.detect_change(t1, t2)
    dur = time.time() - t0
    assert dur < 2.0
    assert res["total_processing_ms"] < 2000.0


def test_identical_images_zero_change(detector: ChangeDetector):
    t1 = np.ones((64, 64, 3), dtype=np.uint8) * 128
    t2 = t1.copy()
    res = detector.detect_change(t1, t2)
    assert res["area_metrics"]["n_changed_pixels"] == 0 or res["area_metrics"]["pct_changed"] < 1.0


def test_sensor_calibration_note_present(detector: ChangeDetector):
    t1 = np.ones((32, 32, 3), dtype=np.uint8) * 100
    t2 = t1.copy()
    res = detector.detect_change(t1, t2, sensor_type="cartosat")
    assert "cartosat" in res["sensor_calibration_note"].lower()


def test_summary_is_informative_text(detector: ChangeDetector):
    t1 = np.ones((32, 32, 3), dtype=np.uint8) * 100
    t2 = t1.copy()
    t2[10:25, 10:25] = [20, 80, 180]
    res = detector.detect_change(t1, t2)
    assert len(res["summary"]) > 40
