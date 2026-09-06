"""
Test Suite for OpticalSARFusionEngine
=====================================
Validates cross-modal joint information extraction as required by SIH26167.
"""

import numpy as np
import pytest
from backend.core.optical_sar_fusion import OpticalSARFusionEngine


def test_optical_sar_fusion_initialization():
    engine = OpticalSARFusionEngine(sar_weight=0.6)
    assert engine.sar_weight == 0.6


def test_optical_sar_fusion_joint_extraction():
    engine = OpticalSARFusionEngine(sar_weight=0.5)

    H, W = 100, 100
    # Optical: vegetation background + water patch
    optical = np.ones((H, W, 3), dtype=np.float32) * 0.4
    optical[20:60, 20:60] = [0.1, 0.4, 0.9]  # Water (low red, high blue)

    # SAR: specular reflection in water patch (dark) + bright double bounce urban
    sar = np.ones((H, W), dtype=np.float32) * 0.35
    sar[20:60, 20:60] = 0.05  # Specular water
    sar[70:90, 70:90] = 0.85  # Bright built-up

    res = engine.extract_joint_features(optical, sar)

    assert "water_mask" in res
    assert "built_up_mask" in res
    assert "metrics" in res
    assert "fusion_confidence" in res
    assert "explanation" in res
    assert res["metrics"]["water_percentage"] > 5.0
    assert res["metrics"]["built_up_percentage"] > 2.0
    assert 0.0 <= res["fusion_confidence"] <= 1.0


def test_optical_sar_fusion_cloud_shadow_disambiguation():
    """
    Ensures that dark cloud shadows in optical imagery are NOT classified as water
    when SAR reveals normal terrestrial roughness.
    """
    engine = OpticalSARFusionEngine(sar_weight=0.5)

    H, W = 64, 64
    optical = np.ones((H, W, 3), dtype=np.float32) * 0.5
    # Simulate dark cloud shadow
    optical[10:30, 10:30] = [0.05, 0.05, 0.05]

    # SAR: penetrates cloud shadow, normal terrain backscatter (not specular low)
    sar = np.ones((H, W), dtype=np.float32) * 0.45

    res = engine.extract_joint_features(optical, sar)
    # Cloud shadow area should not be classified as water
    water_mask = np.array(res["water_mask"])
    assert np.sum(water_mask[10:30, 10:30]) == 0
