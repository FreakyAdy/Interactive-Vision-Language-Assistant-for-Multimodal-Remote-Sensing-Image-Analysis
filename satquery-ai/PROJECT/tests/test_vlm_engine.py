"""
Unit tests for SatQuery AI VLM Engine.
"""

import numpy as np
import pytest
from core.vlm_engine import VLMEngine, generate_answer, is_model_loaded


@pytest.fixture
def vlm() -> VLMEngine:
    return VLMEngine()


def test_is_model_loaded(vlm: VLMEngine):
    assert vlm.is_loaded() is True


def test_vlm_demo_mode_flood(vlm: VLMEngine):
    img = np.zeros((64, 64, 3), dtype=np.float32)
    res = vlm.generate_answer(img, "How much has the flood spread?", {"sensor": "Cartosat-2S"})
    assert "flood" in res["answer"].lower() or "inundation" in res["answer"].lower()
    assert res["confidence"] > 0.8
    assert len(res["reasoning_steps"]) > 0
    assert len(res["recommended_actions"]) > 0


def test_vlm_demo_mode_vegetation(vlm: VLMEngine):
    img = np.zeros((64, 64, 3), dtype=np.float32)
    res = vlm.generate_answer(img, "Assess the forest vegetation canopy health", {"sensor": "ResourceSat-2A"})
    assert "vegetation" in res["answer"].lower() or "canopy" in res["answer"].lower()
    assert res["confidence"] > 0.8


def test_vlm_demo_mode_building(vlm: VLMEngine):
    img = np.zeros((64, 64, 3), dtype=np.float32)
    res = vlm.generate_answer(img, "Count how many buildings are in this sector", {"sensor": "Cartosat-3"})
    assert len(res["reasoning_steps"]) >= 1
    assert res["confidence"] > 0.7


def test_vlm_batch_inference(vlm: VLMEngine):
    img1 = np.zeros((32, 32, 3), dtype=np.float32)
    img2 = np.ones((32, 32, 3), dtype=np.float32)
    queries = ["Check flood area", "Assess crop vigor"]
    results = vlm.batch_analyze([img1, img2], queries)
    assert len(results) == 2
    assert "answer" in results[0]
    assert "answer" in results[1]
