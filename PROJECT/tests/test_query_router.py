"""
Unit tests for SatQuery AI Query Routing Engine.
"""

import pytest
from core.query_router import QueryRouter


@pytest.fixture
def router() -> QueryRouter:
    return QueryRouter()


def test_route_scene_classification(router: QueryRouter):
    res = router.route("What type of land cover or area is this?")
    assert res["task_type"] == "scene_classification"
    assert "pipeline_steps" in res


def test_route_object_detection(router: QueryRouter):
    res = router.route("Count the number of ships and boats in the harbor")
    assert res["task_type"] == "object_detection"
    assert "object_detect" in str(res["pipeline_steps"]).lower()


def test_route_change_detection(router: QueryRouter):
    res = router.route("What has changed between these two dates?")
    assert res["task_type"] == "change_detection"
    assert res["requires_temporal"] is True


def test_route_spectral_analysis(router: QueryRouter):
    res = router.route("Compute the NDVI vegetation index for this forest")
    assert res["task_type"] == "spectral_analysis"
    assert "ndvi" in res["suggested_indices"]


def test_route_area_measurement(router: QueryRouter):
    res = router.route("How large is this water reservoir in hectares?")
    assert res["task_type"] == "area_measurement"


def test_route_disaster_assessment(router: QueryRouter):
    res = router.route("Assess the flood inundation and cyclone damage extent")
    assert res["task_type"] == "disaster_assessment"


def test_route_confidence_range(router: QueryRouter):
    res = router.route("Identify agricultural field boundaries")
    assert 0.0 <= res["confidence"] <= 1.0


def test_fallback_on_unclear_query(router: QueryRouter):
    res = router.route("Hello world foo bar baz 12345")
    assert res["task_type"] == "scene_classification"
