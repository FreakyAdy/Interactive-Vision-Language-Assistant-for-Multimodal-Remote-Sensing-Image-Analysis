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
    assert res["task_type"] in ["change_detection", "bitemporal_change_analysis"]
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
    assert res["task_type"] in ["scene_classification", "single_image_caption_grounding"]


def test_route_sih_rep_query_1(router: QueryRouter):
    res = router.route("Describe the land-cover and major objects visible in this image.")
    assert res["task_type"] in ["single_image_caption_grounding", "scene_classification"]


def test_route_sih_rep_query_2(router: QueryRouter):
    res = router.route("Highlight the water body referred to in the query.")
    assert res["task_type"] == "single_image_caption_grounding"


def test_route_sih_rep_query_3(router: QueryRouter):
    res = router.route("What changed between these two dates, and where did the change occur?")
    assert res["task_type"] in ["bitemporal_change_analysis", "change_detection"]
    assert res["requires_temporal"] is True


def test_route_sih_rep_query_4(router: QueryRouter):
    res = router.route("Use the optical and SAR images together to identify built-up and water-covered regions.")
    assert res["task_type"] == "cross_modal_fusion"


def test_route_sih_rep_query_5(router: QueryRouter):
    res = router.route("Has the built-up area increased, decreased, or remained unchanged?")
    assert res["task_type"] == "bitemporal_cdvqa"


def test_auditable_trace_generation():
    trace = QueryRouter.build_auditable_execution_trace(
        selected_task="cross_modal_fusion",
        input_scope="CROSS_MODAL_PAIR",
        invoked_models_tools=["OpticalSARFusionEngine", "SpectralIndices"],
        permitted_parameters={"sar_weight": 0.5},
        outputs_summary={"water_pct": 18.4},
        confidence_metrics={"multimodal_confidence": 0.94},
        latency_ms=45.2
    )
    assert trace["compliance_status"] == "STRICT_SIH26167_COMPLIANT"
    assert "selected_task" in trace
    assert "orchestrated_models_and_tools" in trace
    assert "configured_permitted_parameters" in trace
    assert "observable_outputs" in trace
    assert trace["total_execution_latency_ms"] == 45.2

