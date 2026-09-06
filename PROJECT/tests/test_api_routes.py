"""
FastAPI route tests for SatQuery AI backend using TestClient.
"""

import io
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app

client = TestClient(app)


def _create_test_image_bytes(w: int = 64, h: int = 64, color: tuple = (100, 150, 80)) -> bytes:
    """Helper to produce in-memory PNG bytes."""
    img = Image.new("RGB", (w, h), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert data["problem_statement"] == "SIH26167"
    assert "supported_sensors" in data
    assert len(data["supported_sensors"]) >= 4


def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "endpoints" in data
    assert data["title"].startswith("SatQuery AI")


def test_analyze_valid_image():
    img_bytes = _create_test_image_bytes()
    resp = client.post(
        "/api/analyze",
        files={"image": ("test.png", img_bytes, "image/png")},
        data={"query": "What is the vegetation density in this scene?", "sensor_type": "cartosat"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "answer" in data
    assert data["confidence"] > 0.0
    assert "spectral_indices" in data
    assert "execution_trace" in data


def test_analyze_empty_file_fails():
    resp = client.post(
        "/api/analyze",
        files={"image": ("empty.png", b"", "image/png")},
        data={"query": "Test query"},
    )
    assert resp.status_code == 422


def test_analyze_missing_query_fails():
    img_bytes = _create_test_image_bytes()
    resp = client.post(
        "/api/analyze",
        files={"image": ("test.png", img_bytes, "image/png")},
        data={"query": "   "},
    )
    assert resp.status_code == 422


def test_change_detection_route():
    b1 = _create_test_image_bytes(color=(100, 150, 80))
    b2 = _create_test_image_bytes(color=(30, 80, 180))
    resp = client.post(
        "/api/change-detection",
        files={
            "image_t1": ("t1.png", b1, "image/png"),
            "image_t2": ("t2.png", b2, "image/png"),
        },
        data={
            "query": "How much has the water level risen?",
            "sensor_type": "cartosat",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "area_metrics" in data
    assert "change_direction" in data
    assert "execution_trace" in data


def test_spectral_indices_route():
    img_bytes = _create_test_image_bytes()
    resp = client.post(
        "/api/spectral-indices",
        files={"image": ("test.png", img_bytes, "image/png")},
        data={"sensor_type": "resourcesat"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "indices" in data
    assert "ndvi" in data["indices"]


def test_demo_scenarios_list():
    resp = client.get("/api/demo/scenarios")
    assert resp.status_code == 200
    scenarios = resp.json()
    assert len(scenarios) == 5
    ids = [s["id"] for s in scenarios]
    assert "flood_kerala_2023" in ids
    assert "deforestation_assam" in ids


def test_demo_run_scenario():
    resp = client.post("/api/demo/run/flood_kerala_2023")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "scenario" in data


def test_demo_run_nonexistent_fails():
    resp = client.post("/api/demo/run/nonexistent_scenario_xyz")
    assert resp.status_code == 404


def test_api_latency_under_five_seconds():
    t0 = time.time()
    resp = client.get("/api/demo/scenarios")
    elapsed = time.time() - t0
    assert resp.status_code == 200
    assert elapsed < 5.0


def test_compatibility_check_api():
    b1 = _create_test_image_bytes()
    resp = client.post(
        "/api/compatibility-check",
        files={"image1": ("optical.png", b1, "image/png")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_compatible"] is True
    assert data["scope"] == "SINGLE_IMAGE"


def test_cross_modal_analysis_api():
    b_opt = _create_test_image_bytes(color=(100, 160, 80))
    b_sar = _create_test_image_bytes(color=(120, 120, 120))
    resp = client.post(
        "/api/cross-modal-analysis",
        files={
            "optical_image": ("optical.png", b_opt, "image/png"),
            "sar_image": ("sar.png", b_sar, "image/png"),
        },
        data={"query": "Use the optical and SAR images together to identify built-up and water-covered regions."}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["input_scope"] == "CROSS_MODAL_PAIR"
    assert "metrics" in data
    assert "auditable_summary" in data
    assert data["auditable_summary"]["compliance_status"] == "STRICT_SIH26167_COMPLIANT"


def test_cdvqa_api():
    b1 = _create_test_image_bytes(color=(110, 150, 90))
    b2 = _create_test_image_bytes(color=(190, 190, 200))
    resp = client.post(
        "/api/cdvqa",
        files={
            "t1_image": ("t1.png", b1, "image/png"),
            "t2_image": ("t2.png", b2, "image/png"),
        },
        data={"query": "Has the built-up area increased, decreased, or remained unchanged?"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["input_scope"] == "BITEMPORAL_PAIR"
    assert "categorical_answer" in data
    assert "auditable_summary" in data

