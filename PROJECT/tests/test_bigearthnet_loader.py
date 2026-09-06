"""
Unit Tests for BigEarthNet.txt Data Loader and Real-Data API Endpoints.
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from backend.main import app
from backend.core.bigearthnet_loader import BigEarthNetRealDataLoader, calculate_iou, ALL_TASKS
from ml_models.rs_internvl import RSInternVL


client = TestClient(app)


def test_iou_calculation():
    boxA = [0.0, 0.0, 1.0, 1.0]
    boxB = [0.0, 0.0, 1.0, 1.0]
    assert calculate_iou(boxA, boxB) == 1.0

    boxC = [0.0, 0.0, 0.5, 0.5]
    assert 0.20 <= calculate_iou(boxA, boxC) <= 0.30


def test_real_data_loader_sample():
    loader = BigEarthNetRealDataLoader()
    sample = loader.get_real_sample()
    assert sample["patch_id"] == "S2A_MSIL2A_20170818T103021_N9999_R108_T32TMT_61_44"
    assert sample["country"] == "Switzerland"
    assert "captioning" in sample["tasks"]
    assert "referring_lulc_detection" in sample["tasks"]


def test_real_data_loader_load_tensors():
    loader = BigEarthNetRealDataLoader()
    s1, s2, rgb = loader.load_tensors_for_patch(img_size=120)
    assert s1.shape == (1, 2, 120, 120)
    assert s2.shape == (1, 10, 120, 120)
    assert rgb.shape == (1, 3, 120, 120)


def test_evaluate_task_real_sample():
    loader = BigEarthNetRealDataLoader()
    model = RSInternVL(vit_dim=64, llm_dim=128, img_size=120)
    res = loader.evaluate_task(model, "binary_presence")
    assert "model_prediction" in res
    assert res["ground_truth"] == "No"


def test_api_bigearthnet_tasks():
    resp = client.get("/api/bigearthnet/tasks")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_tasks"] == 15
    assert "captioning" in data["tasks"]


def test_api_bigearthnet_sample():
    resp = client.get("/api/bigearthnet/sample")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["country"] == "Switzerland"


def test_api_bigearthnet_infer():
    resp = client.post("/api/bigearthnet/infer", data={"task": "captioning"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "result" in data


def test_api_bigearthnet_benchmark_results():
    resp = client.get("/api/bigearthnet/benchmark-results")
    assert resp.status_code == 200
    data = resp.json()
    assert "sota_comparison" in data
