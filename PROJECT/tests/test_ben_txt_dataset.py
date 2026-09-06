"""
Unit tests for BigEarthNet.txt Parquet Dataset & DataLoader integration (arXiv:2603.29630).
"""

import os
import pytest
import torch
from torch.utils.data import DataLoader
from ml_models.ben_txt_dataset import BigEarthNetTxtParquetDataset


def test_ben_txt_dataset_init():
    """Verify BigEarthNetTxtParquetDataset instantiates with real parquet file."""
    parquet_path = os.path.join(
        os.path.dirname(__file__), "../demo_data/BigEarthNet.txt.parquet"
    )
    if not os.path.exists(parquet_path):
        pytest.skip("BigEarthNet.txt.parquet not present")

    ds = BigEarthNetTxtParquetDataset(parquet_path=parquet_path, max_samples=20)
    assert len(ds) == 20
    assert len(ds.categories) > 0
    assert len(ds.task_types) > 0


def test_ben_txt_dataset_getitem():
    """Verify getitem returns valid multi-sensor rasters and annotations."""
    parquet_path = os.path.join(
        os.path.dirname(__file__), "../demo_data/BigEarthNet.txt.parquet"
    )
    if not os.path.exists(parquet_path):
        pytest.skip("BigEarthNet.txt.parquet not present")

    ds = BigEarthNetTxtParquetDataset(parquet_path=parquet_path, max_samples=5)
    item = ds[0]

    assert "patch_id" in item
    assert "s1_tensor" in item
    assert "s2_tensor" in item
    assert "rgb_tensor" in item
    assert item["s1_tensor"].shape == (1, 2, 120, 120)
    assert item["s2_tensor"].shape == (1, 10, 120, 120)
    assert item["rgb_tensor"].shape == (1, 3, 120, 120)
    assert len(item["input"]) > 0
    assert len(item["output"]) > 0


def test_ben_txt_dataset_dataloader():
    """Verify PyTorch DataLoader integration and batch collation."""
    parquet_path = os.path.join(
        os.path.dirname(__file__), "../demo_data/BigEarthNet.txt.parquet"
    )
    if not os.path.exists(parquet_path):
        pytest.skip("BigEarthNet.txt.parquet not present")

    ds = BigEarthNetTxtParquetDataset(parquet_path=parquet_path, max_samples=4)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    batch = next(iter(loader))
    assert len(batch["patch_id"]) == 2
    assert batch["s1_tensor"].shape == (2, 1, 2, 120, 120)
    assert batch["s2_tensor"].shape == (2, 1, 10, 120, 120)
    assert batch["rgb_tensor"].shape == (2, 1, 3, 120, 120)


def test_ben_txt_dataset_summary():
    """Verify benchmark summary report generation."""
    ds = BigEarthNetTxtParquetDataset(max_samples=10)
    summary = ds.get_benchmark_summary()
    assert "total_samples" in summary
    assert "task_distribution" in summary
    assert summary["total_samples"] >= 1
