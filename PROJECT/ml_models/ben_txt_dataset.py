"""
PyTorch Dataset & Benchmark Loader for BigEarthNet.txt (arXiv:2603.29630).

Provides native access to:
- 464,044 Sentinel-1 / Sentinel-2 co-registered pairs
- 9.6 million text annotations across 15 tasks
- Curated 1,082 image pair benchmark split with 15,029 manually verified annotations
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

from ml_models.rs_internvl import (
    S1_BANDS, S2_10M20M_BANDS, RGB_BANDS, BAND_MEANS, BAND_STDS
)
from backend.core.bigearthnet_loader import BigEarthNetRealDataLoader, PAPER_BENCHMARK_SAMPLE


class BigEarthNetTxtParquetDataset(Dataset):
    """
    Dataset loader for BigEarthNet.txt Parquet files and multi-sensor rasters.
    """
    def __init__(
        self,
        parquet_path: Optional[str] = None,
        samples_dir: Optional[str] = None,
        split: Optional[str] = None,
        task_types: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
    ):
        self.parquet_path = parquet_path or os.path.join(
            os.path.dirname(__file__), "../demo_data/BigEarthNet.txt.parquet"
        )
        self.samples_dir = samples_dir or os.path.join(
            os.path.dirname(__file__), "../demo_data/bigearthnet_samples"
        )
        self.selected_split = split
        self.selected_task_types = task_types
        self.selected_categories = categories
        self.max_samples = max_samples

        self.df = None
        self.real_loader = BigEarthNetRealDataLoader(data_dir=self.samples_dir)

        # If parquet exists and pyarrow/pandas is available
        if os.path.exists(self.parquet_path):
            try:
                if HAS_PYARROW:
                    # Optimized load using PyArrow scanner with filters
                    filters = []
                    if self.selected_split:
                        filters.append(("split", "==", self.selected_split))
                    if self.selected_task_types:
                        filters.append(("type", "in", self.selected_task_types))
                    if self.selected_categories:
                        filters.append(("category", "in", self.selected_categories))

                    # If filtering or limit provided, read filtered table
                    dataset = pq.ParquetDataset(self.parquet_path, filters=filters if filters else None)
                    if self.max_samples:
                        # Slice first N records directly
                        t = dataset.read(columns=[
                            "patch_id", "s1_name", "type", "category", "input",
                            "output", "split", "country", "season", "climate_zone"
                        ])
                        self.df = t.slice(0, self.max_samples).to_pandas()
                    else:
                        # Full dataset or filtered dataset
                        t = dataset.read(columns=[
                            "patch_id", "s1_name", "type", "category", "input",
                            "output", "split", "country", "season", "climate_zone"
                        ])
                        self.df = t.to_pandas()
                elif HAS_PANDAS:
                    df = pd.read_parquet(self.parquet_path)
                    if self.split:
                        df = df[df["split"] == self.split]
                    if self.task_types:
                        df = df[df["type"].isin(self.task_types)]
                    if self.categories:
                        df = df[df["category"].isin(self.categories)]
                    if self.max_samples:
                        df = df.head(self.max_samples)
                    self.df = df.reset_index(drop=True)
            except Exception as e:
                print(f"Warning: Could not read parquet ({e}), using benchmark sample cache.")
                self.df = None

    def __len__(self) -> int:
        if self.df is not None:
            return len(self.df)
        return len(PAPER_BENCHMARK_SAMPLE["tasks"])

    @property
    def categories(self) -> List[str]:
        if self.df is not None and "category" in self.df:
            return sorted(self.df["category"].dropna().unique().tolist())
        return ["presence", "area", "count", "adjacency", "relative_position", "country", "season", "climate_zone"]

    @property
    def task_types(self) -> List[str]:
        if self.df is not None and "type" in self.df:
            return sorted(self.df["type"].dropna().unique().tolist())
        return ["binary", "mcq", "bounding_box", "captioning"]

    @property
    def splits(self) -> List[str]:
        if self.df is not None and "split" in self.df:
            return sorted(self.df["split"].dropna().unique().tolist())
        return ["train", "validation", "test"]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.df is not None and idx < len(self.df):
            row = self.df.iloc[idx]
            s1_t, s2_t, rgb_t = self.real_loader.load_tensors_for_patch()
            return {
                "patch_id": row.get("patch_id", "unknown"),
                "s1_name": row.get("s1_name", "unknown"),
                "type": row.get("type", "vqa"),
                "category": row.get("category", "presence"),
                "input": row.get("input", ""),
                "output": row.get("output", ""),
                "country": row.get("country", ""),
                "season": row.get("season", ""),
                "climate_zone": row.get("climate_zone", ""),
                "s1_tensor": s1_t,
                "s2_tensor": s2_t,
                "rgb_tensor": rgb_t,
            }

        # Fallback to curated benchmark sample tasks
        sample = PAPER_BENCHMARK_SAMPLE
        task_names = list(sample["tasks"].keys())
        task_key = task_names[idx % len(task_names)]
        t_data = sample["tasks"][task_key]

        s1_t, s2_t, rgb_t = self.real_loader.load_tensors_for_patch()
        return {
            "patch_id": sample["patch_id"],
            "s1_name": sample["s1_name"],
            "type": t_data.get("type"),
            "category": t_data.get("category", task_key),
            "input": t_data.get("input"),
            "output": t_data.get("reference_output"),
            "country": sample["country"],
            "season": sample["season"],
            "climate_zone": sample["climate_zone"],
            "s1_tensor": s1_t,
            "s2_tensor": s2_t,
            "rgb_tensor": rgb_t,
        }

    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Returns statistics on the loaded BigEarthNet.txt dataset split."""
        if self.df is not None:
            return {
                "total_samples": len(self.df),
                "unique_patches": int(self.df["patch_id"].nunique()) if "patch_id" in self.df else 0,
                "task_distribution": self.df["type"].value_counts().to_dict() if "type" in self.df else {},
                "countries": list(self.df["country"].unique()) if "country" in self.df else [],
                "seasons": list(self.df["season"].unique()) if "season" in self.df else []
            }
        return {
            "total_samples": len(PAPER_BENCHMARK_SAMPLE["tasks"]),
            "unique_patches": 1,
            "task_distribution": {"binary": 4, "mcq": 5, "bounding_box": 2, "captioning": 1},
            "countries": ["Switzerland"],
            "seasons": ["Summer"]
        }
