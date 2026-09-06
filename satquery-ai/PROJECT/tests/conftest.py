"""
Pytest configuration and shared fixtures for SatQuery AI.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure backend and PROJECT directories are on sys.path
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TESTS_DIR.parent
BACKEND_DIR = PROJECT_DIR / "backend"

for p in [str(PROJECT_DIR), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    """Returns a synthetic 64x64 RGB uint8 image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_multispectral_image() -> np.ndarray:
    """Returns a synthetic 64x64 4-band (Red, Green, Blue, NIR) float64 image."""
    rng = np.random.default_rng(42)
    return rng.uniform(0.0, 1.0, (64, 64, 4)).astype(np.float64)


@pytest.fixture
def sample_flood_pair() -> tuple[np.ndarray, np.ndarray]:
    """Returns a registered pair where T2 has water expansion."""
    t1 = np.ones((64, 64, 3), dtype=np.uint8) * 100
    t2 = t1.copy()
    t2[20:45, 20:45] = [20, 70, 180]  # water
    return t1, t2
