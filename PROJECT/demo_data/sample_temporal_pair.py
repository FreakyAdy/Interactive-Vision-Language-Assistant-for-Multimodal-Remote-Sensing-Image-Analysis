"""
SatQuery AI — Sample Temporal Image Pair Generator.

Generates a co-registered T1 (Before) and T2 (After) image pair
demonstrating seasonal / environmental change for bi-temporal testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image

OUTPUT_DIR = Path(__file__).resolve().parent


def generate_temporal_pair(
    size: int = 512,
    change_type: str = "water_expansion",
) -> Tuple[Path, Path]:
    """
    Generate co-registered T1 and T2 images.
    change_type: 'water_expansion', 'vegetation_loss', or 'urban_growth'.
    """
    rng = np.random.default_rng(303)
    p1 = OUTPUT_DIR / "temporal_t1.png"
    p2 = OUTPUT_DIR / "temporal_t2.png"

    yy, xx = np.mgrid[:size, :size]

    # Common background terrain
    t1 = np.zeros((size, size, 3), dtype=np.uint8)
    t1[:, :, 0] = rng.integers(70, 90, size=(size, size), dtype=np.uint8)
    t1[:, :, 1] = rng.integers(130, 165, size=(size, size), dtype=np.uint8)
    t1[:, :, 2] = rng.integers(60, 80, size=(size, size), dtype=np.uint8)

    t2 = t1.copy()

    if change_type == "water_expansion":
        # Add reservoir in T1
        lake_t1 = ((xx - 0.4 * size) ** 2 + (yy - 0.5 * size) ** 2) < (0.12 * size) ** 2
        t1[lake_t1] = [25, 75, 160]

        # Expand reservoir in T2
        lake_t2 = ((xx - 0.4 * size) ** 2 + (yy - 0.5 * size) ** 2) < (0.28 * size) ** 2
        t2[lake_t2] = [35, 90, 175]

    elif change_type == "vegetation_loss":
        # Dense forest in T1
        forest_zone = (xx > 0.2 * size) & (xx < 0.8 * size) & (yy > 0.2 * size) & (yy < 0.8 * size)
        t1[forest_zone] = [20, 140, 40]

        # Deforestation patch in T2
        t2[forest_zone] = [20, 140, 40]
        cleared_patch = (xx > 0.35 * size) & (xx < 0.65 * size) & (yy > 0.3 * size) & (yy < 0.7 * size)
        t2[cleared_patch] = [170, 130, 85]

    else:  # urban_growth
        # Farmland in T1
        t1[:, :] = [110, 160, 90]
        t2 = t1.copy()
        # Concrete development in T2
        urban_zone = (xx > 0.5 * size) & (yy > 0.4 * size)
        t2[urban_zone] = [195, 190, 205]

    Image.fromarray(t1).save(p1)
    Image.fromarray(t2).save(p2)
    return p1, p2


if __name__ == "__main__":
    p1, p2 = generate_temporal_pair()
    print(f"Generated temporal pair: {p1} and {p2}")
