"""
SatQuery AI — Sample Optical Satellite Image Generator (Cartosat-like).

Generates a realistic 4-band / 3-band high-resolution optical scene
mimicking ISRO Cartosat-2S / Cartosat-3 imagery over an Indian urban/river landscape.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

OUTPUT_DIR = Path(__file__).resolve().parent


def generate_sample_optical(size: int = 512, output_path: Path | None = None) -> Path:
    """Generate Cartosat-like optical image sample."""
    rng = np.random.default_rng(101)
    if output_path is None:
        output_path = OUTPUT_DIR / "sample_cartosat_optical.png"

    img = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.mgrid[:size, :size]

    # River curve in middle
    river_y = 0.5 * size + 0.15 * size * np.sin(xx / 50.0)
    river_mask = np.abs(yy - river_y) < 25

    # Vegetation zone (upper left)
    veg_mask = (xx < 0.45 * size) & ~river_mask
    img[veg_mask, 0] = rng.integers(40, 70, size=np.sum(veg_mask), dtype=np.uint8)
    img[veg_mask, 1] = rng.integers(130, 185, size=np.sum(veg_mask), dtype=np.uint8)
    img[veg_mask, 2] = rng.integers(35, 65, size=np.sum(veg_mask), dtype=np.uint8)

    # Urban / built-up zone (lower right)
    urban_mask = (xx >= 0.45 * size) & ~river_mask
    img[urban_mask, 0] = rng.integers(160, 210, size=np.sum(urban_mask), dtype=np.uint8)
    img[urban_mask, 1] = rng.integers(160, 205, size=np.sum(urban_mask), dtype=np.uint8)
    img[urban_mask, 2] = rng.integers(170, 215, size=np.sum(urban_mask), dtype=np.uint8)

    # River water
    img[river_mask, 0] = 30
    img[river_mask, 1] = 85
    img[river_mask, 2] = 165

    # Add realistic sensor noise
    noise = rng.normal(0, 3, (size, size, 3)).astype(np.int16)
    img_final = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    Image.fromarray(img_final).save(output_path)
    return output_path


if __name__ == "__main__":
    out = generate_sample_optical()
    print(f"Generated optical sample: {out}")
