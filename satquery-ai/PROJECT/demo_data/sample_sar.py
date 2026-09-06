"""
SatQuery AI — Sample SAR Satellite Image Generator (RISAT-like).

Generates a realistic C-band Synthetic Aperture Radar (SAR) scene
mimicking ISRO RISAT-1C / EOS-04 with multiplicative Rayleigh/Gamma speckle,
rough surface backscatter, and dark specular water reflection.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

OUTPUT_DIR = Path(__file__).resolve().parent


def generate_sample_sar(size: int = 512, output_path: Path | None = None) -> Path:
    """Generate RISAT-like SAR image sample with realistic speckle."""
    rng = np.random.default_rng(202)
    if output_path is None:
        output_path = OUTPUT_DIR / "sample_risat_sar.png"

    # Base reflectivity map
    base = np.zeros((size, size), dtype=np.float32)

    yy, xx = np.mgrid[:size, :size]
    # Rough terrain background: moderate backscatter (~0.4)
    base[:, :] = 0.45

    # Forest patch: high diffuse volume scattering (~0.75)
    forest_mask = (xx > 0.1 * size) & (xx < 0.5 * size) & (yy > 0.1 * size) & (yy < 0.6 * size)
    base[forest_mask] = 0.75

    # Calm water river / lake: specular reflection towards space, very dark (~0.05)
    water_mask = np.abs(yy - (0.7 * size + 0.08 * size * np.sin(xx / 40.0))) < 20
    base[water_mask] = 0.04

    # Point targets (corner reflectors, metallic towers, vessels): very high backscatter (~1.0)
    targets = [(150, 150), (280, 200), (340, 360), (410, 180)]
    for tx, ty in targets:
        base[ty - 2 : ty + 3, tx - 2 : tx + 3] = 1.0

    # Multiplicative SAR speckle (Gamma distribution with 4 looks)
    looks = 4
    speckle = rng.gamma(looks, 1.0 / looks, size=(size, size)).astype(np.float32)
    sar_intensity = base * speckle

    # Convert to 8-bit grayscale
    sar_u8 = np.clip(sar_intensity * 255.0, 0, 255).astype(np.uint8)

    # Save as grayscale PNG and false-color RGB
    Image.fromarray(sar_u8).save(output_path)
    return output_path


if __name__ == "__main__":
    out = generate_sample_sar()
    print(f"Generated SAR sample: {out}")
