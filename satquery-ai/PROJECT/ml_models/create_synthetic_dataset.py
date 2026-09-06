"""
SatQuery AI — Synthetic Instruction Dataset Generator for ISRO Satellite VLM.

Generates paired (Image, Query, Target Response, Grounding Bounding Boxes)
JSON Lines instruction-tuning dataset mimicking ISRO satellite scenarios:
- Cartosat-2S/3 optical queries
- RISAT-1C SAR queries
- ResourceSat-2A multispectral agricultural queries
- Bi-temporal change detection instructions
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("satquery.dataset_gen")

OUTPUT_PATH = Path(__file__).resolve().parent / "isro_synthetic_instruct_dataset.jsonl"

TEMPLATES = [
    {
        "sensor": "Cartosat-2S",
        "task": "disaster_assessment",
        "query": "Assess the inundation extent along this river corridor following heavy precipitation.",
        "answer": "Analysis reveals 11.93 hectares of flood inundation spanning two low-lying sectors. Surface water NDWI values peaked at 0.72.",
        "regions": [{"class": "flood_water", "bbox": [120, 180, 360, 420]}],
    },
    {
        "sensor": "ResourceSat-2A",
        "task": "vegetation_monitoring",
        "query": "Evaluate canopy density and detect any signs of illegal logging or clearance in this reserve.",
        "answer": "Vegetation index indicates dense canopy across 82% of the sector, with an active 4.2-hectare clearance corridor detected in the north-east quadrant.",
        "regions": [{"class": "cleared_canopy", "bbox": [210, 150, 410, 310]}],
    },
    {
        "sensor": "RISAT-1C",
        "task": "maritime_surveillance",
        "query": "Identify and count vessels approaching the designated anchorage zone through heavy cloud cover.",
        "answer": "Synthetic Aperture Radar backscatter analysis identified 5 maritime vessels with high specular contrast (> -8 dB) against sea surface background.",
        "regions": [
            {"class": "vessel", "bbox": [100, 160, 140, 200]},
            {"class": "vessel", "bbox": [150, 220, 190, 260]},
            {"class": "vessel", "bbox": [200, 110, 240, 150]},
            {"class": "vessel", "bbox": [300, 290, 340, 330]},
            {"class": "vessel", "bbox": [370, 320, 410, 360]},
        ],
    },
    {
        "sensor": "Cartosat-3",
        "task": "urban_expansion",
        "query": "Detect new built-up construction on agricultural parcels along this peri-urban boundary.",
        "answer": "NDBI anomaly detection confirms 2.8 hectares of new impervious structures with high surface reflectance.",
        "regions": [{"class": "new_built_up", "bbox": [280, 250, 460, 430]}],
    },
]


def generate_synthetic_dataset(num_samples: int = 100, output_file: Path = OUTPUT_PATH) -> None:
    """Generate synthetic instruction-tuning dataset."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    samples: List[Dict[str, Any]] = []

    for i in range(num_samples):
        base = TEMPLATES[i % len(TEMPLATES)]
        sample = {
            "id": f"isro_vlm_{i+1:04d}",
            "sensor": base["sensor"],
            "task_type": base["task"],
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n[Sensor: {base['sensor']}] {base['query']}",
                },
                {
                    "from": "gpt",
                    "value": base["answer"],
                },
            ],
            "grounding": base["regions"],
        }
        samples.append(sample)

    with open(output_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    logger.info("Generated %d synthetic instruction samples to %s", len(samples), output_file)


if __name__ == "__main__":
    generate_synthetic_dataset()
