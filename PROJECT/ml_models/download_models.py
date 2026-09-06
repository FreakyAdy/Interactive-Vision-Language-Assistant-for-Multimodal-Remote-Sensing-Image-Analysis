"""
SatQuery AI — Model Weights Downloader.

Downloads base weights for GeoChat-7B and SentenceTransformers from HuggingFace Hub.
Supports 4-bit quantization and local directory caching.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("satquery.download_models")

DEFAULT_MODELS = [
    {"repo_id": "MBZUAI/geochat-7B", "type": "vlm", "optional": True},
    {"repo_id": "sentence-transformers/all-MiniLM-L6-v2", "type": "embeddings", "optional": False},
]


def download_models(cache_dir: Path | None = None, include_vlm: bool = False) -> None:
    """Download required weights with optional VLM skip for low-disk environments."""
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent / "weights"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub is not installed. Install via `pip install huggingface_hub`.")
        sys.exit(1)

    logger.info("Downloading models to cache directory: %s", cache_dir)

    for m in DEFAULT_MODELS:
        repo = m["repo_id"]
        if m["type"] == "vlm" and not include_vlm:
            logger.info("Skipping heavy VLM (%s) — use --include-vlm to download 14GB weights.", repo)
            continue

        logger.info("Downloading %s (%s)...", repo, m["type"])
        try:
            path = snapshot_download(repo_id=repo, cache_dir=str(cache_dir))
            logger.info("Successfully fetched %s -> %s", repo, path)
        except Exception as exc:
            logger.warning("Failed to download %s: %s", repo, exc)
            if not m["optional"]:
                raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SatQuery AI ML Model Weights")
    parser.add_argument("--include-vlm", action="store_true", help="Download full 7B VLM weights (~14GB)")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Custom weights directory")
    args = parser.parse_args()

    download_models(cache_dir=args.cache_dir, include_vlm=args.include_vlm)
