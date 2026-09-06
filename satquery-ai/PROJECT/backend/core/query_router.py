"""
SatQuery AI — Query Router Module.

Classifies a user's natural-language query into one of six task types and
determines the appropriate analysis pipeline to invoke.  Uses keyword
matching combined with optional embedding-based semantic similarity
(sentence-transformers ``all-MiniLM-L6-v2``).

Task Types
----------
1. ``scene_classification`` — "What kind of area is this?"
2. ``object_detection``     — "How many buildings/ships/vehicles?"
3. ``change_detection``     — "What changed between these two images?"
4. ``spectral_analysis``    — "What is the vegetation/water/urban index?"
5. ``area_measurement``     — "How large is this lake/forest?"
6. ``disaster_assessment``  — "How much flood damage / burnt area?"
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Attempt to load sentence-transformers for semantic similarity
_EMBEDDER = None
try:
    from sentence_transformers import SentenceTransformer
    _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    logger.info("Sentence-transformers loaded for semantic query routing")
except ImportError:
    logger.info("sentence-transformers not available; using keyword-only routing")
except Exception as exc:
    logger.warning("Could not load embedding model: %s; using keyword-only routing", exc)


# ┌──────────────────────────────────────────────────────────────────────────┐
# │                          Task Definitions                               │
# └──────────────────────────────────────────────────────────────────────────┘

TASK_TYPES: dict[str, dict[str, Any]] = {
    "scene_classification": {
        "label": "Scene Classification",
        "description": "Identify the type of land cover or scene in the image",
        "keywords": [
            "what is this", "identify", "classify", "scene", "land cover",
            "land use", "type of area", "kind of area", "describe",
            "what do you see", "terrain", "landscape",
        ],
        "canonical_query": "What type of land cover or scene is shown in this satellite image?",
        "requires_temporal": False,
        "suggested_indices": [],
        "pipeline_steps": ["image_preprocess", "vlm_inference", "report"],
    },
    "object_detection": {
        "label": "Object Detection",
        "description": "Detect and count objects such as buildings, vehicles, ships, or aircraft",
        "keywords": [
            "count", "how many", "detect", "find", "locate", "building",
            "buildings", "ship", "ships", "vehicle", "vehicles", "car",
            "aircraft", "airplane", "plane", "tank", "bridge", "road",
            "infrastructure", "number of",
        ],
        "canonical_query": "How many buildings or vehicles are visible in this satellite image?",
        "requires_temporal": False,
        "suggested_indices": [],
        "pipeline_steps": ["image_preprocess", "object_detection", "vlm_inference", "report"],
    },
    "change_detection": {
        "label": "Change Detection",
        "description": "Detect what has changed between two temporal images",
        "keywords": [
            "change", "changed", "difference", "before and after", "compare",
            "temporal", "over time", "between", "since", "growth", "shrunk",
            "expanded", "spread", "lost", "gained", "deforestation",
            "encroachment", "two images", "two dates",
        ],
        "canonical_query": "What has changed between these two satellite images taken at different times?",
        "requires_temporal": True,
        "suggested_indices": ["ndvi", "ndwi", "ndbi"],
        "pipeline_steps": [
            "image_preprocess", "coregister", "change_detection",
            "vlm_inference", "report",
        ],
    },
    "spectral_analysis": {
        "label": "Spectral Analysis",
        "description": "Compute and interpret spectral indices (NDVI, NDWI, NDBI, EVI, RVI)",
        "keywords": [
            "vegetation", "ndvi", "water index", "ndwi", "built-up", "ndbi",
            "evi", "rvi", "spectral", "index", "health", "greenness",
            "moisture", "chlorophyll", "biomass", "reflectance",
        ],
        "canonical_query": "What are the vegetation and water index values for this satellite image?",
        "requires_temporal": False,
        "suggested_indices": ["ndvi", "ndwi", "ndbi", "evi"],
        "pipeline_steps": ["image_preprocess", "spectral_computation", "vlm_inference", "report"],
    },
    "area_measurement": {
        "label": "Area Measurement",
        "description": "Measure the area of features like lakes, forests, or fields",
        "keywords": [
            "area", "how large", "how big", "size", "extent", "measure",
            "hectares", "square", "coverage", "boundary", "perimeter",
            "acreage", "spread",
        ],
        "canonical_query": "How large is the lake or forest area visible in this satellite image?",
        "requires_temporal": False,
        "suggested_indices": ["ndvi", "ndwi"],
        "pipeline_steps": ["image_preprocess", "segmentation", "area_calculation", "report"],
    },
    "disaster_assessment": {
        "label": "Disaster Assessment",
        "description": "Assess damage from floods, fires, cyclones, or earthquakes",
        "keywords": [
            "flood", "flooded", "inundation", "damage", "disaster",
            "cyclone", "hurricane", "fire", "burnt", "burned", "wildfire",
            "earthquake", "landslide", "erosion", "relief", "rescue",
            "affected", "impact", "devastation", "submerged",
        ],
        "canonical_query": "How much area has been affected by the flood or fire disaster?",
        "requires_temporal": True,
        "suggested_indices": ["ndwi", "ndvi"],
        "pipeline_steps": [
            "image_preprocess", "change_detection", "spectral_computation",
            "vlm_inference", "report",
        ],
    },
}

# Pre-compute canonical embeddings
_CANONICAL_EMBEDDINGS: dict[str, np.ndarray] | None = None


def _get_canonical_embeddings() -> dict[str, np.ndarray]:
    """Lazily compute embeddings for canonical queries."""
    global _CANONICAL_EMBEDDINGS
    if _CANONICAL_EMBEDDINGS is not None:
        return _CANONICAL_EMBEDDINGS
    if _EMBEDDER is None:
        return {}
    texts = {k: v["canonical_query"] for k, v in TASK_TYPES.items()}
    embeddings = _EMBEDDER.encode(list(texts.values()))
    _CANONICAL_EMBEDDINGS = {k: embeddings[i] for i, k in enumerate(texts.keys())}
    return _CANONICAL_EMBEDDINGS


# ┌──────────────────────────────────────────────────────────────────────────┐
# │                          Query Router                                   │
# └──────────────────────────────────────────────────────────────────────────┘

class QueryRouter:
    """Routes user queries to the appropriate analysis pipeline.

    Combines keyword matching with optional embedding-based semantic
    similarity for robust task classification.
    """

    def __init__(self, keyword_weight: float = 0.6, embedding_weight: float = 0.4) -> None:
        """Initialise the router.

        Args:
            keyword_weight: Weight for keyword-based scoring (0-1).
            embedding_weight: Weight for embedding-based scoring (0-1).
                Ignored if sentence-transformers is unavailable.
        """
        self.keyword_weight = keyword_weight
        self.embedding_weight = embedding_weight

    def route(self, query: str) -> dict[str, Any]:
        """Classify a query and return routing information.

        Args:
            query: The user's natural-language question about a satellite image.

        Returns:
            Dict with keys:
            - ``task_type``: str — one of the six task type identifiers.
            - ``task_label``: str — human-readable label.
            - ``confidence``: float — classification confidence in [0, 1].
            - ``requires_temporal``: bool — whether a second image is needed.
            - ``suggested_indices``: list[str] — recommended spectral indices.
            - ``pipeline_steps``: list[str] — ordered processing stages.
            - ``fallback_used``: bool — True if confidence was too low and
              scene_classification was used as fallback.
        """
        query_lower = query.lower().strip()
        if not query_lower:
            return self._fallback_result("Empty query")

        # ── Keyword scoring ───────────────────────────────────────────
        keyword_scores = self._keyword_score(query_lower)

        # ── Embedding scoring ─────────────────────────────────────────
        embedding_scores = self._embedding_score(query_lower)

        # ── Combined score ────────────────────────────────────────────
        combined: dict[str, float] = {}
        has_embeddings = bool(embedding_scores)

        for task in TASK_TYPES:
            kw = keyword_scores.get(task, 0.0)
            if has_embeddings:
                em = embedding_scores.get(task, 0.0)
                combined[task] = self.keyword_weight * kw + self.embedding_weight * em
            else:
                combined[task] = kw

        best_task = max(combined, key=lambda k: combined[k])
        best_score = combined[best_task]

        # ── Fallback ──────────────────────────────────────────────────
        if best_score < 0.15:
            result = self._fallback_result(query_lower)
            result["all_scores"] = combined
            return result

        info = TASK_TYPES[best_task]
        return {
            "task_type": best_task,
            "task_label": info["label"],
            "confidence": round(min(best_score, 1.0), 3),
            "requires_temporal": info["requires_temporal"],
            "suggested_indices": info["suggested_indices"],
            "pipeline_steps": info["pipeline_steps"],
            "fallback_used": False,
            "all_scores": {k: round(v, 3) for k, v in combined.items()},
        }

    def _keyword_score(self, query_lower: str) -> dict[str, float]:
        """Score each task type by keyword hit count.

        Args:
            query_lower: Lowercased query string.

        Returns:
            Dict mapping task_type → normalised score in [0, 1].
        """
        scores: dict[str, float] = {}
        for task, info in TASK_TYPES.items():
            hits = sum(
                1 for kw in info["keywords"]
                if re.search(rf"\b{re.escape(kw)}\b", query_lower)
            )
            # Normalise: more than 3 keyword hits → confidence 1.0
            scores[task] = min(hits / 3.0, 1.0)
        return scores

    def _embedding_score(self, query_lower: str) -> dict[str, float]:
        """Score each task type by cosine similarity to canonical query.

        Args:
            query_lower: Lowercased query string.

        Returns:
            Dict mapping task_type → similarity score in [0, 1].
            Empty dict if embedder is unavailable.
        """
        if _EMBEDDER is None:
            return {}

        canonical = _get_canonical_embeddings()
        if not canonical:
            return {}

        query_emb = _EMBEDDER.encode([query_lower])[0]
        scores: dict[str, float] = {}
        for task, canon_emb in canonical.items():
            cos_sim = float(
                np.dot(query_emb, canon_emb)
                / (np.linalg.norm(query_emb) * np.linalg.norm(canon_emb) + 1e-10)
            )
            # Map cosine similarity [-1, 1] → [0, 1]
            scores[task] = (cos_sim + 1.0) / 2.0
        return scores

    @staticmethod
    def _fallback_result(reason: str) -> dict[str, Any]:
        """Return a scene_classification fallback result.

        Args:
            reason: Why the fallback was triggered.

        Returns:
            Routing result dict with ``fallback_used=True``.
        """
        info = TASK_TYPES["scene_classification"]
        return {
            "task_type": "scene_classification",
            "task_label": info["label"],
            "confidence": 0.3,
            "requires_temporal": False,
            "suggested_indices": [],
            "pipeline_steps": info["pipeline_steps"],
            "fallback_used": True,
            "fallback_reason": (
                f"I interpreted your query as a general scene description request. "
                f"(Reason: {reason})"
            ),
        }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    router = QueryRouter()

    test_queries = [
        "How much has the flood spread between these two dates?",
        "Count the buildings visible in this area",
        "What is the NDVI for this agricultural field?",
        "How large is this lake?",
        "Assess the cyclone damage in this coastal region",
        "What kind of land cover is this?",
        "Show me something",  # Low-confidence → fallback
    ]

    for q in test_queries:
        result = router.route(q)
        print(f"Q: \"{q}\"")
        print(f"  → {result['task_type']} (conf={result['confidence']:.2f})"
              f" {'[FALLBACK]' if result.get('fallback_used') else ''}")
        print()

    print("QueryRouter OK ✅")
