"""
SatQuery AI — Pydantic Schemas for API Requests and Responses.

Defines the contract for:
- Query analysis (single image + natural language question)
- Bi-temporal change detection (T1 + T2 image comparison)
- Spectral indices extraction (NDVI, NDWI, NDBI, EVI, RVI)
- Health check and system capabilities
- Demo scenarios
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExecutionTraceStep(BaseModel):
    stage: int = Field(..., description="Stage sequence index")
    name: str = Field(..., description="Stage name")
    tool: str = Field(..., description="Underlying algorithm or tool")
    observation: str = Field(..., description="Output observation or intermediate metric")
    duration_ms: float = Field(..., description="Execution time in milliseconds")
    why: str = Field(..., description="Scientific or operational rationale")


class AreaMetrics(BaseModel):
    area_m2: float = Field(..., description="Area in square meters")
    area_ha: float = Field(..., description="Area in hectares")
    area_km2: float = Field(..., description="Area in square kilometers")
    n_changed_pixels: Optional[int] = Field(None, description="Number of changed pixels")
    total_pixels: Optional[int] = Field(None, description="Total pixels evaluated")
    pct_changed: Optional[float] = Field(None, description="Percentage of scene changed")


class AnalysisResponse(BaseModel):
    status: str = Field("ok", description="Status code ('ok' or 'error')")
    task_type: str = Field(..., description="Detected task category")
    answer: str = Field(..., description="Natural language detailed analysis")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    confidence_label: str = Field("HIGH", description="HIGH, MEDIUM, or LOW")
    sensor_badge: str = Field(..., description="Sensor descriptor badge (e.g. 'ISRO Cartosat-2S | 0.65m')")
    reasoning_steps: List[str] = Field(default_factory=list, description="VLM chain of thought reasoning")
    spectral_indices: Dict[str, Any] = Field(default_factory=dict, description="Extracted spectral index metrics")
    annotated_image: Optional[str] = Field(None, description="Base64 encoded PNG of visual annotations")
    geojson: Optional[Dict[str, Any]] = Field(None, description="GeoJSON Feature or FeatureCollection")
    recommended_actions: List[str] = Field(default_factory=list, description="Actionable follow-up recommendations")
    execution_trace: List[ExecutionTraceStep] = Field(default_factory=list, description="12-stage pipeline trace")
    total_processing_ms: float = Field(..., description="Total turnaround time in milliseconds")


class ChangeDetectionRequest(BaseModel):
    query: Optional[str] = Field("What has changed between these two dates?", description="Natural language query")
    t1_timestamp: Optional[str] = Field(None, description="Acquisition date/time for T1 (Before)")
    t2_timestamp: Optional[str] = Field(None, description="Acquisition date/time for T2 (After)")
    sensor_type: Optional[str] = Field("cartosat", description="ISRO sensor identifier")
    suppress_pseudo_change: bool = Field(True, description="Enable STSF-Net pseudo-change suppression")


class ChangeDetectionResponse(BaseModel):
    status: str = Field("ok", description="Status code")
    primary_index: str = Field(..., description="Spectral index used for differencing (e.g. 'ndwi')")
    change_direction: str = Field(..., description="water_expansion, vegetation_loss, urban_growth, etc.")
    confidence: float = Field(..., description="Confidence score 0.0 - 1.0")
    confidence_label: str = Field(..., description="HIGH, MEDIUM, LOW")
    area_metrics: AreaMetrics = Field(..., description="Area quantification in m², ha, km²")
    n_regions: int = Field(..., description="Count of distinct connected change regions")
    otsu_threshold: float = Field(..., description="Calculated Otsu separation threshold")
    n_pseudo_removed: int = Field(..., description="Number of pseudo-change pixels suppressed")
    summary: str = Field(..., description="Structured natural language summary")
    annotated_image: Optional[str] = Field(None, description="Base64 heat/change overlay image")
    geojson: Optional[Dict[str, Any]] = Field(None, description="GeoJSON polygon boundaries")
    execution_trace: List[ExecutionTraceStep] = Field(default_factory=list, description="Detailed stage logs")
    sensor_calibration_note: str = Field(..., description="Sensor calibration parameters applied")
    total_processing_ms: float = Field(..., description="Processing time in milliseconds")


class SpectralIndicesRequest(BaseModel):
    sensor_type: Optional[str] = Field("cartosat", description="Sensor name")
    indices: Optional[List[str]] = Field(None, description="List of indices (ndvi, ndwi, ndbi, evi, rvi)")


class SpectralIndicesResponse(BaseModel):
    status: str = Field("ok", description="Status code")
    sensor_type: str = Field(..., description="Sensor evaluated")
    indices: Dict[str, Any] = Field(..., description="Computed index maps summary stats and interpretations")
    recommendation: str = Field(..., description="Summary ecological/landcover inference")


class DemoScenario(BaseModel):
    id: str = Field(..., description="Scenario unique ID")
    title: str = Field(..., description="Scenario display title")
    description: str = Field(..., description="Scenario background and context")
    sensor: str = Field(..., description="ISRO satellite sensor")
    image_t1: str = Field(..., description="Relative path to T1 image")
    image_t2: Optional[str] = Field(None, description="Relative path to T2 image (if temporal)")
    example_query: str = Field(..., description="Sample user query")
    expected_answer_summary: str = Field(..., description="Expected core finding")
    talking_points: List[str] = Field(default_factory=list, description="Pitch talking points for judges")


class HealthResponse(BaseModel):
    status: str = Field("healthy", description="Overall health status")
    version: str = Field("1.0.0", description="SatQuery AI API version")
    service: str = Field("SatQuery AI — ISRO Remote Sensing Assistant", description="Service title")
    problem_statement: str = Field("SIH26167", description="SIH Problem Statement ID")
    demo_mode: bool = Field(..., description="Whether system is operating in DEMO_MODE")
    model_loaded: str = Field(..., description="Primary VLM model identifier")
    supported_sensors: List[str] = Field(default_factory=list, description="List of calibrated ISRO sensors")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
