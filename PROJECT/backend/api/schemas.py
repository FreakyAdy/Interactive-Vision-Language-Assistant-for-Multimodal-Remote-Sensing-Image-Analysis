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
    stage: int = Field(1, description="Stage sequence index")
    name: str = Field(..., description="Stage name")
    tool: str = Field(..., description="Underlying algorithm or tool")
    observation: str = Field(..., description="Output observation or intermediate metric")
    duration_ms: float = Field(..., description="Execution time in milliseconds")
    why: str = Field("", description="Scientific or operational rationale")
    step: Optional[int] = Field(None, description="Step sequence index")
    task: Optional[str] = Field(None, description="Task name")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Tool parameters")
    output: Optional[str] = Field(None, description="Output description")


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


class AuditableExecutionSummary(BaseModel):
    auditable_trace_version: str = Field("SIH26167-v1.0", description="Specification version")
    selected_task: Dict[str, Any] = Field(..., description="Selected task classification & SIH ID")
    input_compatibility: Dict[str, Any] = Field(..., description="Observable input validation & format verification")
    orchestrated_models_and_tools: List[str] = Field(..., description="Registered models and tools executed")
    configured_permitted_parameters: Dict[str, Any] = Field(..., description="Task-specific permitted parameters configured")
    observable_outputs: Dict[str, Any] = Field(..., description="Textual and spatial deliverables")
    confidence_metrics: Dict[str, float] = Field(..., description="Quantitative confidence scores")
    total_execution_latency_ms: float = Field(..., description="Total execution time in ms")
    compliance_status: str = Field("STRICT_SIH26167_COMPLIANT", description="Compliance status")


class CompatibilityCheckResponse(BaseModel):
    is_compatible: bool = Field(..., description="Whether input images meet scope requirements")
    scope: str = Field(..., description="Detected scope: SINGLE_IMAGE, CROSS_MODAL_PAIR, BITEMPORAL_PAIR")
    message: str = Field(..., description="Compatibility summary message")
    errors: List[str] = Field(default_factory=list, description="Validation errors if any")
    image_metas: List[Dict[str, Any]] = Field(default_factory=list, description="Metadata of inspected images")
    auditable_compatibility_trace: Dict[str, Any] = Field(default_factory=dict, description="Observable trace")


class CrossModalAnalysisRequest(BaseModel):
    query: Optional[str] = Field(
        "Use the optical and SAR images together to identify built-up and water-covered regions.",
        description="Natural language instruction for cross-modal analysis"
    )
    sar_weight: float = Field(0.5, ge=0.0, le=1.0, description="Weight of SAR radar backscatter vs Optical [0, 1]")
    cloud_suppression: bool = Field(True, description="Enable cloud shadow rejection using radar roughness")


class CrossModalAnalysisResponse(BaseModel):
    status: str = Field("ok", description="Status code")
    input_scope: str = Field("CROSS_MODAL_PAIR", description="Verified input scope")
    query: str = Field(..., description="Evaluated query")
    metrics: Dict[str, Any] = Field(..., description="Water, built-up, and vegetation percentages")
    confidence: float = Field(..., description="Multimodal fusion confidence [0, 1]")
    confidence_label: str = Field("HIGH", description="Confidence label")
    explanation: str = Field(..., description="Evidence-grounded explanation")
    modality_synergy: Dict[str, Any] = Field(..., description="SAR vs Optical physical contributions")
    annotated_image: Optional[str] = Field(None, description="Base64 encoded false-color composite")
    auditable_summary: AuditableExecutionSummary = Field(..., description="Observable SIH execution trace")
    total_processing_ms: float = Field(..., description="Turnaround time in ms")


class CDVQARequest(BaseModel):
    query: Optional[str] = Field(
        "Has the built-up area increased, decreased, or remained unchanged?",
        description="Temporal change question"
    )
    target_category: Optional[str] = Field("built_up", description="Category of interest: built_up, vegetation, water")


class CDVQAResponse(BaseModel):
    status: str = Field("ok", description="Status code")
    input_scope: str = Field("BITEMPORAL_PAIR", description="Verified input scope")
    query: str = Field(..., description="User question")
    categorical_answer: str = Field(..., description="INCREASED, DECREASED, or UNCHANGED")
    quantitative_delta: str = Field(..., description="Measured quantitative difference (e.g. +3.45 ha, +14.2%)")
    explanation: str = Field(..., description="Evidence-grounded physical explanation")
    confidence: float = Field(..., description="Bimodal confidence score [0, 1]")
    annotated_image: Optional[str] = Field(None, description="Base64 difference visualization")
    auditable_summary: AuditableExecutionSummary = Field(..., description="Observable SIH execution trace")
    total_processing_ms: float = Field(..., description="Turnaround time in ms")

