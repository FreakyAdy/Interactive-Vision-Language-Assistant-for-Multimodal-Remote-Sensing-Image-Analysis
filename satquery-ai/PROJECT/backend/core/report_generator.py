"""
SatQuery AI — Report Generator Module.

Generates structured natural-language reports from analysis results.
Formats findings for both human readers (Markdown/HTML reports) and
machine consumption (structured JSON).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def generate_report(
    query: str,
    task_type: str,
    vlm_result: dict[str, Any],
    spectral_data: dict[str, Any] | None = None,
    change_data: dict[str, Any] | None = None,
    sensor_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a structured analysis report.

    Combines VLM output, spectral index data, and change detection results
    into a comprehensive report suitable for display and download.

    Args:
        query: The user's original query.
        task_type: Classified task type (e.g. ``"change_detection"``).
        vlm_result: Output from :func:`vlm_engine.generate_answer`.
        spectral_data: Optional spectral index computation results.
        change_data: Optional change detection results.
        sensor_metadata: Optional sensor/image metadata.

    Returns:
        Dict with keys:
        - ``title``: str — report title.
        - ``generated_at``: str — ISO timestamp.
        - ``query``: str — original query.
        - ``task_type``: str — classified task.
        - ``summary``: str — executive summary.
        - ``detailed_answer``: str — full VLM answer.
        - ``confidence``: float — overall confidence.
        - ``confidence_label``: str — HIGH/MEDIUM/LOW.
        - ``reasoning_steps``: list[str].
        - ``spectral_indices``: dict or None.
        - ``change_metrics``: dict or None.
        - ``highlighted_regions``: list[dict].
        - ``recommended_actions``: list[str].
        - ``sensor_info``: dict or None.
        - ``report_markdown``: str — full Markdown report text.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    confidence = vlm_result.get("confidence", 0.0)
    if confidence > 0.7:
        conf_label = "HIGH"
    elif confidence > 0.4:
        conf_label = "MEDIUM"
    else:
        conf_label = "LOW"

    # Build title
    task_labels = {
        "scene_classification": "Scene Classification Report",
        "object_detection": "Object Detection Report",
        "change_detection": "Change Detection Report",
        "spectral_analysis": "Spectral Analysis Report",
        "area_measurement": "Area Measurement Report",
        "disaster_assessment": "Disaster Assessment Report",
    }
    title = task_labels.get(task_type, "Satellite Image Analysis Report")

    # Executive summary
    summary_parts = [vlm_result.get("answer", "Analysis complete.")]
    if change_data and "summary" in change_data:
        summary_parts.append(change_data["summary"])
    summary = " ".join(summary_parts)

    # Build Markdown report
    md = _build_markdown(
        title=title,
        timestamp=timestamp,
        query=query,
        task_type=task_type,
        summary=summary,
        vlm_result=vlm_result,
        spectral_data=spectral_data,
        change_data=change_data,
        sensor_metadata=sensor_metadata,
        confidence=confidence,
        conf_label=conf_label,
    )

    return {
        "title": title,
        "generated_at": timestamp,
        "query": query,
        "task_type": task_type,
        "summary": summary,
        "detailed_answer": vlm_result.get("answer", ""),
        "confidence": confidence,
        "confidence_label": conf_label,
        "reasoning_steps": vlm_result.get("reasoning_steps", []),
        "spectral_indices": spectral_data,
        "change_metrics": change_data.get("area_metrics") if change_data else None,
        "highlighted_regions": vlm_result.get("highlighted_regions", []),
        "recommended_actions": vlm_result.get("recommended_actions", []),
        "sensor_info": sensor_metadata,
        "report_markdown": md,
    }


def _build_markdown(
    title: str,
    timestamp: str,
    query: str,
    task_type: str,
    summary: str,
    vlm_result: dict[str, Any],
    spectral_data: dict[str, Any] | None,
    change_data: dict[str, Any] | None,
    sensor_metadata: dict[str, Any] | None,
    confidence: float,
    conf_label: str,
) -> str:
    """Build a full Markdown report from analysis components.

    Args:
        All parameters correspond to the structured report fields.

    Returns:
        Formatted Markdown string.
    """
    lines = [
        f"# {title}",
        "",
        f"**Generated:** {timestamp}  ",
        f"**Query:** {query}  ",
        f"**Task Type:** {task_type}  ",
        f"**Confidence:** {confidence:.1%} ({conf_label})",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        summary,
        "",
    ]

    # Sensor info
    if sensor_metadata:
        lines.extend([
            "## Sensor Information",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
        ])
        for key, value in sensor_metadata.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")

    # Spectral indices
    if spectral_data:
        lines.extend([
            "## Spectral Index Results",
            "",
            f"| Index | Min | Max | Mean | Std | Interpretation |",
            f"|-------|-----|-----|------|-----|----------------|",
        ])
        if isinstance(spectral_data, dict):
            idx = spectral_data.get("index", "")
            lines.append(
                f"| {idx} "
                f"| {spectral_data.get('min', 0):.3f} "
                f"| {spectral_data.get('max', 0):.3f} "
                f"| {spectral_data.get('mean', 0):.3f} "
                f"| {spectral_data.get('std', 0):.3f} "
                f"| {spectral_data.get('interpretation', '')} |"
            )
        lines.append("")

    # Change metrics
    if change_data:
        metrics = change_data.get("area_metrics", change_data)
        lines.extend([
            "## Change Detection Metrics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Area (m²) | {metrics.get('area_m2', 'N/A')} |",
            f"| Area (ha) | {metrics.get('area_ha', 'N/A')} |",
            f"| Area (km²) | {metrics.get('area_km2', 'N/A')} |",
            f"| Changed pixels | {metrics.get('n_changed_pixels', 'N/A')} |",
            f"| % Changed | {metrics.get('pct_changed', 'N/A')}% |",
            "",
        ])

    # Reasoning
    steps = vlm_result.get("reasoning_steps", [])
    if steps:
        lines.extend(["## Reasoning Steps", ""])
        for i, step in enumerate(steps, 1):
            lines.append(f"{i}. {step}")
        lines.append("")

    # Recommended actions
    actions = vlm_result.get("recommended_actions", [])
    if actions:
        lines.extend(["## Recommended Actions", ""])
        for action in actions:
            lines.append(f"- {action}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "*Report generated by SatQuery AI — ISRO SIH26167*",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

generate_structured_report = generate_report


if __name__ == "__main__":
    sample_vlm = {
        "answer": "Flood detected in 11.93 hectares.",
        "confidence": 0.97,
        "reasoning_steps": ["Step 1", "Step 2"],
        "highlighted_regions": [],
        "recommended_actions": ["Deploy teams", "Monitor NDWI"],
    }
    report = generate_report(
        query="How much has the flood spread?",
        task_type="disaster_assessment",
        vlm_result=sample_vlm,
        sensor_metadata={"sensor": "Cartosat-2S", "resolution_m": 0.65},
    )
    print(report["report_markdown"])
    print("\nReport Generator OK ✅")
