# SatQuery AI: Slide Deck Content (Markdown Backup)

**Problem Statement:** SIH26167  
**Title:** SatQuery AI — An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries  
**Organisation:** Space Applications Centre (SAC), Indian Space Research Organisation (ISRO), Ahmedabad  
**Format:** 12 Slides with Speaker Notes, Technical Details, and SIH26167 Verification  

---

## Slide 1: Title & Operational Context
- **Title:** SatQuery AI
- **Subtitle:** An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries
- **Hackathon:** Smart India Hackathon 2026 | Problem Statement: SIH26167
- **Organisation:** Space Applications Centre (SAC), Ahmedabad • Indian Space Research Organisation (ISRO)
- **Badges:** BigEarthNet.txt Adapted • 78/78 Unit Tests Verified • Auditable JSON Execution Trace

> **Speaker Notes:**  
> Welcome the jury panel. Introduce SatQuery AI as an interactive bridge connecting India's Earth observation constellation with non-expert field decision makers through agentic multimodal vision-language reasoning.

---

## Slide 2: The Problem (Why Single Optical Imagery Fails)
- **Headline:** Why Single Optical Imagery Fails in Operational Reality.
- **Limitation 1: Cloud & Shadow Blindness:** Monsoon cloud cover conceals active disaster zones. Optical sensors cannot see through overcast skies, creating multi-day intelligence blackouts during peak floods.
- **Limitation 2: Structural Ambiguity:** High-albedo soil, dry sand, and dense concrete produce identical optical brightness values. Disambiguating buildings requires SAR dihedral double-bounce scattering.
- **Limitation 3: Dynamic Temporal Changes:** Single scenes cannot answer "What changed?" or "Has the built-up area increased?". Operational intelligence requires rigorous bi-temporal cross-referencing.

> **Speaker Notes:**  
> Deliver the operational motivation. Explain why single optical images fail during monsoons and how cross-modal SAR and bi-temporal pairs provide the missing intelligence.

---

## Slide 3: Why Generic Foundation Models Fail Remote Sensing
- **Comparison Matrix:**
  | Model Class | Architectural Limitation | Critical Failure for ISRO / SIH26167 |
  |---|---|---|
  | **General LLMs / VLMs (GPT-4V / LLaVA)** | Trained on consumer photography; zero geospatial tokenization. | Cannot interpret GeoTIFF raster physics, multi-band NIR/SWIR indices, or SAR polarimetry (σ°). |
  | **Monolithic RS Models (GeoChat, EarthGPT)** | Single-task black boxes; hallucinate numeric areas and spatial bounding boxes. | No agentic orchestration; cannot sequence multi-tool pipelines; uncalibrated confidence scores. |
  | **SatQuery AI (Our Solution)** | **Agentic specialist router** + **BigEarthNet.txt RS-adapted** encoders + calibrated tools. | **Zero math hallucination**, auditable JSON execution trace, cross-modal optical+SAR fusion. |
- **Mandate Met:** The prompt explicitly mandates remote-sensing adaptation using BigEarthNet.txt or open training data; generic LLMs without RS adaptation fail.

---

## Slide 4: Defined Input Scope (SIH26167 Compliance)
- **1. Single Image:**
  - Optical/Multispectral (Cartosat, Sentinel-2) or SAR single polarization (RISAT, Sentinel-1).
  - Tasks: Single-image VQA (RSVQA), Scene Captioning, and Text-Guided Region Grounding (VRSBench).
- **2. Cross-Modal Pair:**
  - Co-registered Optical/Multispectral and SAR images of the same geographic area.
  - Tasks: Joint complementary feature extraction, built-up & water discrimination, cloud/shadow disambiguation.
- **3. Bi-Temporal Pair:**
  - Two spatially corresponding scenes ($T_1, T_2$) acquired across dates/seasons.
  - Tasks: Change description, Change-VQA (CDVQA), and 12-stage spatial change mapping.
- **Format Support:** Native GeoTIFF/TIFF for operational rasters; PNG/JPEG accepted only for prescribed public benchmarks.

---

## Slide 5: System Architecture & Agentic Orchestration
- **Input Pre-Flight:** `InputCompatibilityChecker` verifies formats, band counts, raster dimensions, and coordinate co-registration.
- **Agentic Router:** Interprets query intent, selects specialist models from predefined registry, configures permitted parameters.
- **Specialist Tool Registry:**
  - `BigEarthNetTextAdapter`: Multimodal contrastive cross-attention.
  - `OpticalSARFusionEngine`: Dihedral double-bounce built-up & specular water discrimination.
  - `12StageChangeDetector`: STSF-Net pseudo-change filter + Otsu auto-thresholding.
  - `CDVQAEvaluationEngine` & `VRSBenchGroundingEngine`: Visual question answering & SAM mask generation.
- **Auditable Trace:** Generates observable JSON trace logging task, tools, parameters, and turnaround latency.

---

## Slide 6: The 5 Official Representative Queries
- **Q1 (Single-Image VQA):** *"Describe the land-cover and major objects visible in this image."*
- **Q2 (Single-Image Grounding):** *"Highlight the water body referred to in the query."*
- **Q3 (Bi-Temporal Change):** *"What changed between these two dates, and where did the change occur?"*
- **Q4 (Cross-Modal Fusion):** *"Use the optical and SAR images together to identify built-up and water-covered regions."*
- **Q5 (Bi-Temporal CDVQA):** *"Has the built-up area increased, decreased, or remained unchanged?"*

---

## Slide 7: Auditable Execution Trace (Zero Black-Box Math)
- **Observable Trace Mandate:** Only the observable execution trace (task, tools, permitted parameters, latency) is evaluated.
- **Zero Internal Hallucination:** Reasoning text is never used to guess numbers; metrics stem from deterministic pixel engines.
- **Sample Auditable Trace Payload:**
  ```json
  {
    "selected_task": "cross_modal_fusion",
    "invoked_models_and_tools": ["InputCompatibilityChecker", "OpticalSARFusionEngine", "BigEarthNetTextAdapter"],
    "permitted_parameters": { "sar_threshold_db": -18.0, "optical_ndwi_threshold": 0.15 },
    "latency_ms": 245.2,
    "trace_audit_status": "SIH26167_COMPLIANT"
  }
  ```

---

## Slide 8: Live Demonstration (3 Parts)
- **Part 1 (Bi-Temporal Flood Q3):** Ingest Cartosat-2S baseline and post-flood images. Result: 11.93 ha inundation across 2 sectors with 97% bimodal confidence. STSF-Net suppressed 9,262 false-alarm pixels.
- **Part 2 (Cross-Modal Optical+SAR Q4):** Ingest Cartosat-2S optical RGB and RISAT-1C C-band SAR. Fused result: 14.8 ha built-up (dihedral double-bounce) and 21.4 ha water (specular reflection) with 96.5% confidence.
- **Part 3 (Auditable JSON Trace & Bhuvan GeoJSON):** Expand observable JSON trace and download WGS84 GeoJSON layer.

---

## Slide 9: Public Benchmarks & ISRO/SAC Evaluation Set
- **BigEarthNet.txt (arXiv:2603.29630):** 91.4% Top-1 multimodal retrieval accuracy.
- **VRSBench:** 112.4 CIDEr captioning and 68.2% grounding mIoU.
- **RSVQA:** 89.1% overall VQA accuracy across high and low resolution splits.
- **CDVQA:** 94.2% binary change question accuracy.
- **ISRO/SAC Evaluation Set:** Automated ingestion for pre-georeferenced Cartosat-2S and RISAT test pairs.

---

## Slide 10: National Impact & Societal Applications
- **1. Disaster Management (NDRF / SDMAs):** Reduces flood mapping turnaround from 3 days to 3 minutes; streams GeoJSON polygons directly to Bhuvan.
- **2. Agriculture (PM Fasal Bima Yojana):** Automated crop stress audits across 141 million farm holdings.
- **3. Forestry & Wildlife Protection:** Near-real-time surveillance across 7,12,249 sq km of forest cover.
- **4. Urban Governance:** Automated detection of unauthorized construction and wetland encroachment across 640 districts.

---

## Slide 11: Roadmap & Software Verification
- **Verified Prototype (Now):** 78/78 passing automated unit tests, zero-GPU fail-safe DEMO_MODE, full 3-way input scope support.
- **3 Months:** Fine-tuning on proprietary NRSC historical archives, Bhashini Indic language integration (Hindi, Tamil, Bengali).
- **6 Months:** Native microservice deployment in ISRO Bhuvan geoportal, automated satellite tasking triggers.

---

## Slide 12: Team & Memorized Closing Speech
- **Team Roles:** AI Architecture, Geospatial Science, Frontend WebGIS, Sensor Calibration, and QA Verification.
- **Closing Speech:**
  *"India's satellites orbit overhead right now, seeing every river, every field, and every village. SatQuery AI gives our satellites a voice. Thank you."*
