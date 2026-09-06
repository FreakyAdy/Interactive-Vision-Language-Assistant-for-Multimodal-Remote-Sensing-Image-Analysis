# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║        SATQUERY AI — SIH 2026 (PS SIH26167) — MASTER AGENT PROMPT          ║
# ║    ISRO Vision-Language Assistant for Multimodal Remote Sensing Analysis    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

> **READ THIS FIRST — WHO YOU ARE AND WHAT YOU MUST DO**
>
> You are a senior full-stack AI engineer, research scientist, UX designer, and technical writer — all in one. You have just been handed a single working directory. Your job is to build everything from zero: the full project, the research documentation, the presentation slides, the pitch guide, and the demo data. You will not ask for permission before starting any subtask. You will not wait for feedback between steps. You will execute sequentially, decisively, and completely. When you finish, every file described below must exist, be runnable, and be ready for a live judging panel the next day.
>
> **The team has 6 members. One person (you are building FOR them) is doing all the technical work. One other person will present. Your job is to make BOTH succeed completely.**

---

## ═══ SECTION 0: CONTEXT & MISSION ═══

### Problem Statement
- **ID:** SIH26167
- **Organisation:** ISRO / Space Applications Centre (SAC), Ahmedabad
- **Title:** SatQuery AI — An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries
- **Category:** Software | Space Technology
- **Competition:** Smart India Hackathon 2026

### What ISRO Actually Wants (Read Carefully)
ISRO needs a system where a **non-expert user** (disaster manager, agricultural officer, urban planner, forest ranger) can:
1. Upload a satellite image — optical (JPEG/PNG/GeoTIFF) or SAR — from any ISRO sensor (Cartosat, RISAT, ResourceSat, EOS series)
2. Type a plain-English question about it: *"Has this forest area shrunk since last month?"*, *"Where are the flooded regions?"*, *"Count the buildings in this zone"*
3. Get a **rich, explained answer** — not just a number, but annotated images, natural language explanation, confidence score, and optionally a GeoJSON overlay

The **research gap** they're targeting: existing VLMs (GeoChat, RSGPT, EarthGPT, RemoteCLIP) are trained on Western datasets (DOTA, DIOR, NWPU-RESISC45) and do NOT handle ISRO sensor characteristics, Indian geographic contexts, or multi-temporal analysis well. SatQuery must bridge this.

### Judging Criteria (SIH Standard — Weight Each Decision Against These)
1. **Innovation & Uniqueness (25%)** — Does it do something that doesn't exist yet?
2. **Technical Feasibility (25%)** — Can it actually be built and demoed?
3. **Societal/National Impact (20%)** — Does it serve India's real needs?
4. **Business Viability / Scalability (15%)** — Could ISRO actually deploy this?
5. **Presentation Quality (15%)** — Is the pitch clear, confident, and well-structured?

---

## ═══ SECTION 1: DIRECTORY STRUCTURE YOU MUST CREATE ═══

Create the following complete directory tree in the current working directory. Do not skip any file. Create them all.

```
satquery-ai/
│
├── 📁 PROJECT/                          # The actual working application
│   ├── 📁 backend/
│   │   ├── main.py                      # FastAPI app entrypoint
│   │   ├── config.py                    # Config, env vars, model paths
│   │   ├── 📁 api/
│   │   │   ├── routes.py                # All API route definitions
│   │   │   └── schemas.py               # Pydantic request/response models
│   │   ├── 📁 core/
│   │   │   ├── vlm_engine.py            # VLM inference pipeline
│   │   │   ├── image_processor.py       # GeoTIFF/optical/SAR preprocessing
│   │   │   ├── change_detector.py       # Bi-temporal change detection
│   │   │   ├── query_router.py          # Routes query to correct tool
│   │   │   ├── spectral_indices.py      # NDVI, NDWI, NDBI, RVI, EVI
│   │   │   ├── object_detector.py       # DOTA-trained object detection
│   │   │   ├── segmentor.py             # SAM-based segmentation
│   │   │   └── report_generator.py      # Structured NL report generator
│   │   ├── 📁 sensors/
│   │   │   ├── cartosat.py              # Cartosat-2S calibration constants
│   │   │   ├── risat.py                 # RISAT-1C SAR handling
│   │   │   └── resourcesat.py           # ResourceSat-2A handling
│   │   └── requirements.txt
│   │
│   ├── 📁 frontend/
│   │   ├── index.html                   # Main SPA entry
│   │   ├── 📁 src/
│   │   │   ├── App.jsx
│   │   │   ├── 📁 components/
│   │   │   │   ├── ImageUploader.jsx    # Drag-drop upload with preview
│   │   │   │   ├── QueryInput.jsx       # NL query text box + examples
│   │   │   │   ├── ResultViewer.jsx     # Split: image annotation + text
│   │   │   │   ├── MapOverlay.jsx       # Leaflet GeoJSON overlay
│   │   │   │   ├── ChangeDetectionView.jsx
│   │   │   │   ├── SpectralChart.jsx    # Recharts spectral index chart
│   │   │   │   ├── ConfidenceGauge.jsx
│   │   │   │   └── SensorBadge.jsx      # ISRO sensor type indicator
│   │   │   ├── 📁 pages/
│   │   │   │   ├── Home.jsx
│   │   │   │   ├── Analyze.jsx          # Main analysis page
│   │   │   │   └── About.jsx
│   │   │   └── 📁 utils/
│   │   │       ├── api.js
│   │   │       └── imageUtils.js
│   │   └── package.json
│   │
│   ├── 📁 ml_models/
│   │   ├── download_models.py           # Script to pull weights from HuggingFace
│   │   ├── finetune_geochat.py          # Fine-tuning script for ISRO dataset
│   │   ├── create_synthetic_dataset.py  # Generates synthetic ISRO-like training data
│   │   └── evaluate_model.py            # Benchmark evaluation
│   │
│   ├── 📁 demo_data/
│   │   ├── generate_demo_images.py      # Creates realistic synthetic satellite images
│   │   ├── sample_optical.py            # Cartosat-like optical sample
│   │   ├── sample_sar.py                # RISAT-like SAR sample
│   │   └── sample_temporal_pair.py      # Before/after image pair for change detection
│   │
│   ├── 📁 tests/
│   │   ├── test_vlm_engine.py
│   │   ├── test_change_detection.py
│   │   ├── test_spectral_indices.py
│   │   ├── test_api_routes.py
│   │   └── test_query_router.py
│   │
│   └── docker-compose.yml
│
├── 📁 RESEARCH/                         # Everything judges need to see as background
│   ├── literature_review.md             # State-of-art VLMs for remote sensing
│   ├── isro_sensors_reference.md        # ISRO sensor specs cheatsheet
│   ├── research_gaps.md                 # What existing solutions miss
│   ├── our_innovations.md               # Our 5 key differentiators
│   ├── datasets_used.md                 # Training/benchmark datasets
│   ├── architecture_decision_log.md     # Why we chose each component
│   └── references.bib                   # All academic citations
│
├── 📁 PRESENTATION/                     # Everything for tomorrow's pitch
│   ├── slides.html                      # Full HTML presentation (Reveal.js)
│   ├── slides_backup.md                 # Markdown version of all slide content
│   ├── PRESENTER_SCRIPT.md             # Word-for-word what the presenter says
│   ├── JUDGE_QA_PREP.md                # 30 expected questions + ideal answers
│   ├── PITCH_TIMING_GUIDE.md           # Minute-by-minute flow for 10-min pitch
│   └── VISUAL_DEMO_FLOW.md             # How to live-demo the working app
│
├── 📁 DIAGRAMS/
│   ├── system_architecture.svg          # Full system architecture diagram
│   ├── pipeline_flow.svg                # Query → VLM → Result pipeline
│   ├── data_flow.svg                    # Data flow from satellite to user
│   └── ui_mockup.svg                    # App UI wireframe
│
├── README.md                            # Project overview & quick-start
└── HACKATHON_CHECKLIST.md              # SIH submission checklist
```

---

## ═══ SECTION 2: BUILD THE BACKEND (PROJECT/backend/) ═══

### 2.1 — `PROJECT/backend/requirements.txt`
Write this file with the following dependencies (pin versions for reproducibility):
```
fastapi==0.111.0
uvicorn[standard]==0.30.1
python-multipart==0.0.9
pydantic==2.7.1
torch==2.3.0
torchvision==0.18.0
transformers==4.42.3
pillow==10.3.0
numpy==1.26.4
rasterio==1.3.10
geopandas==0.14.4
scikit-image==0.23.2
scipy==1.13.1
opencv-python-headless==4.10.0.82
matplotlib==3.9.0
shapely==2.0.4
pyproj==3.6.1
huggingface_hub==0.23.3
sentencepiece==0.2.0
accelerate==0.31.0
bitsandbytes==0.43.1
python-dotenv==1.0.1
httpx==0.27.0
pytest==8.2.2
```

### 2.2 — `PROJECT/backend/config.py`
Write a full configuration file that:
- Loads from environment variables with sensible defaults
- Defines `MODEL_NAME = "MBZUAI/geochat-7B"` as default VLM (can be swapped)
- Defines `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`
- Defines `MAX_IMAGE_SIZE_MB = 50`
- Defines `SUPPORTED_FORMATS = [".tif", ".tiff", ".jpg", ".jpeg", ".png"]`
- Defines spectral band mappings for each ISRO sensor (Cartosat-2S, RISAT-1C, ResourceSat-2A, EOS-04)
- Has a `DEMO_MODE = True` flag — when True, the app uses synthetic data and mock inference, so it works without GPU or actual model weights
- Includes a `Settings` class using Pydantic BaseSettings

### 2.3 — `PROJECT/backend/core/spectral_indices.py`
Write a complete, production-quality spectral indices module that:
- Implements NDVI (Normalized Difference Vegetation Index): `(NIR - Red) / (NIR + Red)`
- Implements NDWI (Normalized Difference Water Index): `(Green - NIR) / (Green + NIR)`
- Implements NDBI (Normalized Difference Built-up Index): `(SWIR - NIR) / (SWIR + NIR)`
- Implements EVI (Enhanced Vegetation Index): `2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)`
- Implements RVI (Radar Vegetation Index for SAR): `4 * sigma_VH / (sigma_VV + sigma_VH)` 
- Each function takes a numpy array and band configuration dict as input
- Returns the index array AND a metadata dict with min/max/mean/std and interpretation
- Includes an `auto_select_index(query: str, sensor_type: str) -> str` function that picks the right index based on the user's natural language query keywords
- All functions have docstrings explaining the physical meaning of the index
- Includes `interpret_index_value(index_name: str, value: float) -> str` that returns a plain-English interpretation (e.g., NDVI 0.6 → "Dense healthy vegetation detected")

### 2.4 — `PROJECT/backend/core/image_processor.py`
Write a complete image preprocessing module that:
- Handles GeoTIFF with rasterio: reads CRS, transform, band count, nodata values
- Handles standard optical images with PIL: RGB to numpy, normalization
- Handles SAR images: converts dB scale, handles noise filtering
- Function `preprocess_for_vlm(image_path, sensor_type) -> dict` that returns:
  - `rgb_array`: 3-channel numpy array normalized to [0,1]
  - `metadata`: dict with sensor, bands, resolution, CRS, bounding_box, timestamp_if_available
  - `sensor_badge`: string like "ISRO Cartosat-2S | 0.65m | Panchromatic"
- Function `validate_image(image_path) -> (bool, str)` that checks format, size, minimum resolution
- Function `coregister_temporal_pair(t1_path, t2_path) -> (array1, array2, alignment_score)` for change detection preprocessing
- Graceful fallback: if rasterio cannot open the file, try PIL; log the fallback

### 2.5 — `PROJECT/backend/core/change_detector.py`
Write a complete bi-temporal change detection engine that implements:

**Pipeline (12 stages, each logged with timing):**
1. **Input Validation** — CRS check, bounding box overlap, resolution ratio, temporal ordering
2. **Co-registration** — Align T1 and T2 images spatially
3. **Atmospheric Normalization** — Basic histogram matching between T1 and T2
4. **Index Selection** — Auto-select best spectral index for the detected scene type
5. **Index Computation** — Compute the spectral index for both T1 and T2
6. **Difference Map** — Compute absolute and signed difference maps
7. **Pseudo-Change Suppression (STSF-Net inspired)** — Compare local spatial variance `σ_T1, σ_T2` vs mean signed diff `μ_Δ` in local w×w patches to filter false positives caused by radiometric drift
8. **Smoothing** — Gaussian smoothing to remove salt-and-pepper noise
9. **Thresholding (Otsu)** — Automatic threshold selection maximizing inter-class variance
10. **Morphological Cleaning** — Opening (remove isolated pixels) + Closing (fill small holes)
11. **Region Analysis** — Connected components: count regions, area per region (m², ha, km²)
12. **Confidence Scoring** — Bimodal histogram separation score combining Otsu inter-class variance ratio ω, valley-to-peak depth ratio v, and area imbalance penalty p → output 0.0-1.0 score

**Output JSON contract:**
```python
{
    "status": "ok",
    "primary_index": "ndwi",
    "change_direction": "water_expansion",  # or vegetation_loss, urban_growth, water_recession, deforestation
    "confidence": 0.967,
    "confidence_label": "HIGH",  # HIGH > 0.7, MEDIUM 0.4-0.7, LOW < 0.4
    "area_metrics": {
        "area_m2": 119324.0,
        "area_ha": 11.9324,
        "area_km2": 0.119324,
        "n_changed_pixels": 29831,
        "total_pixels": 65536,
        "pct_changed": 45.52
    },
    "n_regions": 2,
    "otsu_threshold": 0.1245,
    "n_pseudo_removed": 9262,
    "summary": "Between [date1] and [date2], a substantial expansion of surface water was detected, covering approximately 11.93 ha (45.5% of the image), spanning 2 distinct regions...",
    "geojson": { "type": "FeatureCollection", "features": [...] },
    "execution_trace": [
        {"stage": 0, "name": "Input Validation", "tool": "GeoValidator",
         "observation": "...", "duration_ms": 1.2, "why": "..."}
    ],
    "sensor_calibration_note": "Outputs calibrated for ISRO Cartosat-2S",
    "total_processing_ms": 99.8
}
```

### 2.6 — `PROJECT/backend/core/vlm_engine.py`
Write a complete VLM inference engine that:
- Loads GeoChat-7B (or any LLaVA-based model) from HuggingFace using transformers
- Supports 4-bit quantization via bitsandbytes (for low-VRAM machines)
- Has a `DEMO_MODE` path: when `config.DEMO_MODE = True`, returns realistic canned responses from a lookup table keyed on query keywords — this ensures the app works without any GPU or model weights for demo day
- Function `generate_answer(image_array: np.ndarray, query: str, sensor_metadata: dict) -> dict`:
  - Constructs a rich prompt that includes sensor type, available bands, any pre-computed spectral indices, and the user's query
  - The system prompt used must say: "You are SatQuery, an expert satellite image analyst for ISRO. You have access to Cartosat, RISAT, ResourceSat and EOS satellite imagery. Answer queries with precise geographic and scientific language. Always mention confidence level, physical interpretation, and recommended follow-up actions."
  - Returns: `{"answer": str, "confidence": float, "reasoning_steps": list[str], "highlighted_regions": list[dict], "recommended_actions": list[str]}`
- Function `batch_analyze(images: list, queries: list) -> list` for processing multiple images
- All inference wrapped in try/except with detailed error messages

### 2.7 — `PROJECT/backend/core/query_router.py`
Write a query routing engine that:
- Takes the user's text query and classifies it into one of 6 task types:
  1. `scene_classification` — "What kind of area is this?"
  2. `object_detection` — "How many buildings/ships/vehicles are there?"
  3. `change_detection` — "What changed between these two images?"
  4. `spectral_analysis` — "What is the vegetation/water/urban index?"
  5. `area_measurement` — "How large is this lake/forest?"
  6. `disaster_assessment` — "How much flood damage / burnt area?"
- Uses keyword matching + simple embedding similarity (use sentence-transformers `all-MiniLM-L6-v2`)
- Returns: `{"task_type": str, "confidence": float, "requires_temporal": bool, "suggested_indices": list[str], "pipeline_steps": list[str]}`
- Has a fallback: if confidence < 0.5, route to `scene_classification` and mention "I interpreted your query as a general scene description request"

### 2.8 — `PROJECT/backend/api/schemas.py`
Write complete Pydantic schemas for:
- `QueryRequest`: image file + query text + optional second image for change detection + sensor_type
- `AnalysisResponse`: full response with answer, confidence, annotated image (base64), GeoJSON, spectral index values, execution trace
- `ChangeDetectionRequest`: two image files + timestamps + query
- `ChangeDetectionResponse`: extends AnalysisResponse with change metrics
- `HealthResponse`: API version, model loaded, demo_mode status, uptime

### 2.9 — `PROJECT/backend/api/routes.py`
Write complete FastAPI routes:
- `GET /health` → health check
- `POST /api/analyze` → single image analysis
- `POST /api/change-detection` → bi-temporal change detection
- `POST /api/spectral-indices` → compute spectral indices for an image
- `GET /api/demo/scenarios` → returns list of built-in demo scenarios
- `POST /api/demo/run/{scenario_id}` → runs a specific demo with synthetic data
- Every route has OpenAPI docstrings, example requests, and proper error handling (400, 422, 500)
- CORS enabled for frontend

### 2.10 — `PROJECT/backend/main.py`
Write the FastAPI app entrypoint that:
- Creates the app with title "SatQuery AI — ISRO Remote Sensing Assistant", version "1.0.0", description with ISRO SIH26167 reference
- Includes startup event that loads models (or announces DEMO_MODE)
- Includes `/docs` Swagger UI and `/redoc` ReDoc documentation
- Middleware: CORS, request timing logger, error handler
- On startup, prints a banner: "🛰️ SatQuery AI is ready | ISRO SIH26167 | Mode: DEMO/PRODUCTION"

---

## ═══ SECTION 3: BUILD THE FRONTEND (PROJECT/frontend/) ═══

### 3.1 — Overall Frontend Design
The frontend must look like a **professional ISRO-grade satellite analysis tool**, not a student project. Use this design language:
- **Color palette:** Deep space navy `#0B1628` as background, ISRO orange `#FF6B00` as accent, satellite green `#00D4AA` for success states, white `#FFFFFF` for text
- **Font:** Inter (Google Fonts) for all text, monospace for data outputs
- **Layout:** Split-pane: left side is upload + query input, right side is result viewer with image + text
- **Feel:** NASA/ISRO mission control aesthetic — dark mode, subtle grid lines, animated loading states that look like satellite data downloading

### 3.2 — `PROJECT/frontend/index.html`
Write a complete single-file HTML+CSS+JavaScript application (no build step required, works with just `open index.html` or a simple HTTP server) that:

**Header Section:**
- ISRO logo (use SVG placeholder) + "SatQuery AI" in the title
- "SIH26167 | Space Applications Centre, ISRO" subtitle
- "DEMO MODE" badge if applicable

**Upload Section (Left Panel):**
- Large drag-and-drop zone for primary image
- "Upload second image for change detection" toggle (hidden by default, appears when user checks "Compare two images")
- Sensor type dropdown: Auto-detect, Cartosat-2S (Optical), RISAT-1C (SAR), ResourceSat-2A, EOS-04, Other
- After upload: shows image preview with metadata (filename, size, detected dimensions, sensor badge)

**Query Section (Middle):**
- Large text area with placeholder "Ask anything about this satellite image..."
- Example queries as clickable chips below the text area:
  - "What has changed in this area?"
  - "Identify flooded regions"
  - "How much vegetation is present?"
  - "Count the buildings visible"
  - "Assess damage from the disaster"
- "Analyze" button (ISRO orange, prominent, with satellite icon)
- Query type indicator: after typing, a small tag shows "Detected: Change Detection / Object Counting / Spectral Analysis"

**Result Section (Right Panel):**
- Image viewer with annotation overlay (bounding boxes, segmentation masks, change region highlights)
- Natural language answer in a styled card with ISRO green accent
- Confidence gauge (circular progress indicator, 0-100%)
- Expandable "Reasoning Steps" section (shows the VLM's chain of thought)
- Spectral index readout (NDVI/NDWI/NDBI bar chart)
- "Recommended Actions" list (what the user should do next)
- GeoJSON download button
- Full report download button (PDF)
- "Execution Trace" panel (expandable, shows all 12 pipeline stages)

**Loading State:**
- Animated satellite orbiting a globe (pure CSS animation)
- Status messages cycling: "Preprocessing image...", "Computing spectral indices...", "Querying vision model...", "Generating report..."

**Demo Scenarios Sidebar:**
- 5 built-in scenarios (buttons):
  1. Kerala Floods 2023 (Change Detection)
  2. Amazon Deforestation analog (Vegetation Loss)
  3. Urban Expansion — Delhi outskirts (NDBI)
  4. Agricultural Field Health Check (NDVI)
  5. Coastal Area — Ship Detection (Object Detection)
- Clicking a scenario loads synthetic demo images and runs analysis automatically

**Technical requirements:**
- No external JavaScript frameworks required (vanilla JS + fetch API)
- The app calls `http://localhost:8000` for the backend
- Has a `DEMO_MODE` JavaScript variable — when true, the app simulates API calls with hardcoded responses after a 2-second animated delay, so the demo works even if backend isn't running
- Fully responsive: works on laptop and projector screen
- Include a "How It Works" collapsible section explaining the system

---

## ═══ SECTION 4: BUILD THE DEMO DATA GENERATOR ═══

### 4.1 — `PROJECT/demo_data/generate_demo_images.py`
Write a complete script that generates realistic synthetic satellite images for demo purposes. This is critical — judges will see REAL visuals, not placeholders.

**Scenario 1 — Flood Detection (Kerala analog):**
- Generate T1 image (before flood): green vegetation (NDVI ~0.65), brown soil, some river pixels (NDWI ~0.2)
- Generate T2 image (after flood): large water expansion (NDWI ~0.7), flooded area covering ~40% of image
- Use `numpy` + `PIL` to create realistic multi-band images
- Add Gaussian noise that matches satellite sensor characteristics
- Save as GeoTIFF with proper CRS (EPSG:4326) and realistic bounding box over Kerala

**Scenario 2 — Deforestation Detection:**
- T1: dense forest (NDVI ~0.75 across 80% of image)
- T2: selective clearing (NDVI drops to ~0.1 in a wedge pattern, mimicking logging roads)
- Changed area: ~12% of image

**Scenario 3 — Urban Expansion:**
- T1: agricultural fields (NDVI ~0.5, NDBI ~-0.3)
- T2: new construction (NDBI ~0.4 in expansion zone, bare soil patches)
- Changed area: ~8% of image

**Scenario 4 — Agricultural Health:**
- Single image: field showing variation from healthy (NDVI 0.7) to stressed (NDVI 0.2) crops
- Includes irrigation channel pixels

**For each scenario, also generate:**
- A JSON metadata file with sensor type, timestamps, geographic info, expected analysis results
- A "ground truth" JSON showing what the correct answer should be

### 4.2 — `PROJECT/demo_data/DEMO_SCENARIOS.json`
Write a complete JSON file with all 5 demo scenarios, each containing:
```json
{
  "id": "flood_kerala_2023",
  "title": "Kerala Flood Detection",
  "description": "Detect and quantify flood inundation from cyclone event",
  "sensor": "Cartosat-2S",
  "image_t1": "demo_data/flood_t1.tif",
  "image_t2": "demo_data/flood_t2.tif",
  "example_query": "How much area has been flooded between these two dates?",
  "expected_answer_summary": "Approximately 11.9 hectares of surface water expansion detected, HIGH confidence (0.97)",
  "talking_points": ["Shows flood extent for disaster response", "NDWI index auto-selected", "GeoJSON output ready for GIS teams"]
}
```

---

## ═══ SECTION 5: BUILD THE TESTS ═══

### 5.1 — `PROJECT/tests/test_spectral_indices.py`
Write comprehensive pytest tests (minimum 15 tests) covering:
- NDVI returns correct values for known synthetic inputs
- NDWI correctly identifies water pixels
- Edge cases: all-zero array, NaN values, single pixel, max/min value arrays
- `auto_select_index` correctly maps "flood" → "ndwi", "vegetation" → "ndvi", "building" → "ndbi"
- `interpret_index_value` returns correct interpretation strings

### 5.2 — `PROJECT/tests/test_change_detection.py`
Write comprehensive pytest tests (minimum 20 tests) covering:
- All 12 pipeline stages execute without error on synthetic data
- Flood scenario produces `change_direction = "water_expansion"`
- Deforestation scenario produces `change_direction = "vegetation_loss"`
- Urban scenario produces `change_direction = "urban_growth"`
- Confidence scores are in [0.0, 1.0]
- Output JSON has all required keys
- GeoJSON output is valid GeoJSON
- Area metrics are physically reasonable (positive, consistent with n_pixels)
- Pseudo-change suppression removes expected false positives on radiometrically drifted synthetic data
- Processing time < 2 seconds for a 512x512 image pair

### 5.3 — `PROJECT/tests/test_api_routes.py`
Write FastAPI TestClient tests covering:
- `GET /health` returns 200
- `POST /api/analyze` with valid image returns 200 with expected response schema
- `POST /api/analyze` with invalid file returns 422
- `POST /api/change-detection` with two images returns 200
- `GET /api/demo/scenarios` returns all 5 scenarios
- Response time < 5 seconds per request in demo mode

---

## ═══ SECTION 6: BUILD THE RESEARCH DOCUMENTATION ═══

### 6.1 — `RESEARCH/literature_review.md`
Write a comprehensive, academically rigorous literature review (1500-2000 words) covering:

**Section 1 — Vision-Language Models for Remote Sensing (2023-2025)**
Cover these specific papers with accurate details:
- **RSGPT** (Hu et al., 2023) — first RS-specific VLM, trained on RSICap dataset, strong captioning
- **GeoChat** (Kuckreja et al., CVPR 2024) — grounded VLM, 318K instruction pairs, LLaVA-1.5 architecture, 7B parameters, supports region-level reasoning
- **EarthGPT** (Zhang et al., IEEE TGRS 2024) — universal multi-sensor model, handles optical + SAR
- **RemoteCLIP** (Liu et al., IEEE TGRS 2024) — CLIP-based foundation model for RS, strong zero-shot
- **SkyEyeGPT** (Zhan et al., ISPRS 2025) — instruction-tuning approach, unifies multiple RS tasks
- **LHRS-Bot** (Muhtar et al., ECCV 2024) — VGI-enhanced multimodal model
- **VHM** (Pang et al., AAAI 2025) — "Versatile and Honest" VLM, addresses hallucination in RS
- **GeoPixel / GeoPix** (2025) — pixel-level grounding for RS images

**Section 2 — Change Detection Methods**
Cover:
- Traditional methods: image differencing, PCA, CVA
- DL methods: Siamese networks, STSF-Net
- VLM-based change detection: ChangeChat, TEOChat
- Key challenge: pseudo-change suppression

**Section 3 — Indian/ISRO Context**
- ISRO satellite constellation: Cartosat series, RISAT, ResourceSat, EOS
- Why existing models fail on ISRO data: sensor response curves differ, training data is Western-centric, Indian geographic contexts underrepresented
- National applications: disaster management, agricultural monitoring, urban planning, forest mapping

**Section 4 — Research Gaps (What SatQuery Addresses)**
- Gap 1: No existing VLM is calibrated for ISRO sensor characteristics
- Gap 2: No system supports agentic tool use (routing queries to specialized modules)
- Gap 3: Existing change detection lacks automated pseudo-change suppression with confidence scoring
- Gap 4: No system provides explainable reasoning traces for non-expert users
- Gap 5: No system provides GeoJSON output directly compatible with ISRO GIS workflows

### 6.2 — `RESEARCH/our_innovations.md`
Write a detailed document covering SatQuery's 5 key innovations:

**Innovation 1: ISRO Sensor Calibration Module**
- Specific radiometric calibration constants for Cartosat-2S (0.65m resolution, panchromatic + multispectral), RISAT-1C (SAR C-band), ResourceSat-2A (LISS-III, AWiFS)
- Sensor-specific noise models and atmospheric correction parameters
- Impact: analysis accuracy improves by [expected %] on ISRO imagery vs uncalibrated models

**Innovation 2: Agentic Query Router**
- ReAct-style (Reasoning + Acting) agent that decomposes complex queries
- Routes to specialized tools: spectral indexing, change detection, object detection, segmentation
- Provides full execution trace (every step visible to user)
- Novel: unlike GeoChat which answers in one pass, SatQuery orchestrates multiple specialized tools

**Innovation 3: STSF-Net Inspired Pseudo-Change Suppression**
- Eliminates false positives from radiometric drift, illumination differences, seasonal variation
- Uses local spatial variance comparison (σ_T1 vs σ_T2) against mean signed difference μ_Δ
- Mathematically sound: reduces false positive rate vs naive differencing by ~30-40%

**Innovation 4: Bimodal Histogram Confidence Scoring**
- Novel confidence metric combining: Otsu inter-class variance ratio ω, valley-to-peak depth ratio v, area imbalance penalty p
- Score is physically meaningful (not just model softmax)
- Enables automated quality flagging: LOW confidence → human expert review triggered

**Innovation 5: ISRO-Native Output (GeoJSON + Report)**
- Outputs GeoJSON directly compatible with ISRO's VEDAS and Bhuvan GIS platforms
- Auto-generates structured reports matching ISRO's disaster response report format
- Timestamps, sensor IDs, bounding boxes all in ISRO's internal format spec

### 6.3 — `RESEARCH/isro_sensors_reference.md`
Write a detailed technical reference (formatted as a clean Markdown table + text) covering:
- Cartosat-2S: Launch 2017, orbit 505km, PAN 0.65m resolution, MS 2.1m, swath 9.6km, spectral bands 0.5-0.85μm
- Cartosat-3: Launch 2019, PAN 0.25m (world's highest from civilian Indian satellite at time of launch)
- RISAT-1C: SAR, C-band (5.35 GHz), VV+VH polarization, 3m resolution (FRS mode)
- ResourceSat-2A: LISS-III (23.5m, 4-band), AWiFS (56m), LISS-IV (5.8m)
- EOS-04 (RISAT-1A): SAR, L-band, 1m resolution, agriculture/forestry focus
- EOS-05 (GISAT-1A): Launched Sept 4 2026 on GSLV-F17, geosynchronous, hyperspectral (158 VNIR bands, 256 SWIR bands)
- Bhuvan portal and VEDAS (Visualization of Earth observation Data and Archival System) as access platforms

---

## ═══ SECTION 7: BUILD THE PRESENTATION ═══

### 7.1 — `PRESENTATION/slides.html`
Write a complete, professional Reveal.js presentation. Use CDN links for Reveal.js. The presentation must have exactly these slides in this order:

**Slide 1 — Title**
- Large: "SatQuery AI"
- Subtitle: "An Interactive Vision-Language Assistant for Multimodal Remote Sensing"
- Below: "SIH26167 | ISRO / Space Applications Centre"
- Design: dark background, ISRO logo placeholder (top right), satellite animation (CSS)

**Slide 2 — The Problem (Make Judges Feel It)**
- Headline: "India operates 15+ active satellites. But their data is locked away from the people who need it most."
- Three pain point cards:
  1. "A disaster manager in Kerala has satellite images of the flood. But needs a remote sensing PhD to interpret them. People are waiting."
  2. "A forest officer in Assam wants to know if the protected area shrank. The analysis takes weeks and thousands of rupees."
  3. "An agricultural officer needs to identify stressed crops before harvest fails. There's no simple tool."
- Bottom: "ISRO collects terabytes of data daily. Most of it goes unanalyzed."

**Slide 3 — Current Solutions & Why They Fail**
- Comparison table:
  | Tool | Limitation |
  | GeoChat (CVPR 2024) | Trained on Western datasets, no ISRO sensor support |
  | EarthGPT (IEEE 2024) | No agentic routing, no confidence scoring |
  | VEDAS (ISRO's own) | No natural language interface, requires experts |
  | Commercial tools | $$$, no ISRO integration, security concerns |
- Bottom: "None of them were built for India. None of them were built for ISRO."

**Slide 4 — Introducing SatQuery AI**
- One-liner: "Type a question. Upload a satellite image. Get the answer."
- Three pillars in large cards with icons:
  1. 🛰️ ISRO-Native — Calibrated for Cartosat, RISAT, ResourceSat, EOS
  2. 🤖 Agentic AI — Routes your query to the right analysis tool automatically
  3. 📊 Explainable — See every step, every confidence score, every data point
- Screenshot/mockup of the UI

**Slide 5 — System Architecture**
- Clean diagram showing:
  - User → Web UI → FastAPI Backend
  - Backend → Query Router → [4 parallel tools: VLM Engine | Change Detector | Spectral Index | Object Detector]
  - Tools → Report Generator → User
- Each component labeled with the technology (GeoChat-7B, FastAPI, React, Rasterio, etc.)
- ISRO Sensor Calibration Module highlighted in ISRO orange

**Slide 6 — The 5 Innovations (Technical Depth)**
- Five cards, each with: innovation name, the research gap it solves, and one key metric/result
- Present this as "Why SatQuery is not just a chatbot for satellite images"

**Slide 7 — Live Demo (3 slides)**
- Slide 7a: Kerala Flood Demo setup — "Before: Oct 10, 2023. After: Oct 12, 2023. Question: How much has the flood spread?"
- Slide 7b: Show the result — annotated image + "11.93 hectares flooded, 2 distinct inundation zones, Confidence: 97%"
- Slide 7c: Show the execution trace — "Here's exactly how we got that answer, step by step"

**Slide 8 — Applications & National Impact**
- Four use cases with statistics:
  1. Disaster Response — "Reduce analysis time from 3 days to 3 minutes for flood assessment"
  2. Agriculture — "Monitor 141 million farms across India with NDVI alerts"
  3. Forestry — "Real-time deforestation detection for 7,12,249 sq km of forest cover"
  4. Urban Planning — "Track unauthorized construction in 640+ districts"
- "ISRO already has the satellites. SatQuery makes the data usable."

**Slide 9 — Roadmap**
- Timeline with 3 phases:
  - Phase 1 (Done — Hackathon): Working demo, 5 sensor types, 6 query types, change detection
  - Phase 2 (3 months): Fine-tuned model on ISRO's actual archive data, Bengali/Hindi query support
  - Phase 3 (6 months): VEDAS integration, real-time alert system, mobile app

**Slide 10 — Team**
- 6 member boxes with roles
- All labeled: The presenter's name + "Presentation"
- All others labeled as "Core AI/ML Engineering" (since one person is doing all the core work)

**Slide 11 — Closing**
- Large: "SatQuery AI"
- "Making India's space data accessible to every Indian."
- ISRO mission quote (if available) or "India's satellites see everything. SatQuery AI understands it."
- QR code placeholder for GitHub repo
- Contact details

**Reveal.js configuration:**
- Theme: custom dark (as described)
- Transition: `slide`
- Add `data-auto-animate` to key slides
- Fragments (`.fragment`) used to reveal pain points one by one on Slide 2
- Speaker notes (`<aside class="notes">`) on EVERY slide with word-for-word speaker text

### 7.2 — `PRESENTATION/PRESENTER_SCRIPT.md`
Write a complete, word-for-word script for the presenting team member. This is written for someone who may not fully understand the technical details — every line must be simple, confident, and impactful. Include:

**Format for each slide:**
```
--- SLIDE [N]: [Title] ---
[What to say — verbatim, in simple English]
[Transition to next slide]
Time: [how many seconds/minutes to spend here]
```

**Key requirements for the script:**
- No jargon without explanation (when "VLM" is said, immediately follow with "that's a Vision-Language Model, which means an AI that understands both images and text")
- Every statistic has a source (even if paraphrased)
- Transitions are smooth and written out
- Anticipated interruptions handled (e.g., "I can show you more details after the presentation")
- Closing is memorized and powerful: [write a 45-second closing statement that references ISRO's mission and India's space ambitions]
- Total script fits 8-10 minutes (competition time limit)
- Includes cues like "[CLICK]" for slide advances, "[DEMO]" for live demonstration

### 7.3 — `PRESENTATION/JUDGE_QA_PREP.md`
Write answers to exactly 30 expected questions, organized by category:

**Category 1: Technical Questions (10 questions)**
1. What model exactly are you using? Why GeoChat and not GPT-4V or Gemini?
2. How does your system handle SAR images differently from optical?
3. What is pseudo-change suppression and why does it matter?
4. What training data did you use? Is it ISRO data?
5. How do you calculate confidence score?
6. What is the API latency? Can it handle real-time use?
7. Why FastAPI and not Django/Flask?
8. How do you handle images larger than 50MB?
9. Can this work offline, without internet?
10. What happens if the model hallucinates (gives a wrong answer)?

**Category 2: Innovation Questions (8 questions)**
11. How is this different from just using GPT-4V with a satellite image?
12. Isn't GeoChat already doing this? What's new?
13. What does "ISRO sensor calibration" mean exactly?
14. What's the novelty in your confidence scoring?
15. How is your routing agent different from a simple if-else classifier?
16. Can you prove the pseudo-change suppression works?
17. What research papers inspired this?
18. Did you test it on real ISRO images?

**Category 3: Impact Questions (6 questions)**
19. Who exactly are the end users?
20. How would ISRO deploy this? As a Bhuvan/VEDAS plugin?
21. What is the cost to run this at scale?
22. Does it work for Hindi or regional language queries?
23. How does this help in actual disaster response? Give a concrete story.
24. What are the limitations?

**Category 4: Team & Execution Questions (6 questions)**
25. How did you divide the work in 36 hours?
26. What would you do differently if you had 6 more months?
27. What technical challenges did you face and how did you overcome them?
28. Is the code available? Can we see it?
29. What external APIs or paid services does this use?
30. How would you get ISRO to actually adopt this?

**For each question, write:**
- The ideal answer (2-4 sentences, confident, specific)
- A fallback if you don't know: "That's an excellent question. In our current prototype, we handle it this way [...], and the full production solution would involve [...]."

### 7.4 — `PRESENTATION/PITCH_TIMING_GUIDE.md`
Write a detailed minute-by-minute guide:
```
00:00 - 00:30  Walk to podium, set up, breathe. Projector check.
00:30 - 01:00  Slide 1 - Title. Quick intro of team and problem.
01:00 - 02:30  Slide 2 - Problem. Three pain points with pauses for impact.
02:30 - 03:00  Slide 3 - Existing solutions. Be crisp.
03:00 - 03:45  Slide 4 - Our solution. The "aha" moment.
03:45 - 04:30  Slide 5 - Architecture. Show technical depth.
04:30 - 05:00  Slide 6 - Innovations. Name them, don't explain them all.
05:00 - 07:30  LIVE DEMO - Most important. Run Kerala flood demo.
07:30 - 08:00  Slide 8 - Impact.
08:00 - 08:30  Slide 9 - Roadmap. Brief.
08:30 - 09:00  Closing statement.
09:00 - 10:00  Q&A begins. Stay calm. Use JUDGE_QA_PREP.md answers.
```

### 7.5 — `PRESENTATION/VISUAL_DEMO_FLOW.md`
Write a step-by-step guide for running the live demo during the presentation:

```
DEMO SETUP (before presentation):
1. Open browser, navigate to localhost:8000 (or index.html if backend not needed)
2. Have the Kerala flood T1 and T2 images ready in /demo_data/
3. Test the full flow once. Screenshot results as backup.
4. Set DEMO_MODE = true in index.html (line 12) as failsafe

DEMO FLOW (live, during slide 7):
Step 1: Open the app — show the UI. "This is SatQuery AI."
Step 2: Upload T1 (before flood image). Point out: "It detects automatically it's a Cartosat-2S image."
Step 3: Enable change detection. Upload T2.
Step 4: Type the query: "How much has the flood spread between these two dates?"
Step 5: Press Analyze. [Show the animated loading state — "it's computing 12 pipeline stages"]
Step 6: Point to the annotated image: "Red regions = new flood water."
Step 7: Read out the answer: "11.93 hectares. 97% confidence. 2 flood zones."
Step 8: Click "Show Reasoning": Walk through 3 stages of the execution trace.
Step 9: Click "Download GeoJSON": "This goes directly into ISRO's GIS systems."

BACKUP PLAN (if demo fails):
- Screenshots of demo pre-loaded as PNG in /PRESENTATION/demo_screenshots/
- 3 slides with annotated screenshots ready to substitute
```

---

## ═══ SECTION 8: BUILD THE ARCHITECTURE DIAGRAMS ═══

### 8.1 — `DIAGRAMS/system_architecture.svg`
Create a clean, professional SVG diagram (800×600px minimum) showing:

**Components (use boxes with labels):**
- **User** (person icon, left side)
- **SatQuery Web UI** (browser icon, styled in dark navy)
- **FastAPI Backend** (server icon, center)
- **Query Router** (diamond/decision node)
- **VLM Engine** (brain icon, GeoChat-7B labeled)
- **Change Detector** (dual-image icon)
- **Spectral Index Engine** (graph icon)
- **Object Detector** (bounding box icon)
- **ISRO Sensor Calibration** (satellite icon, ISRO orange)
- **Report Generator** (document icon)
- **GeoJSON Output** (map icon)

**Connections:**
- Arrows with labels for data flow
- Parallel processing shown for the 4 tool modules

**Visual style:**
- Dark background (#0B1628)
- White text
- ISRO orange (#FF6B00) for critical path
- Satellite green (#00D4AA) for output/success nodes
- Clean, no decorative elements

### 8.2 — `DIAGRAMS/pipeline_flow.svg`
Create a vertical flowchart (600×900px) showing the 12-stage change detection pipeline:
1. Input Validation → 2. Co-registration → 3. Atmospheric Normalization → 4. Index Selection → 5. Index Computation → 6. Difference Map → 7. Pseudo-Change Suppression (STSF-Net) → 8. Gaussian Smoothing → 9. Otsu Thresholding → 10. Morphological Cleaning → 11. Region Analysis → 12. Confidence Scoring → Output

Each stage box has: stage number, name, timing (e.g., "~5ms"), and a one-line description. Color-coded: pre-processing stages blue, core analysis stages orange, output stages green.

---

## ═══ SECTION 9: BUILD SUPPORTING DOCUMENTS ═══

### 9.1 — `README.md`
Write a complete, professional README that:
- Has an ASCII art "SatQuery AI" header
- Problem statement reference: SIH26167
- One-paragraph description
- Architecture overview (text summary)
- Quick start guide:
  ```bash
  git clone [repo]
  cd satquery-ai/PROJECT/backend
  pip install -r requirements.txt
  # Set DEMO_MODE=true in .env for no-GPU demo
  uvicorn main:app --reload
  # Open PROJECT/frontend/index.html in browser
  ```
- API documentation link (`/docs`)
- Running tests: `pytest tests/ -v`
- Running demo: `python demo_data/generate_demo_images.py`
- Team section
- License (MIT)
- ISRO attribution

### 9.2 — `HACKATHON_CHECKLIST.md`
Write a submission checklist formatted as checkboxes covering:
- [ ] Working demo (frontend + backend running together)
- [ ] Demo data generated and preloaded
- [ ] All 5 demo scenarios tested end-to-end
- [ ] Presentation slides open correctly in browser
- [ ] Speaker has practiced full pitch at least 3 times
- [ ] JUDGE_QA_PREP.md reviewed by all team members
- [ ] GitHub repo created and code pushed
- [ ] README complete and accurate
- [ ] Tests all passing (run `pytest -v` and confirm green)
- [ ] Backend runs in DEMO_MODE without GPU
- [ ] Frontend works with backend offline (DEMO_MODE=true in JS)
- [ ] Presentation backup screenshots taken
- [ ] QR code for GitHub repo generated and added to slide 11
- [ ] Team introduction slide has all 6 members
- [ ] Report generator produces sensible output
- [ ] Change detection produces correct results for all 3 scenarios
- [ ] Architecture diagram renders cleanly on projector
- [ ] Contact email set up for post-hackathon follow-up

---

## ═══ SECTION 10: EXECUTION RULES FOR THE AGENT ═══

### Order of Operations
Execute in exactly this order. Do not skip steps. Do not reorder:

1. **First:** Create the entire directory structure (all directories, even empty ones with `.gitkeep`)
2. **Second:** Write `PROJECT/backend/requirements.txt` and `PROJECT/backend/config.py`
3. **Third:** Write all `PROJECT/backend/core/` modules in this order: `spectral_indices.py` → `image_processor.py` → `change_detector.py` → `query_router.py` → `vlm_engine.py` → `report_generator.py`
4. **Fourth:** Write `PROJECT/backend/api/schemas.py` and `PROJECT/backend/api/routes.py` and `PROJECT/backend/main.py`
5. **Fifth:** Write `PROJECT/demo_data/generate_demo_images.py` and `DEMO_SCENARIOS.json`
6. **Sixth:** Write all test files in `PROJECT/tests/`
7. **Seventh:** Write all `RESEARCH/` documents
8. **Eighth:** Write `PRESENTATION/PRESENTER_SCRIPT.md`, `JUDGE_QA_PREP.md`, `PITCH_TIMING_GUIDE.md`, `VISUAL_DEMO_FLOW.md`
9. **Ninth:** Write the main `PRESENTATION/slides.html` (Reveal.js, fully self-contained)
10. **Tenth:** Write `DIAGRAMS/system_architecture.svg` and `DIAGRAMS/pipeline_flow.svg`
11. **Eleventh:** Write `README.md` and `HACKATHON_CHECKLIST.md`
12. **Twelfth:** Write `PROJECT/frontend/index.html` (the full frontend)
13. **Thirteenth:** Write `PROJECT/docker-compose.yml`
14. **Final:** Run a verification pass — check that every file exists, every function is importable, tests can be discovered by pytest, and the frontend's JS DEMO_MODE is set to `true`

### Quality Standards — Every File Must Meet These

**Code quality:**
- Every Python file has a module-level docstring
- Every function has a docstring with args, returns, and raises
- No bare `except` clauses — always catch specific exceptions
- All file paths use `pathlib.Path`, never string concatenation
- Type hints on all function signatures
- `if __name__ == "__main__":` block where applicable for standalone testing

**Documentation quality:**
- Headers use proper Markdown hierarchy (# → ## → ###)
- Tables are properly formatted
- Code blocks have language tags (```python, ```bash, etc.)
- No broken links or placeholder text left unfilled (replace ALL `[...]` placeholders with actual content)

**Presentation quality:**
- Every slide has a concrete talking point, not just bullet points
- All speaker notes are complete sentences, not keywords
- The demo flow is written as if for someone who has never run the app before
- Every Q&A answer is specific enough that a non-technical judge is satisfied

### What Makes This Win SIH (The Agent Must Internalize This)

SIH judges look for FOUR things:
1. **Does it actually work?** — They will ask for a live demo. The demo MUST run without errors. DEMO_MODE ensures this regardless of hardware.
2. **Is there genuine innovation?** — Pseudo-change suppression + confidence scoring + agentic routing + ISRO sensor calibration = 4 specific, named innovations that no judge can dismiss as "just an API wrapper."
3. **Is the national impact real?** — Disaster management, agriculture, forestry, urban planning — these are India's real problems, tied to ISRO's actual mission. The presentation must connect the tech to the mission.
4. **Is the team credible?** — The speaker must sound confident, not nervous. The Q&A answers must be specific, not vague. The JUDGE_QA_PREP.md is as important as the code.

**The one sentence that wins SIH:**
*"ISRO spends thousands of crores building satellites and collecting data. SatQuery AI makes that data usable by every officer, farmer, and disaster responder in India — not just PhD scientists."*

Use variations of this in the pitch, the closing statement, and the Q&A answers.

---

## ═══ SECTION 11: SENSOR-SPECIFIC DETAILS (Reference While Building) ═══

Use these exact technical details when writing sensor calibration code and documentation:

### Cartosat-2S
- Launch: May 2017 | Orbit: 505km sun-synchronous
- Panchromatic: 0.65m resolution, 9.6km swath, 500-850nm
- Multispectral: 2.1m, bands: Blue 450-520nm, Green 520-590nm, Red 620-690nm, NIR 770-860nm
- Radiometric resolution: 10-bit
- Revisit: 4 days

### RISAT-1C  
- SAR, C-band (5.35 GHz), incidence angle 20°-55°
- Fine Resolution Stripmap (FRS-1): 3m, single pol
- Medium Resolution ScanSAR (MRS): 25m, dual pol (VV+VH)
- Used for: agriculture, geology, flood mapping

### ResourceSat-2A
- LISS-III: 23.5m, 4-band (VNIR + SWIR), 141km swath
- AWiFS: 56m, 4-band, 740km swath (wide area coverage)
- LISS-IV: 5.8m, 3-band (multispectral) or 1-band (pan), 23km swath
- Primary use: agriculture, forestry, land-use mapping

### EOS-04 (RISAT-1A)
- SAR, L-band (1.27 GHz)
- 1m resolution (High Resolution mode)
- Dual polarization (HH+HV or VV+VH)
- Best for: agriculture (penetrates vegetation), soil moisture

### EOS-05 (GISAT-1A)  
- Launched: September 4, 2026 on GSLV-F17
- Geosynchronous (GEO) orbit — first Indian GEO imaging satellite of this class
- Multispectral VNIR: 6 bands, 42m GSD, 0.45-0.875μm
- Hyperspectral VNIR: 158 bands, 318m GSD
- Hyperspectral SWIR: 256 bands, 191m GSD, 0.9-2.5μm
- This is the newest ISRO satellite — mention it to impress judges

---

## ═══ SECTION 12: FINAL VERIFICATION COMMANDS ═══

After completing all files, run these commands and ensure they succeed:

```bash
# Verify directory structure
find satquery-ai/ -type f | sort

# Install and test backend
cd satquery-ai/PROJECT/backend
pip install -r requirements.txt
python -c "from core.spectral_indices import compute_ndvi; print('Spectral OK')"
python -c "from core.change_detector import ChangeDetector; print('Change Detector OK')"
python -c "from core.query_router import QueryRouter; print('Router OK')"
python -c "from main import app; print('App OK')"

# Run tests
cd satquery-ai/PROJECT
pytest tests/ -v --tb=short

# Generate demo data
python demo_data/generate_demo_images.py

# Verify demo data generated
ls demo_data/*.tif

# Start backend in demo mode
DEMO_MODE=true uvicorn backend.main:app --reload &

# Test API
curl http://localhost:8000/health

# Verify presentation
python -c "
import os
required_files = [
    'PRESENTATION/slides.html',
    'PRESENTATION/PRESENTER_SCRIPT.md', 
    'PRESENTATION/JUDGE_QA_PREP.md',
    'PRESENTATION/PITCH_TIMING_GUIDE.md',
    'PRESENTATION/VISUAL_DEMO_FLOW.md',
    'RESEARCH/literature_review.md',
    'RESEARCH/our_innovations.md',
    'DIAGRAMS/system_architecture.svg',
    'README.md',
    'HACKATHON_CHECKLIST.md'
]
for f in required_files:
    assert os.path.exists(f), f'MISSING: {f}'
    assert os.path.getsize(f) > 1000, f'TOO SMALL (likely empty): {f}'
print('All required files present and non-empty. ✅')
"
```

If any verification fails, fix the issue before considering the task complete.

---

## ═══ APPENDIX: THE PITCH THEY WILL NEVER FORGET ═══

When the speaker finishes the demo and reaches the closing slide, they must say exactly this (memorized):

*"India has sent missions to Mars. India has landed on the Moon's south pole. India's satellites photograph every corner of this country, every single day.*

*But right now, a flood relief officer sitting in a control room in Kerala has satellite images on his screen — and he cannot tell where the water has spread, because reading satellite data requires years of specialized training he doesn't have.*

*SatQuery AI changes that. He types a question. He gets an answer. In three minutes, not three days.*

*ISRO doesn't need more data. It needs more people who can use the data it already has. SatQuery AI is that bridge.*

*Thank you."*

Then wait. Do not say anything else. The silence after this closing will be more powerful than any additional word.

---

*End of Prompt. Begin execution immediately. Every file. No shortcuts. Win this.*
