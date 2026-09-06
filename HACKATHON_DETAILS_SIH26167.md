# Smart India Hackathon (SIH 2026) — Problem Statement SIH26167
## Official Reference & Hackathon Context Specification

---

### Basic Information
* **Problem Statement ID:** `SIH26167`
* **Problem Statement Title:** `SatQuery AI - An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries`
* **Organization:** Indian Space Research Organisation (ISRO)
* **Department:** Department of Space / Indian Space Research Organisation / Space Applications Centre (SAC), Ahmedabad
* **Category:** Software
* **Theme:** Space Technology

---

### Description & Background
Remote-sensing imagery is widely used for:
* Agricultural monitoring
* Disaster management
* Urban planning
* Forest monitoring
* Water-resource assessment
* Infrastructure mapping
* Environmental analysis

However, most existing remote-sensing AI solutions are developed as **isolated applications for a single predefined task**, such as land-cover classification, object detection, visual question answering, or change detection. These systems often require users to understand satellite-data characteristics, GIS workflows, model selection, and task-specific parameters. Consequently, non-expert users may find it difficult to obtain meaningful information from satellite imagery through simple natural-language queries.

Many operational remote-sensing questions cannot always be answered reliably using a single optical image. Relevant information may be distributed across paired or multiple observations acquired at different times or by different sensors:
* **Optical and multispectral imagery** provides spectral and contextual information (color, surface reflection, phenology).
* **Synthetic Aperture Radar (SAR)** provides complementary structural information (surface roughness, moisture, dielectric properties, volume scattering) and supports day-and-night acquisition through cloud cover and atmospheric haze.
* **Multitemporal image pairs** are required to identify and interpret changes over time.
* **Co-registered optical–SAR pairs** can provide more complete and reliable information than either modality alone.

A general-purpose large language model (LLM) or vision-language model (VLM) cannot be expected to perform these specialised tasks reliably without adaptation to remote-sensing imagery, sensor characteristics, and domain-specific terminology. The proposed solution must therefore include remote-sensing fine-tuning or domain adaptation and may employ multiple specialised models for different tasks:
* **`BigEarthNet.txt`** serves as the primary dataset for adapting image–text representations to multisensor remote-sensing data (co-registered Sentinel-1 SAR + Sentinel-2 multispectral imagery with rich text annotations).
* **`VRSBench`** and **`RSVQA`** are used to evaluate single-image captioning, grounding, and visual question answering.
* **`CDVQA`** is used to evaluate multitemporal change-based visual question answering.

The novelty of **SatQuery AI** lies in its **agentic, query-driven framework**. Instead of applying a single generic VLM, the system:
1. Validates inputs and compatibility across sensor modalities.
2. Selects and sequences suitable remote-sensing specialist models from a registry.
3. Configures only permitted task parameters.
4. Executes the selected workflow.
5. Combines textual and spatial outputs, estimates confidence, and returns an evidence-grounded response.
6. Produces an **auditable execution summary** detailing selected tasks, model/tool names, and key parameters.

---

### Defined Input Scope
1. **Single Image**:
   * One optical/multispectral or SAR image.
   * Tasks: Captioning / scene description, visual question answering (VQA), and text-guided region grounding.
2. **Cross-Modal Pair**:
   * Co-registered optical/multispectral and SAR images of the same geographic area.
   * Tasks: Joint information extraction, feature fusion, and cross-modal complementary analysis (e.g., optical multispectral context + SAR structural penetrate/water backscatter).
3. **Bi-Temporal Pair**:
   * Two spatially corresponding images of the same geographic area acquired at different dates/times ($T_1$ and $T_2$).
   * Tasks: Change detection, spatial change map generation, change description, and change-based visual question answering (CDVQA).
4. **Supported Formats**:
   * **GeoTIFF or TIFF** for operational geospatial imagery (supporting multi-band, coordinate metadata, CRS, GSD).
   * **PNG and JPEG** inputs accepted only for the prescribed public benchmark datasets.

---

### Mandatory Functional Scope
* **Remote-Sensing Adaptation**: At least one visual or vision-language component must be fine-tuned or adapted using `BigEarthNet.txt` or open-source training data.
* **Single-Image Baseline**: Visual question answering (VQA) is mandatory. The solution must additionally implement either captioning/scene description or text-guided region grounding (SatQuery AI implements both).
* **Multi-Image Change Analysis**: Change description or change-based visual question answering (CDVQA) from a bi-temporal image pair is mandatory. A spatial change map must also be generated where reference masks/bands are available.
* **Cross-Modal Pair Analysis**: The system must extract complementary information from a co-registered optical/multispectral and SAR image pair.
* **Agentic Orchestration**: The system must automatically select, sequence, and execute the appropriate specialist models or tools according to the query and input configuration.

---

### Representative Queries
1. **Single Image Captioning / Description**:
   > *"Describe the land-cover and major objects visible in this image."*
2. **Single Image Grounding**:
   > *"Highlight the water body referred to in the query."*
3. **Bi-Temporal Change Analysis & Spatial Mapping**:
   > *"What changed between these two dates, and where did the change occur?"*
4. **Cross-Modal Optical-SAR Joint Extraction**:
   > *"Use the optical and SAR images together to identify built-up and water-covered regions."*
5. **Bi-Temporal Change-VQA (CDVQA)**:
   > *"Has the built-up area increased, decreased, or remained unchanged?"*

---

### Agentic Model & Tool Orchestration Specification
The controller must:
1. **Interpret query** and classify the requested task into one of the canonical categories.
2. **Check input compatibility**:
   * Number of images (single vs pair).
   * Modality (Optical vs SAR vs Multi-band).
   * Format (GeoTIFF/TIFF vs PNG/JPEG).
   * Metadata (Spatial resolution, dimensions, bounding box / CRS alignment).
3. **Select one or more models or tools** from a predefined registry (e.g. Spectral Index Engine, 12-Stage Change Engine, Optical-SAR Fusion Engine, SAM Segmentor, DOTA Object Counter, GeoChat-7B VLM).
4. **Configure only permitted task parameters** and execute the selected pipeline.
5. **Combine textual and spatial outputs**, estimate confidence, and return visual evidence (bounding boxes, masks, GeoJSON, heatmaps).
6. **Provide an auditable execution summary**:
   * Selected task
   * Invoked models/tool names
   * Key permitted parameters
   * Execution latency and confidence metrics
   *(Note: The controller may perform internal task planning; however, only the observable execution trace will be evaluated).*

---

### Evaluation and Judging Criteria
* **Public Benchmarks**:
  * Evaluated on prescribed test splits:
    * `BigEarthNet.txt` — Remote-sensing adaptation (Sentinel-1 SAR + Sentinel-2 Optical) [arXiv:2603.29630](https://arxiv.org/abs/2603.29630).
    * `VRSBench` — Single-image captioning, grounding, and VQA.
    * `RSVQA` — Visual question answering on high-resolution and low-resolution satellite scenes.
    * `CDVQA` — Change-based visual question answering on bi-temporal pairs.
* **ISRO / SAC Evaluation Set**:
  * Co-registered **Cartosat-2S optical** (0.65m Pan / 2m MS) and **RISAT SAR** (C-band, VV/VH polarimetric) image pairs.
  * Task-specific reference answers, labels, bounding boxes, or masks.
  * Pre-georeferenced and co-registered for evaluating real-world Indian operational performance.
