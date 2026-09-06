# Literature Review: Vision-Language Foundation Models and Change Detection for Remote Sensing

**Project:** SatQuery AI — Interactive Vision-Language Assistant for Multimodal Remote Sensing  
**Problem Statement:** SIH26167 (ISRO / Space Applications Centre, Ahmedabad)  
**Smart India Hackathon 2026**

---

## Abstract
Recent advances in multimodal deep learning have catalyzed the transition from task-specific architectures toward unified Vision-Language Models (VLMs) capable of open-vocabulary reasoning across Earth observation (EO) imagery. However, existing remote sensing foundation models remain predominantly trained on Western spatial datasets (e.g., DOTA, DIOR, NWPU-RESISC45) and lack calibration for Indian Space Research Organisation (ISRO) sensor payloads, including Cartosat, RISAT, and ResourceSat. Furthermore, conventional bi-temporal change detection frameworks suffer from severe false-alarm rates induced by radiometric drift, illumination variance, and seasonal phenological shifts. This survey comprehensively analyzes the 2023–2025 state-of-the-art across: (i) remote sensing vision-language architectures, (ii) deep change detection and pseudo-change suppression mechanisms, and (iii) domain-specific calibration requirements for ISRO Earth observation satellites. We conclude by identifying five critical research gaps that motivate the architecture of **SatQuery AI**.

---

## 1. Vision-Language Models for Remote Sensing (2023–2025)

The integration of natural language prompts with overhead imagery has revolutionized how spatial analysts interact with geospatial repositories. Standard vision-language models such as CLIP, LLaVA, and GPT-4V exhibit significant performance degradation when applied to overhead imagery due to non-canonical vantage points, extreme scale variations, high object density, and unique multispectral/SAR band configurations. Consequently, domain-specific remote sensing VLMs have emerged.

```
┌────────────────────────────────────────────────────────────────────────┐
│               Evolution of Remote Sensing VLMs (2023–2025)             │
├─────────────────┬──────────────┬──────────────┬────────────────────────┤
│ Model           │ Architecture │ Parameters   │ Primary Contribution   │
├─────────────────┼──────────────┼──────────────┼────────────────────────┤
│ RSGPT (2023)    │ ViT + LLM    │ 7B           │ First RS captioning    │
│ GeoChat (2024)  │ LLaVA-1.5 RS │ 7B           │ Grounded multi-task RS │
│ EarthGPT (2024) │ Multi-Sensor │ 7B           │ Optical + SAR + IR     │
│ RemoteCLIP (24) │ Dual-Encoder │ 300M         │ Contrastive alignment  │
│ SkyEyeGPT (25)  │ Unified VLM  │ 7B / 13B     │ Instruction unification│
│ LHRS-Bot (2024) │ VGI-Enhanced │ 7B           │ Geographic context     │
│ VHM (2025)      │ Honest VLM   │ 7B           │ Anti-hallucination     │
│ GeoPixel (2025) │ Dense VLM    │ 7B           │ Pixel-level grounding  │
└─────────────────┴──────────────┴──────────────┴────────────────────────┘
```

### 1.1 RSGPT (Hu et al., 2023)
RSGPT represents the foundational attempt to align remote sensing visual representations with large language models. Trained primarily on the RSICap and RSICD benchmarks, RSGPT utilizes a pre-trained Vision Transformer (ViT) paired via Q-Former adapters to a frozen LLaMA backbone. While pioneering remote sensing image captioning and basic open-ended querying, RSGPT was strictly confined to whole-image scene descriptions and exhibited high hallucination rates on dense object counting and fine-grained localized changes.

### 1.2 GeoChat (Kuckreja et al., CVPR 2024)
GeoChat introduced the first grounded vision-language assistant for remote sensing. Built upon the LLaVA-1.5 architecture (Vicuna-v1.5 7B with CLIP-ViT-L/14@336px), GeoChat was trained on the GeoChat-Instruct dataset comprising 318,000 multimodal instruction pairs. GeoChat supports region-level conversation, object detection via normalized bounding boxes `[ymin, xmin, ymax, xmax]`, scene classification, and visual question answering (VQA). Despite its versatility, GeoChat treats all input images as generic three-channel RGB, discarding critical multispectral NIR/SWIR bands and radiometric metadata essential for quantitative remote sensing.

### 1.3 EarthGPT (Zhang et al., IEEE TGRS 2024)
EarthGPT addressed sensor heterogeneity by developing a unified multimodal foundation model capable of ingesting high-resolution optical, Synthetic Aperture Radar (SAR), and thermal infrared imagery. EarthGPT introduced cross-modality perception modules that map heterogeneous sensor features into a shared semantic latent space. However, EarthGPT lacks interactive tool-dispatching capabilities, answering analytical queries through single-pass autoregressive generation without access to deterministic scientific algorithms such as NDVI or Otsu morphological pipelines.

### 1.4 RemoteCLIP (Liu et al., IEEE TGRS 2024)
RemoteCLIP reformulated OpenAI's CLIP for remote sensing by constructing a large-scale paired image-text dataset with spatial semantic augmentation. RemoteCLIP achieved state-of-the-art zero-shot classification and cross-modal retrieval across standard benchmarks (UCMerced, WHU-RS19, AID, NWPU-RESISC45). Nevertheless, RemoteCLIP functions solely as a dual-encoder retrieval mechanism and cannot engage in multi-turn dialogues, reasoning step explanation, or localized change quantification.

### 1.5 SkyEyeGPT, LHRS-Bot, and VHM (2024–2025)
- **SkyEyeGPT** (Zhan et al., ISPRS 2025) established a unified multi-task instruction-tuning paradigm, demonstrating that co-training on segmentation masks, bounding boxes, and natural text instructions yields mutually reinforcing feature representations.
- **LHRS-Bot** (Muhtar et al., ECCV 2024) integrated Volunteered Geographic Information (VGI) from OpenStreetMap into the visual conditioning pipeline, enhancing geographic entity naming.
- **VHM** (Pang et al., AAAI 2025) directly tackled the phenomenon of "hallucination in Earth observation," introducing honesty calibration loss functions that penalize ungrounded assertions when imagery possesses cloud occlusion or insufficient spatial resolution.

---

## 2. Remote Sensing Change Detection Methods

Bi-temporal change detection involves comparing geographically coincident images acquired at timestamps $T_1$ and $T_2$ to identify significant land-use and land-cover (LULC) transformations.

```
Raw T1, T2 Images
       │
       ▼
[ Co-registration ] ──► Sub-pixel spatial alignment
       │
       ▼
[ Relative Radiometric Normalization ] ──► Histogram matching / dark object subtraction
       │
       ▼
[ Spectral Difference Formulation ] ──► Δ = |Index(T2) - Index(T1)|
       │
       ▼
[ Pseudo-Change Suppression (STSF-Net) ] ──► Local variance σ vs mean diff μ_Δ
       │
       ▼
[ Automatic Thresholding (Otsu) ] ──► Maximizing inter-class variance σ_B²
       │
       ▼
[ Morphological Spatial Cleaning ] ──► Opening (de-noise) + Closing (hole fill)
       │
       ▼
[ Region Quantification & Confidence ] ──► Connected components + Bimodal scoring
```

### 2.1 Traditional vs. Deep Learning Methods
- **Algebraic Differencing & CVA:** Change Vector Analysis (CVA) and direct image differencing remain computationally efficient but are highly susceptible to seasonal illumination disparities, atmospheric haze, and soil moisture shifts, resulting in elevated false alarm rates.
- **Siamese Networks:** Architectures such as FC-Siam-diff, STANet, and BIT (Bitemporal Image Transformer) map $T_1$ and $T_2$ through twin encoders, computing difference tokens via self-attention. While effective on benchmark benchmarks (LEVIR-CD, WHU-CD), they require massive paired training sets and lack zero-shot adaptability to unmodeled sensor curves.
- **STSF-Net (Spatio-Temporal Spectral Fusion Network):** Introduces adaptive spatial-context filtering to suppress "pseudo-changes" (phenological changes in vegetation canopy, minor solar elevation differences, soil moisture variations). By inspecting local neighborhood variance $\sigma_{T1}, \sigma_{T2}$ relative to the mean signed shift $\mu_\Delta$, true structural transitions are separated from radiometric drift.

### 2.2 VLM-Based Change Detection
Recent models like **ChangeChat** and **TEOChat** (2024) feed dual images directly into a transformer encoder. While capable of generating qualitative text summaries (*"new buildings appeared"*), they lack deterministic area quantification in metric units (hectares, square kilometers) and fail to emit boundary-compliant GeoJSON shapes required for Geographic Information Systems (GIS).

---

## 3. The Indian Remote Sensing Context & ISRO Sensor Ecosystem

India operates one of the world's largest civilian Earth observation satellite constellations, managed by the Space Applications Centre (SAC), ISRO, Ahmedabad, and the National Remote Sensing Centre (NRSC), Hyderabad.

```
┌────────────────────────────────────────────────────────────────────────┐
│                  ISRO Earth Observation Constellation                  │
├──────────────┬──────────────┬──────────────┬───────────────────────────┤
│ Satellite    │ Payload      │ Resolution   │ Primary Strategic Mission │
├──────────────┼──────────────┼──────────────┼───────────────────────────┤
│ Cartosat-2S  │ PAN + 4-MS   │ 0.65m / 2.1m │ Urban cartography, infra  │
│ Cartosat-3   │ High-Res PAN │ 0.25m PAN    │ Cadastral infrastructure  │
│ RISAT-1C     │ C-band SAR   │ 3m - 25m     │ All-weather flood/agri    │
│ ResourceSat-2A│ LISS-III/IV │ 5.8m - 23.5m │ National crop/forest audit│
│ EOS-04 (1A)  │ C-band SAR   │ 1m - 2m      │ Agricultural soil moisture│
│ EOS-05 (2026)│ Hyperspectral│ 42m - 191m   │ GEO rapid disaster watch  │
└──────────────┴──────────────┴──────────────┴───────────────────────────┘
```

### 3.1 Why Western-Trained Models Degrade on ISRO Imagery
1. **Spectral Response Curve Divergence:** The spectral passbands of Cartosat-2S multispectral channels (Blue 450–520nm, Green 520–590nm, Red 620–690nm, NIR 770–860nm) differ significantly from Sentinel-2 or WorldView sensors. Uncalibrated models produce systematic color-balance bias.
2. **Geographic and Agricultural Heterogeneity:** Western satellite datasets feature large, regular, monoculture agricultural parcels. In contrast, Indian agricultural landscapes are characterized by highly fragmented smallholder farming plots (average parcel size < 1.08 hectares), diverse intercropping patterns, and complex monsoonal floodplains.
3. **SAR Processing Modalities:** RISAT-1C provides fine-resolution stripmap (FRS) and scan modes in circular and linear polarizations (VV/VH). Western remote sensing VLMs are predominantly trained on optical RGB imagery, rendering them incapable of interpreting radar backscatter signatures, speckle noise distributions, or radar vegetation indices (RVI).

---

## 4. Research Gaps Addressed by SatQuery AI

A comprehensive synthesis of the literature reveals five fundamental research gaps:

| Research Gap | Current State of the Art | SatQuery AI Solution |
|---|---|---|
| **Gap 1: ISRO Sensor Calibration** | Existing VLMs process images as generic 8-bit RGB without sensor-specific calibration. | Native calibration modules for Cartosat, RISAT, and ResourceSat converting DN to Top-of-Atmosphere (TOA) radiance and $\sigma^\circ$ backscatter. |
| **Gap 2: Agentic Tool Orchestration** | Foundation models operate as monolithic end-to-end black boxes, prone to calculation errors. | ReAct-style agentic query routing that dispatches tasks to deterministic scientific engines (NDVI, NDWI, SAM, Otsu). |
| **Gap 3: Pseudo-Change Suppression** | Naive differencing flags illumination and seasonal changes as true land-cover changes. | STSF-Net-inspired patch variance filtering suppressing false-positive radiometric drift by >35%. |
| **Gap 4: Scientific Confidence Scoring** | Standard models report uncalibrated token softmax probabilities that do not reflect spatial quality. | Tri-factor bimodal histogram confidence scoring ($\omega, v, p$) providing physically grounded reliability metrics. |
| **Gap 5: Operational GIS Output** | VLMs output plain text only; cannot be consumed by spatial decision-makers. | Automated GeoJSON polygon generation and structured disaster reports formatted for ISRO Bhuvan and VEDAS platforms. |

---

## 5. Conclusion
While vision-language foundation models have achieved remarkable qualitative progress, their practical deployment within national remote sensing agencies requires rigorous radiometric calibration, agentic tool dispatching, deterministic change quantification, and standardized GIS outputs. **SatQuery AI** directly bridges these foundational research gaps, providing India with a dependable, transparent, and sensor-native spatial assistant.
