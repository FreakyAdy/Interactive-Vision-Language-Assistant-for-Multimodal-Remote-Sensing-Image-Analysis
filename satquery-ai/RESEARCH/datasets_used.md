# Datasets Used for Training, Benchmarking & Validation

**Project:** SatQuery AI (SIH26167)  
**Host Organisation:** Indian Space Research Organisation (ISRO) / Space Applications Centre (SAC)

---

## 1. Overview of Data Architecture

SatQuery AI leverages a hybrid training and evaluation methodology combining large-scale academic remote sensing foundation benchmarks with synthetic ISRO sensor datasets and historical disaster imagery.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                SUMMARY OF DATASETS USED                                        │
├─────────────────┬────────────────┬────────────────┬────────────────┬───────────────────────────┤
│ Dataset         │ Modality       │ Spatial Res.   │ Volume / Pairs │ Operational Role in SatQuery│
├─────────────────┼────────────────┼────────────────┼────────────────┼───────────────────────────┤
│ GeoChat-Instruct│ Optical (RGB)  │ 0.5m - 15m     │ 318,000 pairs  │ Base VLM multi-task tuning│
│ DOTA-v2.0       │ Optical (Aerial│ 0.1m - 1.0m    │ 1.7M instances │ Object detector training  │
│ LEVIR-CD        │ Bi-temporal Opt│ 0.5m           │ 637 pairs      │ Building change validation│
│ WHU-CD          │ Aerial Bi-temp │ 0.2m           │ High-res pairs │ Cadastral urban change    │
│ SEN12-CD        │ Sentinel-1 SAR │ 10m - 20m      │ 4,000+ pairs   │ All-weather SAR change    │
│ ISRO-Synthetic  │ Calibrated Opt │ 0.25m - 2.0m   │ 5 Scenarios    │ Zero-GPU Hackathon Demo   │
└─────────────────┴────────────────┴────────────────┴────────────────┴───────────────────────────┘
```

---

## 2. Dataset Profiles

### 2.1 GeoChat-Instruct (CVPR 2024)
- **Source:** Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)
- **Modality:** High-resolution optical satellite imagery (derived from Google Earth, PlanetScope, and aerial surveys).
- **Scale:** 318,000 instruction-following pairs.
- **Role:** Forms the foundation weights for SatQuery AI's natural language conversational assistant. Enables localized reasoning, grounded bounding box generation `[ymin, xmin, ymax, xmax]`, and multi-turn geographic dialogue.

### 2.2 DOTA-v2.0 (Dataset for Object Detection in Aerial Images)
- **Source:** Wuhan University & CAPE
- **Scale:** 11,268 satellite images, 1,793,658 annotated instances across 18 categories.
- **Classes:** Plane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, large vehicle, small vehicle, helicopter, roundabout, soccer ball field, swimming pool, container crane, airport, helipad.
- **Role:** Calibrates SatQuery AI's `ObjectDetector` for oriented and horizontal bounding box counting.

### 2.3 LEVIR-CD & WHU-CD (Bi-Temporal Change Detection)
- **LEVIR-CD:**
  - 637 very high resolution (0.5m) Google Earth image pairs ($1024 \times 1024$).
  - Spans 5 to 14 years of urban development (housing, villa clusters, road construction).
  - Used to benchmark the 12-stage pipeline against STSF-Net pseudo-change suppression.
- **WHU-CD:**
  - Aerial dataset covering Christchurch, New Zealand before and after the 2011 earthquake ($0.2\text{m}$ GSD).
  - Used to validate structural collapse and rapid disaster assessment.

### 2.4 ISRO Calibrated Synthetic Scenarios (Hackathon Gold Benchmark)
To guarantee 100% reproducible judging evaluations without relying on external internet connectivity or proprietary ISRO Level-1 data transfers during SIH, five calibrated synthetic scenarios were engineered using physics-based radiative transfer approximations:

1. **Kerala Floods (2023) — Periyar River Basin:**
   - Multi-temporal optical pair mimicking Cartosat-2S (0.65m PAN / 2.1m MS).
   - Inundation ground truth: 11.93 hectares across 2 distinct hydrological pooling zones.
2. **Assam Forest Reserve Canopy Loss:**
   - ResourceSat-2A LISS-IV (5.8m) multispectral pair.
   - Ground truth: 4.2 hectares cleared via linear logging road.
3. **Delhi NCR Peri-Urban Expansion:**
   - Cartosat-3 sub-meter (0.25m PAN / 1.13m MS) pair.
   - Ground truth: 2.8 hectares of new impervious structural footprint.
4. **Agricultural Field Health & Vigor:**
   - ResourceSat-2A LISS-III (23.5m) multispectral scene with NDVI gradient (0.20 to 0.72) and irrigation channels.
5. **Maritime Harbor Vessel Surveillance:**
   - RISAT-1C C-band SAR scene with multi-look speckle simulation and 5 commercial container vessels.
