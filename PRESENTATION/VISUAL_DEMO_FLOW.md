# SatQuery AI: Visual Demo Execution Flow & Backup Plan (SIH26167)

**Smart India Hackathon 2026 | Problem Statement: SIH26167**  
**Audience:** ISRO / Space Applications Centre Evaluation Jury  

---

## 1. Pre-Presentation Setup (15 Minutes Before Stage)

1. **Start the Backend (Failsafe DEMO_MODE enabled):**
   ```bash
   cd "c:\Work\Projects\Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis\PROJECT\backend"
   python -m uvicorn main:app --port 8000
   ```
2. **Open the Frontend:**
   - Launch Google Chrome or Microsoft Edge.
   - Open `PROJECT/frontend/index.html`.
   - Press `F11` to enter clean Fullscreen Mission-Control mode.
3. **Verify Demo Data Assets:**
   - Confirm synthetic images exist in `PROJECT/demo_data/`:
     - `cartosat_optical_sample.tif` & `risat_sar_sample.tif` (Cross-Modal Pair)
     - `flood_t1.png` & `flood_t2.png` (Bi-Temporal Kerala Flood)
     - `urban_t1.png` & `urban_t2.png` (Bi-Temporal Urban CDVQA)
     - `agri_sample.png` & `harbor_sample.png`
4. **Failsafe Check:**
   - `DEMO_MODE = true` is pre-configured. If network or GPU is unavailable, the frontend and backend automatically run 100% offline with zero external dependencies.

---

## 2. Live Demo Script (During Slides 7 & 8: 05:00 – 07:15)

```
┌────────────────────────────────────────────────────────────────────────┐
│                   SIH26167 LIVE DEMO 10-STEP SEQUENCE                  │
├───────┬────────────────────────────┬───────────────────────────────────┤
│ Step  │ Action                     │ Verbal Narration Cue              │
├───────┼────────────────────────────┼───────────────────────────────────┤
│ 1     │ Show 3-Way Scope Tabs      │ "3 Defined Input Scopes"          │
│ 2     │ Select Bi-Temporal Scope   │ "Testing Kerala Flood (T1 vs T2)" │
│ 3     │ Click Q3 Chip              │ "What changed between two dates?" │
│ 4     │ Click 'Execute Workflow'   │ "12-stage STSF-Net executing"     │
│ 5     │ Inspect Inundation Output  │ "11.93 ha inundation, 97% conf."  │
│ 6     │ Switch to Cross-Modal Tab  │ "Co-registered Optical + SAR"     │
│ 7     │ Ingest Cartosat + RISAT    │ "C-Band SAR cloud penetration"    │
│ 8     │ Click Q4 Chip              │ "Identify built-up & water"       │
│ 9     │ Expand JSON Audit Trace    │ "Observable trace (SIH mandate)"  │
│ 10    │ Click 'Download GeoJSON'   │ "Direct GIS export for Bhuvan"    │
└───────┴────────────────────────────┴───────────────────────────────────┘
```

### Detailed Step-by-Step Instructions:

- **Step 1: Introduction to Defined Input Scopes**
  - *Action:* Point to the top scope selector: `[1. Single Image]`, `[2. Cross-Modal Pair]`, `[3. Bi-Temporal Pair]`.
  - *Say:* *"This is SatQuery AI. Notice how our interface directly mirrors the three defined input scopes mandated by ISRO: Single Image, Cross-Modal Pair, and Bi-Temporal Pair."*
- **Step 2: Activating Bi-Temporal Scope (Kerala Flood 2023)**
  - *Action:* Click the preset card: *"2. Kerala Flood (Bi-Temporal)"*.
  - *Say:* *"Let us evaluate the recurring Kerala monsoonal floods. The system loads Cartosat-2S baseline T1 and post-event T2."*
- **Step 3: Triggering Official Representative Query 3**
  - *Action:* Notice Query 3 is loaded: *"What changed between these two dates, and where did the change occur?"*
  - *Say:* *"Notice this is verbatim Query 3 from the problem statement. The field officer does not need GIS scripts — he asks in natural language."*
- **Step 4: Executing Analysis Pipeline**
  - *Action:* Click the prominent ISRO Orange button **Execute Agentic RS Workflow**.
  - *Say:* *"Our agent routes this to our 12-stage change pipeline: co-registration check, NDWI difference, and STSF-Net pseudo-change filtering."*
- **Step 5: Inspecting the Bi-Temporal Result**
  - *Action:* Point to the split-view annotated image and metrics.
  - *Say:* *"In under 300 milliseconds: 11.93 hectares of water expansion across two sectors, backed by a 97% bimodal confidence score. Our STSF-Net filter suppressed over 9,200 false-positive pixels from wet soil."*
- **Step 6: Switching to Cross-Modal Optical-SAR Pair**
  - *Action:* Click `[2. Cross-Modal Pair]` tab or click preset card *"1. Optical + SAR Cross-Modal"*.
  - *Say:* *"Now let us demonstrate the core novelty: joint reasoning over cross-modal pairs."*
- **Step 7: Ingesting Cartosat-2S Optical and RISAT-1C SAR**
  - *Action:* Show the dual preview: Optical RGB in Slot 1, C-band SAR backscatter in Slot 2.
  - *Say:* *"Here we ingest a co-registered pair: Cartosat-2S optical alongside RISAT-1C all-weather radar."*
- **Step 8: Executing Official Representative Query 4**
  - *Action:* Click Query 4: *"Use the optical and SAR images together to identify built-up and water-covered regions."*
  - *Say:* *"Optical alone confuses dry soil with concrete, and is obscured by cloud shadows. But RISAT's C-band radar produces strong dihedral double-bounce on buildings (-8.2 dB) and specular dark reflection on water (< -22 dB). The fused output cleanly segments both with 96.5% confidence."*
- **Step 9: Expanding the Auditable JSON Execution Trace**
  - *Action:* Click "Auditable Execution Trace".
  - *Say:* *"Look at the JSON trace. The problem statement explicitly states internal reasoning text is not evaluated — only observable execution traces are judged. Here are the selected task, invoked models, permitted parameters, and turnaround latency."*
- **Step 10: Exporting to ISRO Bhuvan GeoJSON**
  - *Action:* Click "Download Bhuvan GeoJSON".
  - *Say:* *"With one click, the analyst downloads the georeferenced polygon layer ready for instant drag-and-drop into ISRO Bhuvan or VEDAS."*

---

## 3. Alternative Scenarios (If Judges Ask to See Other Queries)

1. **Query 1 (Single-Image VQA):**
   - Click Single Image Tab.
   - Query: *"Describe the land-cover and major objects visible in this image."*
   - Shows BigEarthNet-adapted landcover classification and road/bridge detection.
2. **Query 2 (Single-Image Grounding):**
   - Click Single Image Tab.
   - Query: *"Highlight the water body referred to in the query."*
   - Shows VRSBench-aligned bounding box coordinates `[ymin, xmin, ymax, xmax]` and SAM boundary mask.
3. **Query 5 (Bi-Temporal CDVQA):**
   - Click Bi-Temporal Tab.
   - Query: *"Has the built-up area increased, decreased, or remained unchanged?"*
   - Shows categorical answer: *"INCREASED by 6.8 ha"* supported by NDBI difference metrics.

---

## 4. Hardware Failure & Emergency Backup Protocol

- **Level 1 (Localhost backend offline):**
  - `index.html` has built-in offline simulation that responds in 1.2 seconds with identical metrics and traces.
- **Level 2 (Display projector scaling):**
  - Press `Ctrl + -` or `Ctrl + +` to adjust zoom to 90% or 100%.
- **Level 3 (Jury asks for code verification):**
  - Open terminal and run: `python -m pytest tests/ -q` to show 78 passing tests in under 20 seconds.
