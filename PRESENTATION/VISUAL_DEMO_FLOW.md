# SatQuery AI: Visual Demo Execution Flow & Backup Plan

**Smart India Hackathon 2026 | Problem Statement: SIH26167**  
**Audience:** ISRO / Space Applications Centre Evaluation Jury

---

## 1. Pre-Presentation Setup (15 Minutes Before Stage)

1. **Start the Backend (Failsafe DEMO_MODE enabled):**
   ```bash
   cd satquery-ai/PROJECT/backend
   python -m uvicorn main:app --port 8000
   ```
2. **Open the Frontend:**
   - Launch Google Chrome or Microsoft Edge.
   - Open `satquery-ai/PROJECT/frontend/index.html`.
   - Press `F11` to enter clean Fullscreen Mission-Control mode.
3. **Verify Demo Data Assets:**
   - Confirm synthetic images exist in `satquery-ai/PROJECT/demo_data/`:
     - `flood_t1.png` and `flood_t2.png`
     - `forest_t1.png` and `forest_t2.png`
     - `urban_t1.png` and `urban_t2.png`
     - `agri_sample.png` and `harbor_sample.png`
4. **Failsafe Check:**
   - Check that `DEMO_MODE = true` in `index.html`. If the laptop disconnects from localhost, the frontend will automatically simulate responses with satellite animations.

---

## 2. Live Demo Script (During Slide 7: 05:15 – 07:30)

```
┌────────────────────────────────────────────────────────────────────────┐
│                        LIVE DEMO 9-STEP SEQUENCE                       │
├───────┬────────────────────────────┬───────────────────────────────────┤
│ Step  │ Action                     │ Verbal Narration Cue              │
├───────┼────────────────────────────┼───────────────────────────────────┤
│ 1     │ Show App Header & Canvas   │ "This is SatQuery AI."            │
│ 2     │ Drag & Drop T1 (Oct 10)    │ "Detected: ISRO Cartosat-2S 2.1m" │
│ 3     │ Toggle 'Compare Images'    │ "Activating bi-temporal engine"   │
│ 4     │ Drag & Drop T2 (Oct 12)    │ "Co-registration verified"        │
│ 5     │ Click Example Query Chip   │ "How much area has been flooded?" │
│ 6     │ Click 'Analyze' (Orange)   │ "Watch: 12 stages executing"      │
│ 7     │ Reveal Annotated Result    │ "11.93 ha inundation, 97% conf."  │
│ 8     │ Expand 'Execution Trace'   │ "Point out STSF-Net suppression"  │
│ 9     │ Click 'Download GeoJSON'   │ "Direct GIS export for Bhuvan"    │
└───────┴────────────────────────────┴───────────────────────────────────┘
```

### Detailed Step-by-Step Instructions:

- **Step 1: Introduction to Mission Control UI**
  - *Action:* Move cursor across the dark navy interface.
  - *Say:* *"This is SatQuery AI. We designed this interface following ISRO mission-control standards: clean, high-contrast, zero clutter, built for fast decision-making."*
- **Step 2: Uploading T1 Image (Pre-Flood Kerala)**
  - *Action:* Drag `flood_t1.png` into the Primary Upload Zone.
  - *Say:* *"Notice how our sensor calibration module automatically extracts the metadata. It identifies this as an ISRO Cartosat-2S multispectral scene with 2.1-meter resolution."*
- **Step 3: Activating Temporal Comparison**
  - *Action:* Toggle the "Compare Two Images" switch. A second upload container slides in smoothly.
  - *Say:* *"For disaster tracking, single images aren't enough. We activate bi-temporal mode."*
- **Step 4: Uploading T2 Image (Post-Cyclone Inundation)**
  - *Action:* Drag `flood_t2.png` into the Secondary Upload Zone.
  - *Say:* *"Now we upload the post-event pass taken 48 hours later over the same Periyar river basin."*
- **Step 5: Entering the Natural Language Query**
  - *Action:* Click the suggestion chip: *"How much area has been flooded between these two dates?"*
  - *Say:* *"The officer doesn't need to know Python, GIS clipping tools, or band math. He asks in plain English."*
- **Step 6: Executing the Analysis**
  - *Action:* Click the prominent ISRO Orange **Analyze** button.
  - *Say:* *"As the satellite animation orbits, SatQuery's ReAct router selects the NDWI index, matches histograms, applies STSF-Net pseudo-change suppression, and calculates Otsu thresholds."*
- **Step 7: Interpreting the Result**
  - *Action:* Point to the right-hand panel split-view.
  - *Say:* *"Here is the answer: 11.93 hectares of new surface water expansion across 2 distinct inundation sectors. Notice the confidence gauge: 97% HIGH confidence, mathematically computed from the bimodal separation."*
- **Step 8: Revealing the 12-Stage Scientific Trace**
  - *Action:* Click "Expand 12-Stage Execution Trace".
  - *Say:* *"This is not a black box. The analyst can inspect all 12 stages. Look at Stage 6: STSF-Net filtered out 9,262 false-alarm pixels caused by moisture reflection on wet soil."*
- **Step 9: Exporting to Operational GIS**
  - *Action:* Click "Download GeoJSON".
  - *Say:* *"One click downloads the georeferenced polygon layer. The relief officer can immediately overlay this onto ISRO Bhuvan or send it to NDRF search-and-rescue teams on the ground."*

---

## 3. Alternative Scenarios (If Judges Ask to See Other Sensors)

1. **Deforestation in Assam Reserve (ResourceSat-2A):**
   - Click "Scenario 2" in the Demo Sidebar.
   - Shows canopy loss of 4.2 ha with NDVI drops along a logging road corridor.
2. **Delhi Peri-Urban Sprawl (Cartosat-3):**
   - Click "Scenario 3" in the Demo Sidebar.
   - Shows 2.8 ha of new concrete built-up growth using NDBI.
3. **Coastal Vessel Counting (RISAT-1C SAR):**
   - Click "Scenario 5" in the Demo Sidebar.
   - Shows all-weather C-band SAR vessel counting through cloud cover.

---

## 4. Hardware Failure & Emergency Backup Protocol

- **Level 1 (Localhost server port busy):**
  - Simply double-click `index.html` in Chrome. The client-side DEMO_MODE runs 100% offline without any server.
- **Level 2 (Projector display resolution issue):**
  - Press `Ctrl + Minus` or `Ctrl + Plus` to adjust UI scaling. The CSS uses flexible flex/grid units.
- **Level 3 (Catastrophic laptop freeze):**
  - Switch immediately to Slide 7 in `slides.html` which contains pre-rendered, high-resolution annotated screenshots and the complete execution trace table.
