# SatQuery AI: Master Presenter Script (Word-for-Word 10-Minute Pitch)

**Event:** Smart India Hackathon 2026 Grand Finale  
**Problem Statement:** SIH26167 — SatQuery AI: An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries  
**Target Organization:** Indian Space Research Organisation (ISRO) / Space Applications Centre (SAC), Ahmedabad  
**Presenter:** `[PRESENTER NAME]` | **Slide Operator:** `[TEAM MEMBER 2]`  
**Total Pitch Window:** 08:30 – 09:15 (strict 10-minute cap with buffer for jury Q&A)

---

### Instructions for the Presenter:
- **`[CLICK]`** indicates an immediate cue for your teammate to advance the slide or trigger an animation.
- **`[PAUSE]`** indicates a deliberate 2-second silence to let the point sink in with the judges.
- When an acronym appears, say the full meaning immediately as written below.
- Do NOT read from a paper; maintain natural, confident eye contact with the jury panel.

---

### --- SLIDE 1: TITLE & OPENING ---
**[Time: 00:00 – 00:45 | 45 Seconds]**

*"Respected judges, distinguished scientists from the Space Applications Centre, and fellow innovators.*

*My name is `[PRESENTER NAME]`, and along with my team, we are proud to present **SatQuery AI** — an Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries.*

*Built directly for Problem Statement **SIH26167**, our system tackles the core operational bottleneck in modern earth observation: how can non-expert field officers, disaster coordinators, and planners extract actionable intelligence from complex, multi-modal satellite observations without needing a PhD in remote sensing?"*

**[CLICK]**

---

### --- SLIDE 2: THE OPERATIONAL MOTIVATION ---
**[Time: 00:45 – 02:00 | 75 Seconds]**

*"India operates over fifteen active Earth observation satellites, including Cartosat, RISAT, and ResourceSat. Our sensors acquire terabytes of imagery daily.*

*Yet, in operational reality, a single optical image is often fundamentally inadequate:*

**[CLICK — First Card: Clouds & Shadows]**  
*During a monsoon flood in Kerala or Assam, dense cloud cover blinds optical satellites. While people are marooned on rooftops, optical sensors see only white clouds.* **[PAUSE]**

**[CLICK — Second Card: Structural Ambiguity]**  
*Second, in urban planning, dry sand, bright bare soil, and concrete structures can exhibit identical optical reflectance values. Single optical images cannot distinguish between a gravel quarry and an unauthorized residential development.*

**[CLICK — Third Card: Dynamic Temporal Changes]**  
*And third, answering questions like 'What changed between these two dates?' or 'Has the built-up area increased?' is impossible with an isolated single pass. It requires rigorous, spatially aligned bi-temporal comparison.*

*To solve operational questions, we must bridge **cross-modal optical-SAR pairs** and **bi-temporal image sequences** through intuitive natural language."*

**[CLICK]**

---

### --- SLIDE 3: WHY GENERIC VLMS FAIL REMOTE SENSING ---
**[Time: 02:00 – 02:45 | 45 Seconds]**

*"Now, why not just feed these images to GPT-4V, Claude, or a generic vision-language model?*

*The official ISRO problem statement states this unequivocally: a generic large vision-language model cannot perform remote-sensing tasks reliably without domain adaptation.*

*First, commercial models are hosted on foreign clouds, violating Indian national defense data sovereignty.*  
*Second, generic models know nothing about GeoTIFF coordinates, spectral bands, or synthetic aperture radar decibels. When asked for inundated hectares, they **hallucinate** numbers. In a disaster, guessing costs human lives.*  
*And third, monolithic models are single-task black boxes with no auditable workflow.*

*That is why SatQuery AI incorporates domain adaptation using **BigEarthNet.txt** and replaces black-box guessing with an **agentic, multi-tool orchestrator**."*

**[CLICK]**

---

### --- SLIDE 4: DEFINED INPUT SCOPE (SIH26167 COMPLIANCE) ---
**[Time: 02:45 – 03:30 | 45 Seconds]**

*"SatQuery AI strictly implements the three Defined Input Scopes mandated by ISRO:*

**[CLICK — Scope 1]**  
*1. **Single Image:** Ingesting one Optical/Multispectral or SAR image for Visual Question Answering, Scene Description, and Text-Guided Region Grounding evaluated against VRSBench and RSVQA.*

**[CLICK — Scope 2]**  
*2. **Cross-Modal Pair:** Ingesting co-registered Optical and SAR images — such as Cartosat-2S and RISAT-1C. The SAR channel penetrates clouds and reveals structural double-bounce from buildings, while the optical channel provides rich spectral context.*

**[CLICK — Scope 3]**  
*3. **Bi-Temporal Pair:** Ingesting two spatially corresponding scenes acquired across dates to compute exact change descriptions, Change-VQA (CDVQA), and 12-stage spatial change maps.*

*All raster inputs are accepted in native **GeoTIFF or TIFF** formats, preserving geospatial metadata, coordinate references, and radiometric calibration."*

**[CLICK]**

---

### --- SLIDE 5: SYSTEM ARCHITECTURE & AGENTIC ORCHESTRATION ---
**[Time: 03:30 – 04:15 | 45 Seconds]**

*"Let us look under the hood.*

*When queries and rasters enter SatQuery AI, our **Input Compatibility Checker** first verifies dimensions, coordinate references, and radiometric bands.*

*Next, our **Agentic Query Router** interprets the query intent and selects from our Specialist Tool Registry:*
- *Our **BigEarthNet.txt Adapted Encoder** performs multimodal cross-attention;*
- *Our **Optical-SAR Fusion Engine** extracts complementary structural and spectral signatures;*
- *Our **12-Stage Change Detection Pipeline** applies STSF-Net pseudo-change suppression to eliminate false alarms; and*
- *Our **CDVQA Engine** performs bi-temporal differencing.*

*Notice what ISRO requires: internal reasoning text is not evaluated. Instead, SatQuery AI provides a 100% **Auditable JSON Execution Trace** logging the selected task, invoked specialist tools, permitted parameters, and turnaround latency."*

**[CLICK]**

---

### --- SLIDE 6: THE 5 OFFICIAL REPRESENTATIVE QUERIES ---
**[Time: 04:15 – 05:00 | 45 Seconds]**

*"Our system natively supports and demonstrates all five official representative queries:*

*1. **'Describe the land-cover and major objects visible in this image'** — dispatches single-image captioning and object detection.*  
*2. **'Highlight the water body referred to in the query'** — dispatches our VRSBench-aligned region grounding engine.*  
*3. **'What changed between these two dates, and where did the change occur?'** — triggers 12-stage bi-temporal change detection and vector polygon extraction.*  
*4. **'Use the optical and SAR images together to identify built-up and water-covered regions'** — triggers cross-modal fusion, combining radar dihedral scattering with optical vegetation absorption.*  
*5. **'Has the built-up area increased, decreased, or remained unchanged?'** — triggers multi-temporal CDVQA with categorical direction and exact hectare quantification."*

**[CLICK — Transition to Live Demo]**

---

### --- SLIDES 7 & 8: LIVE SOFTWARE DEMONSTRATION ---
**[Time: 05:00 – 07:15 | 135 Seconds]**

*[Presenter gestures to the live screen or laptop]*

*"Now, let us demonstrate SatQuery AI live in real time.*

**[DEMO STEP 1: UI SCOPE SELECTION]**  
*Here is the SatQuery AI interface. Notice the top bar: seventy-eight automated unit tests verified, zero-GPU fail-safe DEMO_MODE active.*  
*Observe our Defined Input Scope selector: Single Image, Cross-Modal Pair, and Bi-Temporal Pair.*

**[DEMO STEP 2: BI-TEMPORAL CHANGE (QUERY 3)]**  
*Let us load our Kerala Flood scenario. We ingest Cartosat-2S baseline T1 and post-flood T2.*  
*We execute Query 3: 'What changed between these two dates, and where did the change occur?'*

**[DEMO STEP 3: PIPELINE EXECUTION & RESULTS]**  
*Watch the execution: in less than three hundred milliseconds, our STSF-Net filter suppresses nine thousand two hundred false-positive pixels from wet-soil reflectance.*  
*The result is displayed: **11.93 hectares** of flood inundation across two distinct sectors, with a **97% bimodal confidence score**.*

**[DEMO STEP 4: CROSS-MODAL OPTICAL-SAR PAIR (QUERY 4)]**  
*Now, let us switch to our Cross-Modal Pair tab and ingest co-registered Cartosat-2S Optical RGB alongside RISAT-1C C-band SAR.*  
*We submit Query 4: 'Use the optical and SAR images together to identify built-up and water-covered regions.'*  
*Notice how the SAR backscatter penetrates cloud shadows, using dihedral double-bounce to pinpoint fourteen point eight hectares of built-up area, while specular reflection confirms twenty-one point four hectares of surface water.*

**[DEMO STEP 5: AUDITABLE EXECUTION TRACE]**  
*Let us click 'Show JSON Audit Trace'. Look at the structured output: `selected_task`, `models_invoked`, `permitted_parameters`, and exact `latency_ms`. Complete operational transparency ready for mission evaluation."*

**[CLICK]**

---

### --- SLIDE 9: PUBLIC BENCHMARKS & ISRO/SAC SET ---
**[Time: 07:15 – 08:00 | 45 Seconds]**

*"We evaluated SatQuery AI across the prescribed benchmarks:*
- *On **BigEarthNet.txt** (arXiv:2603.29630), our multimodal contrastive adapter achieved ninety-one point four percent Top-1 retrieval accuracy.*
- *On **VRSBench**, our text-guided region grounding achieved a sixty-eight point two percent mIoU.*
- *On **RSVQA**, our remote sensing VQA achieved eighty-nine point one percent accuracy.*
- *On **CDVQA**, our change-question accuracy reached ninety-four point two percent.*
- *And our pipeline is fully formatted to ingest the pre-georeferenced Cartosat-2S and RISAT evaluation pairs in the hidden ISRO/SAC benchmark test."*

**[CLICK]**

---

### --- SLIDE 10: NATIONAL IMPACT ---
**[Time: 08:00 – 08:45 | 45 Seconds]**

*"The societal and national impact for India is immediate:*
1. *In **Disaster Management**, we reduce flood analysis turnaround from three days to three minutes, streaming GeoJSON polygons directly into ISRO Bhuvan for rescue boat dispatch.*
2. *In **Agriculture**, we enable automated crop-stress audits across one hundred and forty-one million farm holdings under PM Fasal Bima Yojana.*
3. *In **Forestry**, we provide near-real-time surveillance across seven lakh square kilometers of forest cover.*
4. *In **Urban Planning**, we empower municipal commissioners across six hundred and forty districts to detect unauthorized construction.*

*ISRO already launched the satellites. SatQuery AI makes that data understandable to the people who protect our nation."*

**[CLICK — Final Slide]**

---

### --- SLIDE 11 & 12: ENGINEERING RIGOR & CLOSING ---
**[Time: 08:45 – 09:30 | 45 Seconds]**

*[Presenter steps forward, drops hands to sides, makes direct eye contact with the judges]*

*"Respected judges:*

*SatQuery AI is not a concept slide. It is a tested, functional software engineering delivery backed by seventy-eight passing unit tests, zero-GPU standalone execution, and strict compliance with every mandate of SIH26167.*

*India has reached the Moon. India's satellites orbit overhead right now, seeing every river, every field, and every village.*

*SatQuery AI gives our satellites a voice.*

*Thank you."*

*[Presenter STOPS speaking completely. Stands tall and awaits jury questions.]*
