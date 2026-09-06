# SatQuery AI: Slide Deck Content (Markdown Backup)

**Problem Statement:** SIH26167  
**Title:** SatQuery AI — Interactive Vision-Language Assistant for Multimodal Remote Sensing  
**Organisation:** ISRO / Space Applications Centre (SAC), Ahmedabad  
**Format:** 11 Slides with Speaker Notes & Fragment Sequences

---

## Slide 1: Title & Overview
- **Title:** SatQuery AI
- **Subtitle:** An Interactive Vision-Language Assistant for Multimodal Remote Sensing
- **Competition Reference:** Smart India Hackathon 2026 | Problem Statement: SIH26167
- **Organisation:** Space Applications Centre (SAC), ISRO, Ahmedabad
- **Badge:** ISRO-Native Multi-Modal Earth Observation Assistant

> **Speaker Notes:**  
> Welcome the jury panel. Introduce SatQuery AI as an interactive bridge connecting India's Earth observation constellation with non-expert field decision makers.

---

## Slide 2: The Problem (The Data Bottleneck)
- **Headline:** India operates 15+ active Earth observation satellites. But their data is locked away from the people who need it most.
- **Pain Point 1:** A disaster response officer in Kerala has satellite images of a raging flood on his screen. But he needs a remote sensing PhD to interpret multi-spectral bands. People are trapped on rooftops while analysis takes days.
- **Pain Point 2:** A forest ranger in Assam suspects illegal logging in a protected tiger corridor. Validating canopy loss requires weeks of bureaucratic requests and expensive proprietary software.
- **Pain Point 3:** An agricultural officer in Maharashtra needs to identify crop moisture stress before harvests fail. There is no simple, conversational interface.
- **Key Statistic:** ISRO collects terabytes of Earth observation data daily. Over 80% goes unanalyzed in operational timeframes.

> **Speaker Notes:**  
> Deliver the three human stories with emotional weight. Pause after the Kerala flood officer to let the urgency register with the judges.

---

## Slide 3: Current Solutions & Why They Fail
- **Comparison Table:**
  | Tool / Approach | Core Limitation |
  |---|---|
  | **Commercial VLMs (GPT-4V / Gemini)** | Foreign cloud hosting; violates national defense data sovereignty; zero spatial grounding. |
  | **GeoChat (CVPR 2024)** | Trained on Western RGB aerial photography; no ISRO sensor band calibration; hallucinated math. |
  | **EarthGPT (IEEE TGRS 2024)** | Monolithic black box; no agentic tool routing; uncalibrated softmax confidence scores. |
  | **ISRO VEDAS / Bhuvan** | Traditional expert GIS interface; requires deep technical knowledge; no natural language capability. |
- **Takeaway:** None of them were built for India. None of them were built for ISRO.

> **Speaker Notes:**  
> Contrast SatQuery against existing systems. Emphasize data sovereignty and the fatal flaw of mathematical hallucination in monolithic models.

---

## Slide 4: Introducing SatQuery AI
- **One-Liner:** Type a plain-English question. Upload a satellite image. Get an instant, verified answer.
- **The Three Pillars:**
  1. 🛰️ **ISRO-Native:** Calibrated for Cartosat-2S, Cartosat-3, RISAT-1C, ResourceSat-2A, and EOS-05.
  2. 🤖 **Agentic Multi-Tool Architecture:** Queries are routed to deterministic scientific engines (NDVI, NDWI, SAM, Otsu) instead of guessing.
  3. 📊 **100% Explainable:** Full 12-stage execution traces, physical confidence metrics, and Bhuvan-ready GeoJSON.

> **Speaker Notes:**  
> Introduce the solution clearly and boldly. Point out that SatQuery transforms complex raster data into human decisions.

---

## Slide 5: System Architecture & Agentic Flow
- **User Layer:** Single-page Mission Control Web Interface.
- **API Gateway:** FastAPI asynchronous server with OpenAPI documentation and CORS.
- **Agentic ReAct Query Router:** Semantic embedding classification (`all-MiniLM-L6-v2`) into 6 specialized tasks.
- **Parallel Tool Services:**
  - 12-Stage Bi-Temporal Change Detector
  - Multi-Spectral Index Engine (NDVI, NDWI, NDBI, EVI, RVI)
  - SAM-based Land-cover Segmentor
  - DOTA-calibrated Object & Vessel Counter
- **VLM Inference Engine:** GeoChat-7B with 4-bit quantization and sensor-conditioned prompting.
- **Output Synthesizer:** Structured Markdown reports + Vector GeoJSON.

> **Speaker Notes:**  
> Walk judges through the system architecture diagram. Emphasize that numbers come from deterministic math, not neural hallucination.

---

## Slide 6: Our Five Core Innovations
- **Innovation 1: ISRO Sensor Calibration:** Direct radiometric conversion from DN to TOA radiance ($L_\lambda$) and SAR backscatter ($\sigma^\circ$).
- **Innovation 2: Agentic ReAct Tool Router:** Intent classification with dynamic execution graph generation.
- **Innovation 3: STSF-Net Pseudo-Change Suppression:** Local patch spatial variance filtering eliminating 38.4% of false alarms.
- **Innovation 4: Bimodal Histogram Confidence Scoring:** Tri-factor physical confidence metric combining $\omega$, $v$, and $p$.
- **Innovation 5: ISRO-Native Operational Output:** Direct vector polygon export for Bhuvan and disaster reports matching SAC templates.

> **Speaker Notes:**  
> Present this slide with technical confidence. This demonstrates to the jury that SatQuery is deep engineering, not an API wrapper.

---

## Slide 7: Live Software Demonstration (Kerala Floods 2023)
- **Slide 7a (Setup):**
  - T1: Cartosat-2S Multispectral (October 10, 2023)
  - T2: Cartosat-2S Post-Cyclone Inundation (October 12, 2023)
  - Query: *"How much area has been flooded between these two dates?"*
- **Slide 7b (Results):**
  - Inundation Detected: **11.93 Hectares**
  - Distinct Flood Sectors: **2 Regions**
  - Confidence: **97% HIGH** (Bimodal separation score)
- **Slide 7c (Scientific Trace):**
  - Step 6 highlighted: STSF-Net removed 9,262 false-alarm pixels from wet soil reflectance.
  - GeoJSON polygon preview with EPSG:4326 coordinate reference system.

> **Speaker Notes:**  
> Transition seamlessly into the live app. If running offline, walk through these pre-rendered high-resolution slides.

---

## Slide 8: Applications & National Impact
- **Disaster Response:** Accelerating flood inundation mapping from 72 hours to under 3 minutes.
- **Agriculture:** Supporting 141 million agricultural parcels with NDVI moisture stress monitoring.
- **Forestry:** Protecting 7,12,249 sq km of forest cover through real-time deforestation alerting.
- **Urban Planning:** Auditing peri-urban built-up growth and encroachment across 640 districts.
- **Core Vision:** ISRO already launched the satellites. SatQuery AI makes that data usable by every Indian.

> **Speaker Notes:**  
> Connect the technology directly to national governance priorities, disaster relief, and PM Fasal Bima Yojana.

---

## Slide 9: Project Roadmap
- **Phase 1 (Completed — Hackathon Prototype):**
  - Working FastAPI backend, 12-stage change detection, 5 sensor calibrations, 62 automated unit tests, mission-control frontend.
- **Phase 2 (Next 3 Months):**
  - Fine-tuning on ISRO's proprietary Level-1 archive; integration of Bhashini for Hindi, Tamil, and Bengali queries.
- **Phase 3 (Next 6 Months):**
  - Direct microservice integration into ISRO Bhuvan and VEDAS geoportals; automated satellite tasking triggers.

> **Speaker Notes:**  
> Show that the team has thought beyond the 36-hour hackathon toward real deployment within ISRO.

---

## Slide 10: Team Roles & Responsibilities
- **Presenter (`[PRESENTER NAME]`):** Project Lead & Presentation
- **Core AI Engineer (`[TEAM MEMBER 2]`):** Change Detection & STSF-Net Filter
- **Core AI Engineer (`[TEAM MEMBER 3]`):** ReAct Router & VLM Inference
- **Geospatial Specialist (`[TEAM MEMBER 4]`):** Sensor Calibration & Spectral Math
- **Frontend Engineer (`[TEAM MEMBER 5]`):** Mission Control UI & GeoJSON WebGIS
- **QA & Verification Engineer (`[TEAM MEMBER 6]`):** Test Automation & Benchmarks

> **Speaker Notes:**  
> Introduce team roles briefly and acknowledge the collaborative execution.

---

## Slide 11: Conclusion & Call to Action
- **Large Header:** SatQuery AI
- **Tagline:** Making India's Space Data Accessible to Every Indian.
- **Core Quote:** *"India's satellites see everything. SatQuery AI understands it."*
- **Deliver Memorized Speech:** Appendix speech on India's space achievements and closing silence.

> **Speaker Notes:**  
> Deliver the closing speech without notes. Maintain steady eye contact. Stop speaking immediately upon conclusion.
