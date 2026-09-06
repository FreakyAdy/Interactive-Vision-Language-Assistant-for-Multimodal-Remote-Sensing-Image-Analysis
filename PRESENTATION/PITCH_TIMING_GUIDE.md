# SatQuery AI: Pitch Timing & Stage Flow Guide (SIH26167)

**Hackathon:** Smart India Hackathon 2026 Grand Finale  
**Problem Statement:** SIH26167 — SatQuery AI: An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries  
**Target Organization:** Space Applications Centre (SAC), ISRO  
**Total Allocated Pitch Duration:** 10 Minutes (Strict SIH Protocol)  
**Target Delivery Window:** 08:30 – 09:15 (Leaving 45–90 seconds safety buffer before Jury Q&A)  

---

## 1. Minute-by-Minute Breakdown

```
[00:00 - 00:30]  Stage Entrance, Device Connection & Audio Check
[00:30 - 01:15]  Slide 1: Title, ISRO Problem Context & BigEarthNet.txt Badge
[01:15 - 02:15]  Slide 2: Operational Reality — Why Single Optical Imagery Fails
[02:15 - 03:00]  Slide 3: Why Generic VLMs Fail Remote Sensing
[03:00 - 03:45]  Slide 4: Defined Input Scopes (Single, Cross-Modal, Bi-Temporal)
[03:45 - 04:30]  Slide 5: Agentic Architecture & Specialist Tool Registry
[04:30 - 05:00]  Slide 6: The 5 Official Representative Queries
[05:00 - 07:15]  LIVE DEMO: 
                 - Bi-Temporal Change Detection (Kerala Flood Q3)
                 - Cross-Modal Optical+SAR Fusion (Cartosat+RISAT Q4)
                 - Auditable JSON Execution Trace (SIH Mandate)
[07:15 - 07:45]  Slide 9: Evaluation Benchmarks (BigEarthNet.txt, VRSBench, CDVQA, ISRO/SAC)
[07:45 - 08:15]  Slide 10: National Impact (NDRF, PM Fasal Bima, Forest, Urban)
[08:15 - 08:45]  Slide 11: Roadmap & Verification (78/78 Tests Green)
[08:45 - 09:15]  Slide 12: Memorized Closing Speech & Transition to Q&A
[09:15 - 10:00]  Buffer & Immediate Ready Posture for Jury Questions
```

---

## 2. Detailed Timing Checkpoints & Pacing Rules

### Checkpoint 1 (00:30 – 02:15): Operational Reality & The Gap
- **Target Time:** Exactly 2 minutes elapsed.
- **Presenter State:** Calm, grounded, conversational.
- **Critical Point:** Highlight why a single optical image is inadequate in operational reality: cloud cover during monsoons, structural brightness confusion between soil and concrete, and the impossibility of answering change questions without paired observations.

### Checkpoint 2 (02:15 – 05:00): SIH26167 Compliance & Architecture
- **Target Time:** 4 minutes 30 seconds elapsed.
- **Presenter State:** Confident, authoritative, technically sharp.
- **Key Emphasis:**
  - *Domain Adaptation:* Point out `BigEarthNet.txt` (arXiv:2603.29630) multimodal contrastive pretraining.
  - *Defined Input Scope:* Walk through Single Image, Cross-Modal Pair, and Bi-Temporal Pair in GeoTIFF.
  - *The 5 Official Queries:* Highlight that Q1 through Q5 are natively supported.
  - *Auditable Trace:* Reiterate that internal reasoning text is not evaluated; only observable execution traces are judged.

### Checkpoint 3 (05:00 – 07:15): THE LIVE DEMO (The Winning Moment)
- **Target Time:** 2 minutes 15 seconds dedicated to software interaction.
- **Presenter State:** Crisp, deliberate mouse actions.
- **Protocol:**
  1. Show 3-way Scope tabs on top.
  2. Load Bi-temporal scenario (Kerala flood) and execute Query 3. Show 11.93 ha inundation with 97% confidence. Point out STSF-Net suppressed 9,200 false-alarm pixels.
  3. Switch to Cross-Modal tab. Ingest Cartosat-2S optical + RISAT-1C SAR. Run Query 4. Show SAR dihedral double-bounce and specular reflection disambiguating built-up (14.8 ha) and water (21.4 ha).
  4. Click "Auditable Execution Trace" to display the structured JSON payload.
  5. Click "Download Bhuvan GeoJSON" to show immediate interoperability.

### Checkpoint 4 (07:15 – 09:15): Benchmark Rigor & The Closing Speech
- **Target Time:** Pitch ends at exactly 09:00 – 09:15.
- **Benchmarking:** Emphasize 78 passing unit tests, BigEarthNet.txt top-1 retrieval (91.4%), VRSBench mIoU (68.2%), and CDVQA accuracy (94.2%).
- **The Closing Speech (Memorized):** Stand centered, make direct eye contact with the jury:
  *"India's satellites orbit overhead right now, seeing every river, every field, and every village. SatQuery AI gives our satellites a voice. Thank you."*
- **Immediate Posture:** Once you say *"Thank you,"* STOP speaking immediately. Maintain quiet, confident silence while waiting for the judges' first question.

---

## 3. Presentation Role Cards

- **Primary Presenter (`[PRESENTER NAME]`):** Delivers verbal pitch, controls narrative pacing, handles high-level impact and domain adaptation questions during Q&A.
- **Demo Operator (`[TEAM MEMBER 2]`):** Sits beside laptop, manages HDMI connection, executes mouse clicks on cue, switches tabs smoothly.
- **Core AI Responders (`[TEAM MEMBERS 3–6]`):** Ready to stand up and answer deep technical questions on BigEarthNet.txt InfoNCE loss, SAR polarimetric backscatter calibration, STSF-Net equations, and CDVQA attention heads.
