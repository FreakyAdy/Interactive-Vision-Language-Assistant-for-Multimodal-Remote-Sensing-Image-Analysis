# SatQuery AI: Pitch Timing & Stage Flow Guide

**Hackathon:** Smart India Hackathon 2026 (SIH26167)  
**Total Allocated Pitch Duration:** 10 Minutes (Strict SIH Protocol)  
**Target Delivery Window:** 08:30 – 09:15 (Leaving 45–90 seconds safety buffer before Q&A)

---

## 1. Minute-by-Minute Breakdown

```
[00:00 - 00:30]  Stage Entrance, Device Connection & Audio Check
[00:30 - 01:00]  Slide 1: Title & ISRO Mission Context
[01:00 - 02:30]  Slide 2: The Core Problem (Three Human Stories)
[02:30 - 03:15]  Slide 3: Current Solutions & Why They Fail
[03:15 - 04:00]  Slide 4: Introducing SatQuery AI (The 3 Pillars)
[04:00 - 04:45]  Slide 5: System Architecture & ReAct Routing
[04:45 - 05:15]  Slide 6: Five Core Innovations (Technical Depth)
[05:15 - 07:30]  LIVE DEMO: Kerala Flood Assessment & Execution Trace
[07:30 - 08:00]  Slide 8: National Impact & Real-World Use Cases
[08:00 - 08:30]  Slide 9: Roadmap (Phases 1, 2, 3)
[08:30 - 09:00]  Slide 10 & 11: Team & Memorized Closing Speech
[09:00 - 10:00]  Buffer & Transition into Judge Q&A
```

---

## 2. Detailed Timing Checkpoints & Pacing Rules

### Checkpoint 1 (00:30 – 02:30): The Hook & The Pain Points
- **Target Time:** Exactly 2 minutes elapsed.
- **Presenter State:** Calm, grounded, conversational.
- **Critical Action:** Pause for 2 seconds after describing the Kerala flood officer:  
  *"He has satellite images on his screen right now... and people are stranded on rooftops... and he cannot read the data because it requires a remote sensing PhD."*
- **Pacing Warning:** Do not rush the pain points. If judges do not feel the problem, they will dismiss the technical solution as just another chatbot.

### Checkpoint 2 (02:30 – 05:15): The Pivot to SatQuery AI & Architecture
- **Target Time:** 4 minutes 45 seconds elapsed.
- **Presenter State:** Confident, energetic, authoritative.
- **Key Emphasis:** Emphasize the word **ISRO-Native**. *"This was not built in Silicon Valley on Sentinel data. This was built for Cartosat, RISAT, and ResourceSat."*
- **Architecture Flow:** Spend 30 seconds on the ReAct router: *"We do not ask the neural network to calculate square meters. We route to a deterministic 12-stage spatial engine."*

### Checkpoint 3 (05:15 – 07:30): THE LIVE DEMO (The Winning Moment)
- **Target Time:** 2 minutes 15 seconds dedicated to software interaction.
- **Presenter State:** Crisp, deliberate mouse actions.
- **Protocol:**
  1. Show drag-drop of T1 and T2 images.
  2. Highlight automated detection: *"Notice the sensor badge: ISRO Cartosat-2S."*
  3. Type query: *"How much area has been flooded between these two dates?"*
  4. Press **Analyze** (ISRO orange button). Point out loading orbital animation: *"12 pipeline stages executing."*
  5. Show result card: *"11.93 hectares inundation, 2 flood zones, 97% confidence."*
  6. Expand **Reasoning Trace**: *"Show judges the STSF-Net pseudo-change suppression step."*
  7. Click **Download GeoJSON**: *"This payload loads directly into ISRO Bhuvan."*

### Checkpoint 4 (07:30 – 09:00): National Impact & The Closing Speech
- **Target Time:** Pitch ends at exactly 08:50 – 09:00.
- **The Closing Speech (Memorized):** Stand centered, make direct eye contact with the lead ISRO judge.
- **Delivery:** Deliver the 4-sentence closing in Appendix of megaprompt without looking at the screen.
- **Immediate Posture:** Once you say *"Thank you,"* STOP speaking immediately. Maintain quiet, confident silence while waiting for the judges' first question.

---

## 3. Presentation Role Cards

- **Primary Presenter (`[PRESENTER NAME]`):** Delivers verbal pitch, controls narrative pacing, handles high-level impact and innovation questions during Q&A.
- **Demo Operator (`[TEAM MEMBER 2]`):** Sits beside laptop, manages HDMI connection, executes mouse clicks on cue, triggers backup demo if network drops.
- **Core AI Responders (`[TEAM MEMBERS 3–6]`):** Ready to stand up and answer deep technical questions on STSF-Net equations, Otsu threshold derivation, and GeoChat quantization.
