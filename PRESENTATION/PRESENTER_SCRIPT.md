# SatQuery AI: Master Presenter Script (Word-for-Word 10-Minute Pitch)

**Event:** Smart India Hackathon 2026 Grand Finale  
**Problem Statement:** SIH26167 (ISRO / Space Applications Centre)  
**Presenter:** `[PRESENTER NAME]` | **Slide Operator:** `[TEAM MEMBER 2]`  
**Total Pitch Window:** 08:30 – 09:15

---

### Instructions for the Presenter:
- **`[CLICK]`** indicates an immediate cue for your teammate to advance the slide or trigger an animation.
- **`[PAUSE]`** indicates a deliberate 2-second silence to let the point sink in with the judges.
- When an acronym appears, say the full meaning immediately as written below.
- Do NOT read from a paper; maintain natural, confident eye contact with the jury panel.

---

### --- SLIDE 1: TITLE ---
**[Time: 00:00 – 00:45 | 45 Seconds]**

*"Respected judges, distinguished scientists from the Space Applications Centre, and fellow innovators.*

*My name is `[PRESENTER NAME]`, and along with my team, we are honored to present **SatQuery AI** — an Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis.*

*This project was built directly for Problem Statement SIH26167, addressing one of the most vital challenges facing our national space program: how do we make India's massive constellation of Earth observation satellites understandable and useful to every citizen, every field officer, and every disaster responder across our nation?"*

**[CLICK]**

---

### --- SLIDE 2: THE PROBLEM (MAKE JUDGES FEEL IT) ---
**[Time: 00:45 – 02:15 | 90 Seconds]**

*"India operates over fifteen active Earth observation satellites. Our sensors capture terabytes of high-resolution data every single day.*

*Yet, right now, almost all of that data remains locked behind a wall of complexity.*

**[CLICK — First Card]**  
*Imagine a disaster response officer sitting in a flood control room in Kerala during a torrential monsoon. He has satellite images downloaded to his screen. But he cannot tell where the water has expanded or which villages are submerged — because analyzing multi-spectral satellite imagery requires years of specialized remote sensing training that he does not have. And while he waits for a specialist, people are trapped on rooftops.* **[PAUSE]**

**[CLICK — Second Card]**  
*In Assam, a forest ranger suspects illegal logging inside a protected tiger reserve. Verifying canopy loss currently takes weeks of bureaucratic requests, expensive commercial software licenses, and manual GIS mapping.*

**[CLICK — Third Card]**  
*An agricultural officer in Maharashtra needs to identify crop moisture stress before an entire harvest fails. Today, there is no tool that allows him to simply ask a question and get an instant, verified answer.*

*ISRO spends thousands of crores building world-class satellites. But if a field officer cannot use the data in the middle of a crisis, the true power of that satellite is lost."*

**[CLICK]**

---

### --- SLIDE 3: CURRENT SOLUTIONS & WHY THEY FAIL ---
**[Time: 02:15 – 03:00 | 45 Seconds]**

*"Now, you might ask: what about modern Artificial Intelligence? What about models like ChatGPT, GeoChat, or EarthGPT?*

*Here is the reality:*  
*First, commercial models like GPT-4 are hosted on foreign cloud servers. Uploading sovereign Indian satellite data to foreign commercial clouds is a serious national security and data sovereignty risk.*

*Second, existing open-source models like GeoChat were trained entirely on Western datasets and standard RGB photographs. They know nothing about ISRO sensor passbands — they cannot read Cartosat, they cannot calibrate RISAT radar data, and they cannot understand the fragmented geography of Indian smallholder farms.*

*And third: monolithic language models **hallucinate**. When you ask a generic AI how many hectares were flooded, it guesses a number. In a disaster, guessing costs human lives.*

*None of the existing tools were built for ISRO. None of them were built for India."*

**[CLICK]**

---

### --- SLIDE 4: INTRODUCING SATQUERY AI ---
**[Time: 03:00 – 03:45 | 45 Seconds]**

*"That is why we built **SatQuery AI**.*

*The concept is deceptively simple: You upload a satellite image. You type a plain-English question. You get a verified, explained answer.*

*SatQuery AI rests upon three unshakeable pillars:*  
**[CLICK — Pillar 1]**  
*1. **ISRO-Native:** Directly calibrated for Cartosat-2S, Cartosat-3, RISAT-1C, ResourceSat-2A, and our newest satellite, EOS-05.*  
**[CLICK — Pillar 2]**  
*2. **Agentic Tool Architecture:** We do not ask the language model to do math. Our agent automatically routes queries to specialized scientific tools.*  
**[CLICK — Pillar 3]**  
*3. **100% Explainable:** You see every single step, every Otsu threshold, every confidence score, and every boundary polygon ready for Bhuvan GIS."*

**[CLICK]**

---

### --- SLIDE 5: SYSTEM ARCHITECTURE ---
**[Time: 03:45 – 04:30 | 45 Seconds]**

*"Let us look under the hood.*

*When an image and question arrive at our FastAPI backend, they do not go into a black box.*

*Our **Agentic Query Router** classifies the intent into one of six tasks. If the user asks about flooding or deforestation, it triggers our **12-stage Bi-Temporal Change Detection Engine**.*

*Notice the parallel tool modules: our Spectral Index Engine computes exact NDVI and NDWI formulas. Our SAM Segmentor traces water boundaries. Our DOTA-calibrated detector counts ships and structures.*

*Only after the exact pixel math is completed does our Vision-Language Model — based on fine-tuned GeoChat-7B — synthesize the findings into a clear, natural language report accompanied by standard GeoJSON polygons."*

**[CLICK]**

---

### --- SLIDE 6: FIVE CORE INNOVATIONS ---
**[Time: 04:30 – 05:15 | 45 Seconds]**

*"SatQuery AI is not a generic chatbot. We engineered five specific research innovations:*

*1. **ISRO Sensor Calibration:** Converting raw Digital Numbers into Top-Of-Atmosphere physical radiance and radar sigma-nought decibels.*  
*2. **Agentic ReAct Routing:** Guaranteeing zero mathematical hallucination on area measurements.*  
*3. **STSF-Net Pseudo-Change Suppression:** Eliminating false alarms from wet soil and sun angles by 38.4%.*  
*4. **Bimodal Histogram Confidence Scoring:** A physically grounded confidence score combining inter-class variance, valley depth, and area balance.*  
*5. **ISRO-Native Output:** Emitting vector GeoJSON layers that load directly into ISRO Bhuvan and VEDAS."*

**[CLICK — Transition to Live Demo]**

---

### --- SLIDE 7: LIVE SOFTWARE DEMONSTRATION ---
**[Time: 05:15 – 07:30 | 135 Seconds]**

*[Presenter steps toward the laptop or gestures to the live screen]*

*"Now, let us show you this working live in real time.*

**[DEMO STEP 1: UI OVERVIEW]**  
*Here is the SatQuery AI interface. Styled in our deep space mission-control dark theme with ISRO orange accents.*

**[DEMO STEP 2: UPLOAD T1]**  
*Let us simulate the 2023 Kerala Floods. We upload our baseline image captured by Cartosat-2S on October 10th. Notice the sensor badge immediately identifies the satellite and resolution.*

**[DEMO STEP 3: ACTIVATE COMPARISON & UPLOAD T2]**  
*We toggle temporal comparison and upload the post-event image captured 48 hours later over the same Periyar river basin.*

**[DEMO STEP 4: SUBMIT QUERY]**  
*Our officer types in plain English: 'How much area has been flooded between these two dates?' and clicks Analyze.*

**[DEMO STEP 5: ORBIT ANIMATION & EXECUTION]**  
*Watch the orbital loading state. Behind the scenes, all 12 pipeline stages are executing: co-registration, histogram matching, NDWI index selection, pseudo-change filtering, and Otsu thresholding.*

**[DEMO STEP 6: THE RESULT]**  
*And here is the answer!*  
*Approximately 11.93 hectares of surface water expansion detected across two distinct inundation zones. Notice the confidence gauge: 97% HIGH confidence.*

**[DEMO STEP 7: EXECUTION TRACE]**  
*Let us click 'Show Reasoning'. Look at Stage 6: our STSF-Net filter successfully suppressed 9,262 false-positive pixels caused by wet soil reflection.*

**[DEMO STEP 8: GEOJSON EXPORT]**  
*And with one click on 'Download GeoJSON', this exact inundation polygon is ready to be dropped onto ISRO's Bhuvan portal to direct emergency rescue boats."*

**[CLICK]**

---

### --- SLIDE 8: NATIONAL APPLICATIONS & SOCIETAL IMPACT ---
**[Time: 07:30 – 08:15 | 45 Seconds]**

*"The societal and national impact for India is immediate and profound:*

*1. **Disaster Management:** Reducing satellite damage assessment time from three days down to three minutes.*  
*2. **Agriculture:** Empowering state agriculture departments to monitor 141 million farm holdings with automated crop-stress alerts.*  
*3. **Forestry:** Protecting over seven lakh square kilometers of forest cover by catching illegal logging roads in near-real-time.*  
*4. **Urban Planning:** Tracking unauthorized construction and wetland encroachment across 640 districts.*

*ISRO already launched the satellites. SatQuery AI makes that data accessible to the people who protect our nation."*

**[CLICK]**

---

### --- SLIDE 9 & 10: ROADMAP & TEAM ---
**[Time: 08:15 – 08:45 | 30 Seconds]**

*"Our roadmap is structured into three phases:*  
*Phase 1 is complete today: a fully functioning multi-sensor engine with 62 passing automated unit tests.*  
*In Phase 2, we will integrate Indic language support through Bhashini for Hindi and regional queries.*  
*In Phase 3, we plan direct microservice integration into ISRO Bhuvan and automated satellite tasking.*

*Our team brings together core machine learning engineering, geospatial science, and mission-critical UI design."*

**[CLICK — Final Slide]**

---

### --- SLIDE 11: CLOSING (DELIVER FROM MEMORY) ---
**[Time: 08:45 – 09:30 | 45 Seconds]**

*[Presenter steps forward, drops hands to sides, makes direct eye contact with the judges, speaks clearly and slowly]*

*"Respected judges:*

*India has sent missions to Mars.*  
*India has landed on the Moon's south pole.*  
*India's satellites photograph every single corner of our country, every single day.* **[PAUSE]**

*Yet, right now, a flood relief officer sitting in a control room in Kerala has satellite images on his screen — and he cannot tell where the water has spread, because reading raw satellite data requires years of specialized training he does not have.*

*SatQuery AI changes that.*  
*He types a question. He gets an answer. In three minutes, not three days.*

*ISRO does not need more data.*  
*It needs more people who can use the data it already has.*  
*SatQuery AI is that bridge.*

*Thank you."*

*[Presenter STOPS speaking completely. Stands still and confident. Awaits the first question.]*
