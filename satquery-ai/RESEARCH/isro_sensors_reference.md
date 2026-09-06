# ISRO Earth Observation Satellites: Sensor Specifications & Calibration Reference

**Project:** SatQuery AI (SIH26167)  
**Target Organisation:** Indian Space Research Organisation (ISRO) / Space Applications Centre (SAC)

---

## 1. Overview of the ISRO Earth Observation Constellation

The Indian Space Research Organisation operates one of the largest and most versatile constellations of civilian remote sensing satellites globally. These satellites provide continuous multi-modal data streams ranging from sub-meter panchromatic imaging to multi-polarimetric SAR and geosynchronous hyperspectral observation.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               ISRO SATELLITE SENSOR SPECIFICATIONS MATRIX                        │
├──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬───────────────────────┤
│ Satellite    │ Payload      │ Spatial Res. │ Spectral / Pol│ Swath (km)   │ Revisit & Orbit       │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼───────────────────────┤
│ Cartosat-2S  │ PAN + MS     │ 0.65m / 2.1m │ 4 Bands (VNIR)│ 9.6 km       │ 4 days (505 km SSO)   │
│ Cartosat-3   │ High-Res PAN │ 0.25m PAN    │ 4 Bands (1.1m)│ 17.0 km      │ 4 days (505 km SSO)   │
│ RISAT-1C     │ C-Band SAR   │ 3.0m - 25m   │ VV, VH, HH, HV│ 25 - 223 km  │ 14 days (536 km SSO)  │
│ ResourceSat-2│ LISS-IV / III│ 5.8m / 23.5m │ 4 Bands + SWIR│ 23 - 141 km  │ 5 - 24 days (817 km)  │
│ ResourceSat-2│ AWiFS        │ 56.0m        │ 4 Bands + SWIR│ 740 km       │ 5 days (synoptic)     │
│ EOS-04 (1A)  │ C/L-Band SAR │ 1.0m (HRS)   │ Dual-Pol      │ 10 - 50 km   │ All-weather radar     │
│ EOS-05 (2026)│ Hyperspectral│ 42m - 191m   │ 158 VNIR / 256│ Sub-continental│ Geosynchronous (GEO)│
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴───────────────────────┘
```

---

## 2. Detailed Sensor Characterization

### 2.1 Cartosat-2S (Cartosat-2 Series)
- **Launch Date:** June 23, 2017 (PSLV-C38)
- **Orbit:** 505 km Sun-Synchronous Polar Orbit (SSO), Inclination 97.44°
- **Payloads:**
  - **Panchromatic (PAN):** Ground Sampling Distance (GSD) of **0.65 m**, spectral range 500–850 nm.
  - **Multispectral (MS):** GSD of **2.10 m**, 4 bands:
    - Band 1 (Blue): 450–520 nm
    - Band 2 (Green): 520–590 nm
    - Band 3 (Red): 620–690 nm
    - Band 4 (Near Infrared - NIR): 770–860 nm
- **Swath:** 9.6 km at nadir (steerable up to ±45° along-track and across-track).
- **Radiometric Quantization:** 10-bit or 11-bit.
- **Strategic Applications:** Cadastral cartography, urban infrastructure modeling, coastal land-use regulation, defense utility mapping.

### 2.2 Cartosat-3
- **Launch Date:** November 27, 2019 (PSLV-C47)
- **Orbit:** 505 km Sun-Synchronous Orbit
- **Payloads:**
  - **Panchromatic:** GSD of **0.25 m** (world's highest spatial resolution from a civilian Indian remote sensing satellite at launch).
  - **Multispectral:** GSD of **1.13 m** in 4 spectral bands.
- **Swath:** 17.0 km at nadir.
- **Key Enhancements:** 11-bit radiometric resolution, agile steering with reaction wheels enabling stereo imaging in a single pass.

### 2.3 RISAT-1C (Radar Imaging Satellite)
- **Sensor Type:** Active Synthetic Aperture Radar (SAR)
- **Frequency:** C-band (5.35 GHz, wavelength ~5.6 cm)
- **Operating Modes:**
  - **Fine Resolution Stripmap-1 (FRS-1):** 3 m spatial resolution, single/dual-polarization, 25 km swath.
  - **Medium Resolution ScanSAR (MRS):** 25 m spatial resolution, dual-polarization (VV + VH), 115 km swath.
  - **Coarse Resolution ScanSAR (CRS):** 50 m spatial resolution, 223 km swath.
- **Incidence Angles:** Steerable between 20° and 55°.
- **Key Advantage:** Cloud-penetration and all-weather day/night imaging, vital during Indian monsoonal floods (e.g., Brahmaputra and Periyar inundations).

### 2.4 ResourceSat-2A
- **Launch Date:** December 7, 2016 (PSLV-C36)
- **Orbit:** 817 km Polar Sun-Synchronous, 10:30 AM equator crossing
- **Payloads:**
  - **LISS-IV (Linear Imaging Self-Scanning Sensor IV):** 5.8 m resolution, 3-band multispectral (Green, Red, NIR) or single-band mono, 23 km swath.
  - **LISS-III:** 23.5 m resolution, 4 bands (Green: 520–590 nm, Red: 620–680 nm, NIR: 770–860 nm, Short-Wave Infrared - SWIR: 1550–1700 nm), 141 km swath.
  - **AWiFS (Advanced Wide Field Sensor):** 56 m resolution, 4 bands (same as LISS-III), 740 km swath with a 5-day revisit.
- **Strategic Applications:** National Kharif and Rabi crop inventory, forest canopy density classification, groundwater prospecting, drought surveillance.

### 2.5 EOS-04 (RISAT-1A)
- **Launch Date:** February 14, 2022 (PSLV-C52)
- **Sensor Type:** C-band Synthetic Aperture Radar (5.405 GHz)
- **High Resolution Mode (HRS):** 1.0 m spatial resolution, 10 km swath.
- **Strategic Applications:** Agriculture (soil moisture retrieval, paddy crop transplant monitoring), forestry biomass mapping, river basin flood inundation tracking.

### 2.6 EOS-05 (GISAT-1A) — Advanced Real-Time Sentinel
- **Launch Date:** September 4, 2026 (GSLV-F17)
- **Orbit:** Geosynchronous (GEO) at ~36,000 km
- **Payloads:**
  - **High-Resolution Multi-Spectral VNIR:** 6 bands, 42 m GSD.
  - **Hyperspectral VNIR:** 158 contiguous bands, 318 m GSD.
  - **Hyperspectral SWIR:** 256 contiguous bands, 191 m GSD (0.9 to 2.5 μm).
- **Significance:** First Indian geostationary imaging satellite capable of sub-hourly continuous surveillance over the Indian landmass, providing near-real-time monitoring of natural disasters, cyclones, and rapid environmental events.

---

## 3. Data Ingestion & Distribution Platforms

### 3.1 ISRO Bhuvan Portal
- **URL:** [https://bhuvan.nrsc.gov.in](https://bhuvan.nrsc.gov.in)
- **Role:** India's national geoportal providing free multi-temporal satellite imagery, open thematic layers (soil, water, land use), and GIS analytical tools for governance.
- **SatQuery Integration:** SatQuery AI emits GeoJSON features designed for immediate drag-and-drop layer overlay on Bhuvan WebGIS interfaces.

### 3.2 VEDAS (Visualization of Earth observation Data and Archival System)
- **Maintained By:** Space Applications Centre (SAC), ISRO, Ahmedabad
- **URL:** [https://vedas.sac.gov.in](https://vedas.sac.gov.in)
- **Focus:** Thematic applications for disaster management, planetary sciences, vegetation health index (VHI), and solar/wind energy potential.
- **SatQuery Integration:** SatQuery AI report outputs and spectral index calculations adhere directly to VEDAS disaster assessment and agricultural monitoring schemas.
