/**
 * SatQuery AI - API Client Utilities
 * Connects to FastAPI Backend with Automatic DEMO_MODE Fallback
 * Strictly compliant with SIH26167 endpoints
 */

const API_BASE_URL = import.meta?.env?.VITE_API_BASE_URL || 'http://localhost:8000';

export async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { method: 'GET' });
    if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('Backend offline or unreachable, switching to DEMO_MODE:', err.message);
    return { status: 'offline', mode: 'demo_fallback', message: 'Backend unreachable' };
  }
}

export async function fetchDemoScenarios() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/demo/scenarios`);
    if (!res.ok) throw new Error('Failed to load scenarios');
    return await res.json();
  } catch (err) {
    console.warn('Using local fallback scenarios:', err);
    return [
      { id: 'sih_q1_single_vqa', title: 'Q1: Land-Cover Description (Single Optical)', sensor: 'Cartosat-2S' },
      { id: 'sih_q2_single_grounding', title: 'Q2: Water Body Grounding (Single Optical)', sensor: 'Cartosat-2S' },
      { id: 'sih_q3_bitemporal_change', title: 'Q3: Bi-Temporal Flood Change Detection', sensor: 'Cartosat-2S' },
      { id: 'sih_q4_cross_modal_fusion', title: 'Q4: Optical-SAR Joint Fusion (Cartosat + RISAT)', sensor: 'Cartosat-2S + RISAT-1C' },
      { id: 'sih_q5_bitemporal_cdvqa', title: 'Q5: Urban Expansion CDVQA', sensor: 'Cartosat-3' }
    ];
  }
}

export async function runDemoScenario(scenarioId) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/demo/run/${scenarioId}`, { method: 'POST' });
    if (!res.ok) throw new Error('Demo scenario failed');
    return await res.json();
  } catch (err) {
    console.error('Error running demo scenario:', err);
    throw err;
  }
}

export async function submitAnalysisQuery(formData) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error(`Analysis failed: ${res.statusText}`);
    return await res.json();
  } catch (err) {
    console.error('API analyze error:', err);
    throw err;
  }
}

export async function submitChangeDetection(formData) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/change-detection`, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error(`Change detection failed: ${res.statusText}`);
    return await res.json();
  } catch (err) {
    console.error('API change-detection error:', err);
    throw err;
  }
}

export async function checkInputCompatibility(formData) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/compatibility-check`, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error(`Compatibility check failed: ${res.statusText}`);
    return await res.json();
  } catch (err) {
    console.error('API compatibility-check error:', err);
    throw err;
  }
}

export async function submitCrossModalAnalysis(formData) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/cross-modal-analysis`, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error(`Cross-modal analysis failed: ${res.statusText}`);
    return await res.json();
  } catch (err) {
    console.error('API cross-modal error:', err);
    throw err;
  }
}

export async function submitCDVQA(formData) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/cdvqa`, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error(`CDVQA failed: ${res.statusText}`);
    return await res.json();
  } catch (err) {
    console.error('API CDVQA error:', err);
    throw err;
  }
}
