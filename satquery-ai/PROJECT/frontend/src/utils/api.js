/**
 * SatQuery AI - API Client Utilities
 * Connects to FastAPI Backend with Automatic DEMO_MODE Fallback
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
      { id: 'SCN-01', title: 'Assam Flood Inundation & Infrastructure Impact', sensor: 'ResourceSat-2A' },
      { id: 'SCN-02', title: 'Western Ghats Forest Canopy Loss', sensor: 'Cartosat-2S' },
      { id: 'SCN-03', title: 'Bengaluru Peri-Urban Expansion', sensor: 'ResourceSat-2A' },
      { id: 'SCN-04', title: 'Punjab Crop Phenology & Residue Burning', sensor: 'EOS-04' },
      { id: 'SCN-05', title: 'Visakhapatnam Harbor Maritime Surveillance', sensor: 'Cartosat-2S' }
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
