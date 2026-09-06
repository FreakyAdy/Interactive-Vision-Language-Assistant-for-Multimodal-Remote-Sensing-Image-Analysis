import React, { useState, useEffect } from 'react';
import ImageUploader from '../components/ImageUploader';
import QueryInput from '../components/QueryInput';
import ResultViewer from '../components/ResultViewer';
import ChangeDetectionView from '../components/ChangeDetectionView';
import SpectralChart from '../components/SpectralChart';
import MapOverlay from '../components/MapOverlay';
import { fetchDemoScenarios, runDemoScenario, submitAnalysisQuery } from '../utils/api';

export default function Analyze() {
  const [scenarios, setScenarios] = useState([]);
  const [selectedScenario, setSelectedScenario] = useState(null);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [changeResult, setChangeResult] = useState(null);

  useEffect(() => {
    fetchDemoScenarios().then(data => {
      setScenarios(data);
      if (data.length > 0) handleSelectScenario(data[0]);
    });
  }, []);

  const handleSelectScenario = async (sc) => {
    setSelectedScenario(sc);
    setQuery(sc.query || 'Analyze this scenario');
    setLoading(true);
    try {
      const data = await runDemoScenario(sc.id);
      setResult(data);
      if (data.change_detection) {
        setChangeResult(data.change_detection);
      } else {
        setChangeResult(null);
      }
    } catch (err) {
      console.error('Scenario load error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleCustomSubmit = async () => {
    if (!query) return;
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('query', query);
      const res = await submitAnalysisQuery(formData);
      setResult(res);
      if (res.change_detection) setChangeResult(res.change_detection);
    } catch (err) {
      console.error('Analysis error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', padding: '20px' }}>
      {/* Scenario Selector Ribbon */}
      <div style={{ display: 'flex', gap: '8px', overflowX: 'auto', paddingBottom: '6px' }}>
        {scenarios.map((sc) => (
          <button
            key={sc.id}
            onClick={() => handleSelectScenario(sc)}
            style={{
              padding: '8px 14px',
              borderRadius: '6px',
              border: selectedScenario?.id === sc.id ? '1px solid #FF6B00' : '1px solid #1D2A42',
              background: selectedScenario?.id === sc.id ? '#FF6B0022' : '#0F1C30',
              color: selectedScenario?.id === sc.id ? '#FF6B00' : '#A0AEC0',
              fontSize: '12px',
              fontWeight: '600',
              whiteSpace: 'nowrap',
              cursor: 'pointer'
            }}
          >
            {sc.id}: {sc.title || sc.name}
          </button>
        ))}
      </div>

      {/* Query Bar */}
      <QueryInput
        query={query}
        setQuery={setQuery}
        onSubmit={handleCustomSubmit}
        loading={loading}
      />

      {/* Main Analysis Display Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '20px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <ResultViewer result={result} />
          {changeResult && <ChangeDetectionView changeResult={changeResult} />}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <MapOverlay geojson={result?.geojson} />
          {result?.indices && <SpectralChart indices={result.indices} />}
        </div>
      </div>
    </div>
  );
}
