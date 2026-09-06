import React from 'react';
import ConfidenceGauge from './ConfidenceGauge';
import SensorBadge from './SensorBadge';
import { downloadJsonFile, downloadTextReport } from '../utils/imageUtils';

export default function ResultViewer({ result }) {
  if (!result) return null;

  const {
    task_category = 'GENERAL_VQA',
    answer = '',
    confidence = 0.85,
    metrics = {},
    geojson = null,
    sensor = 'Cartosat-2S',
    pipeline_trace = []
  } = result;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', background: '#0F1C30', padding: '20px', borderRadius: '8px', border: '1px solid #1D2A42' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #1D2A42', paddingBottom: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <SensorBadge sensor={sensor} />
          <span style={{ fontSize: '12px', background: '#FF6B0022', color: '#FF6B00', border: '1px solid #FF6B0055', padding: '2px 8px', borderRadius: '4px', fontFamily: 'monospace' }}>
            {task_category}
          </span>
        </div>
        <ConfidenceGauge score={confidence} />
      </div>

      <div style={{ fontSize: '14px', lineHeight: '1.6', color: '#E2E8F0', whiteSpace: 'pre-wrap', background: '#0B1628', padding: '16px', borderRadius: '6px', border: '1px solid #1A2740' }}>
        {answer}
      </div>

      {metrics && Object.keys(metrics).length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '10px' }}>
          {Object.entries(metrics).map(([key, val]) => (
            <div key={key} style={{ background: '#14223A', padding: '10px', borderRadius: '6px', border: '1px solid #1D2A42' }}>
              <div style={{ fontSize: '11px', color: '#7E8B9B', textTransform: 'uppercase' }}>{key.replace(/_/g, ' ')}</div>
              <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#00D4AA', marginTop: '2px' }}>
                {typeof val === 'number' ? val.toLocaleString() : String(val)}
              </div>
            </div>
          ))}
        </div>
      )}

      <div style={{ display: 'flex', gap: '10px', paddingTop: '10px' }}>
        <button
          onClick={() => downloadTextReport(answer)}
          style={{ padding: '8px 14px', background: '#1D2A42', color: '#F0F4F8', border: 'none', borderRadius: '4px', cursor: 'pointer', fontSize: '12px' }}
        >
          📄 Export Markdown Report
        </button>
        {geojson && (
          <button
            onClick={() => downloadJsonFile(geojson, 'satquery_features.geojson')}
            style={{ padding: '8px 14px', background: '#00D4AA22', color: '#00D4AA', border: '1px solid #00D4AA55', borderRadius: '4px', cursor: 'pointer', fontSize: '12px' }}
          >
            🗺️ Export GeoJSON
          </button>
        )}
      </div>
    </div>
  );
}
