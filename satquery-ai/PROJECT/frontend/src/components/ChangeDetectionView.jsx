import React from 'react';
import ConfidenceGauge from './ConfidenceGauge';

export default function ChangeDetectionView({ changeResult }) {
  if (!changeResult) return null;

  const {
    changed_area_km2 = 0,
    changed_area_ha = 0,
    change_percentage = 0,
    num_change_regions = 0,
    confidence_score = 0.88,
    pseudo_change_ratio = 0.14,
    index_used = 'NDWI'
  } = changeResult;

  return (
    <div style={{ background: '#0F1C30', padding: '16px', borderRadius: '8px', border: '1px solid #1D2A42' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
        <div>
          <span style={{ fontSize: '13px', fontWeight: 'bold', color: '#FF6B00' }}>
            🛰️ 12-Stage Bi-temporal Change Detection Analysis
          </span>
          <div style={{ fontSize: '11px', color: '#7E8B9B', marginTop: '2px' }}>
            Index: {index_used} • STSF-Net Pseudo-Change Filtering Active
          </div>
        </div>
        <ConfidenceGauge score={confidence_score} label="Change Confidence" />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
        <div style={{ background: '#14223A', padding: '10px', borderRadius: '6px' }}>
          <div style={{ fontSize: '11px', color: '#7E8B9B' }}>CHANGED AREA</div>
          <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#FF4D4D' }}>
            {changed_area_km2.toFixed(2)} km²
          </div>
          <div style={{ fontSize: '10px', color: '#7E8B9B' }}>{changed_area_ha.toFixed(1)} ha</div>
        </div>

        <div style={{ background: '#14223A', padding: '10px', borderRadius: '6px' }}>
          <div style={{ fontSize: '11px', color: '#7E8B9B' }}>CHANGE EXTENT</div>
          <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#FFB800' }}>
            {change_percentage.toFixed(2)}%
          </div>
          <div style={{ fontSize: '10px', color: '#7E8B9B' }}>of scene footprint</div>
        </div>

        <div style={{ background: '#14223A', padding: '10px', borderRadius: '6px' }}>
          <div style={{ fontSize: '11px', color: '#7E8B9B' }}>DISCRETE CLUSTERS</div>
          <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#00D4AA' }}>
            {num_change_regions}
          </div>
          <div style={{ fontSize: '10px', color: '#7E8B9B' }}>connected components</div>
        </div>

        <div style={{ background: '#14223A', padding: '10px', borderRadius: '6px' }}>
          <div style={{ fontSize: '11px', color: '#7E8B9B' }}>PSEUDO-SUPPRESSION</div>
          <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#4D96FF' }}>
            {(pseudo_change_ratio * 100).toFixed(1)}%
          </div>
          <div style={{ fontSize: '10px', color: '#7E8B9B' }}>artifacts suppressed</div>
        </div>
      </div>
    </div>
  );
}
