import React from 'react';

export default function SpectralChart({ indices = { ndvi: 0.65, ndwi: 0.12, ndbi: -0.25, evi: 0.58, rvi: 0.72 } }) {
  const items = Object.entries(indices);

  return (
    <div style={{ background: '#0F1C30', padding: '16px', borderRadius: '8px', border: '1px solid #1D2A42' }}>
      <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#00D4AA', marginBottom: '12px' }}>
        📊 Multimodal Spectral Indices Profile
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {items.map(([key, val]) => {
          const num = typeof val === 'number' ? val : 0;
          const displayVal = num.toFixed(3);
          // Normalize range [-1, 1] to [0, 100]%
          const widthPct = Math.max(0, Math.min(100, ((num + 1) / 2) * 100));
          let barColor = '#4D96FF';
          if (key === 'ndvi') barColor = '#00D4AA';
          if (key === 'ndwi') barColor = '#00B4D8';
          if (key === 'ndbi') barColor = '#FF6B00';
          if (key === 'evi') barColor = '#52B788';

          return (
            <div key={key}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '3px' }}>
                <span style={{ fontWeight: '600', color: '#F0F4F8' }}>{key.toUpperCase()}</span>
                <span style={{ fontFamily: 'monospace', color: barColor }}>{displayVal}</span>
              </div>
              <div style={{ height: '6px', background: '#1D2A42', borderRadius: '3px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${widthPct}%`, background: barColor, borderRadius: '3px' }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
