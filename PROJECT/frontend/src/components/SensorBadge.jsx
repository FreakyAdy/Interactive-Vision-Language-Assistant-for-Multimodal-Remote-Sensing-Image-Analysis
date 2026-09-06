import React from 'react';

const SENSOR_CONFIG = {
  'Cartosat-2S': { color: '#FF6B00', type: 'Optical / 0.65m Pan + 2.1m MS' },
  'RISAT-1C': { color: '#00D4AA', type: 'C-band SAR (5.35 GHz)' },
  'ResourceSat-2A': { color: '#4D96FF', type: 'LISS-III (23.5m) / AWiFS' },
  'EOS-04': { color: '#9B51E0', type: 'L-band SAR (1.27 GHz)' },
  'EOS-05': { color: '#FFB800', type: 'GEO VNIR/SWIR Hyperspectral' },
  'Default': { color: '#7E8B9B', type: 'Multispectral Satellite' }
};

export default function SensorBadge({ sensor = 'Cartosat-2S', resolution = '' }) {
  const config = SENSOR_CONFIG[sensor] || SENSOR_CONFIG['Default'];

  return (
    <span 
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '6px',
        padding: '3px 10px',
        borderRadius: '12px',
        fontSize: '12px',
        fontWeight: '600',
        backgroundColor: `${config.color}22`,
        color: config.color,
        border: `1px solid ${config.color}55`,
        fontFamily: 'monospace'
      }}
    >
      <span style={{
        width: '7px',
        height: '7px',
        borderRadius: '50%',
        backgroundColor: config.color,
        boxShadow: `0 0 6px ${config.color}`
      }} />
      {sensor} {resolution ? `• ${resolution}` : ''}
    </span>
  );
}
