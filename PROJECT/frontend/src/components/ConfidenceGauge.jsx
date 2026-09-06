import React from 'react';

export default function ConfidenceGauge({ score = 0.85, label = 'Detection Confidence' }) {
  const percentage = Math.round(score * 100);
  
  let color = '#FF4D4D';
  if (percentage >= 80) color = '#00D4AA';
  else if (percentage >= 60) color = '#FFB800';
  else if (percentage >= 40) color = '#FF6B00';

  const strokeDasharray = `${percentage}, 100`;

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
      <div style={{ position: 'relative', width: '60px', height: '60px' }}>
        <svg viewBox="0 0 36 36" style={{ width: '100%', height: '100%', transform: 'rotate(-90deg)' }}>
          <path
            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
            fill="none"
            stroke="#1D2A42"
            strokeWidth="3.5"
          />
          <path
            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
            fill="none"
            stroke={color}
            strokeWidth="3.5"
            strokeDasharray={strokeDasharray}
            strokeLinecap="round"
          />
        </svg>
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '13px',
          fontWeight: 'bold',
          color: '#F0F4F8'
        }}>
          {percentage}%
        </div>
      </div>
      <div>
        <div style={{ fontSize: '13px', fontWeight: '600', color: '#F0F4F8' }}>{label}</div>
        <div style={{ fontSize: '11px', color: '#7E8B9B' }}>
          {percentage >= 80 ? 'High confidence (Otsu η > 0.75)' : 'Moderate confidence / Review recommended'}
        </div>
      </div>
    </div>
  );
}
