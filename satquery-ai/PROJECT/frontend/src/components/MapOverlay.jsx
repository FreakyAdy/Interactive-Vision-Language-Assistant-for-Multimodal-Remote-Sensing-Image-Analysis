import React from 'react';

export default function MapOverlay({ geojson, height = '300px' }) {
  const featureCount = geojson?.features ? geojson.features.length : 0;

  return (
    <div style={{
      height,
      background: '#0B1628',
      borderRadius: '8px',
      border: '1px solid #1D2A42',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Visual map grid representation */}
      <div style={{
        position: 'absolute',
        top: 0, left: 0, right: 0, bottom: 0,
        backgroundImage: 'radial-gradient(#1D2A42 1px, transparent 1px)',
        backgroundSize: '20px 20px',
        opacity: 0.6
      }} />

      <div style={{ zIndex: 1, textAlign: 'center', padding: '20px' }}>
        <div style={{ fontSize: '32px', marginBottom: '8px' }}>🗺️</div>
        <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#F0F4F8' }}>
          Interactive Geospatial GeoJSON Layer
        </div>
        <div style={{ fontSize: '12px', color: '#00D4AA', marginTop: '4px', fontFamily: 'monospace' }}>
          {featureCount > 0 ? `${featureCount} Polygons / Bounding Boxes Detected` : 'No spatial vector overlay active'}
        </div>
        <div style={{ fontSize: '11px', color: '#7E8B9B', marginTop: '6px' }}>
          Coordinate Reference System: EPSG:4326 (WGS 84)
        </div>
      </div>
    </div>
  );
}
