import React from 'react';

export default function About() {
  return (
    <div style={{ maxWidth: '850px', margin: '0 auto', padding: '30px 20px', color: '#E2E8F0', lineHeight: '1.6' }}>
      <h2 style={{ fontSize: '26px', color: '#F0F4F8', borderBottom: '1px solid #1D2A42', paddingBottom: '12px' }}>
        About SatQuery AI
      </h2>
      <p style={{ marginTop: '16px' }}>
        SatQuery AI was conceived and developed for <strong>Smart India Hackathon 2026 (SIH26167)</strong>,
        under the guidance and problem statement issued by the <strong>Indian Space Research Organisation (ISRO)</strong> and
        the <strong>Space Applications Centre (SAC), Ahmedabad</strong>.
      </p>

      <h3 style={{ fontSize: '18px', color: '#FF6B00', marginTop: '24px' }}>The Operational Challenge</h3>
      <p>
        India operates one of the world's most advanced constellations of Earth observation satellites, including Cartosat,
        RISAT, and ResourceSat. However, extracting actionable insights requires specialized GIS expertise, manual band math,
        and complex multi-temporal analysis that takes days. SatQuery AI democratizes satellite intelligence, allowing
        field responders to converse with Earth observation data directly.
      </p>

      <h3 style={{ fontSize: '18px', color: '#00D4AA', marginTop: '24px' }}>Key Innovations</h3>
      <ul style={{ paddingLeft: '20px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <li><strong>STSF-Net Pseudo-Change Suppression:</strong> Deep cross-temporal attention filtering of phenological and illumination noise.</li>
        <li><strong>Bimodal Confidence Calibration:</strong> Combines empirical spatial separability (Otsu η) with semantic VLM token probability.</li>
        <li><strong>6-Way Agentic Query Router:</strong> Domain-specific natural language routing for automated tool planning.</li>
        <li><strong>Calibrated ISRO Constellation Support:</strong> Direct sensor-specific calibration factors for Cartosat, RISAT, ResourceSat, EOS-04, and EOS-05.</li>
      </ul>

      <h3 style={{ fontSize: '18px', color: '#4D96FF', marginTop: '24px' }}>Hackathon Team</h3>
      <p>
        Developed by Team SatQuery AI for SIH 2026. Built with Python, FastAPI, PyTorch, GeoChat-7B, OpenCV, Leaflet, and React.
      </p>
    </div>
  );
}
