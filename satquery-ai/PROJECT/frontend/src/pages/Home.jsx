import React from 'react';

export default function Home({ onStartDemo }) {
  return (
    <div style={{ padding: '40px 20px', maxWidth: '1000px', margin: '0 auto', textAlign: 'center' }}>
      <div style={{
        display: 'inline-block',
        padding: '6px 14px',
        background: '#FF6B0022',
        border: '1px solid #FF6B0055',
        borderRadius: '20px',
        color: '#FF6B00',
        fontSize: '13px',
        fontWeight: 'bold',
        marginBottom: '20px'
      }}>
        🚀 SMART INDIA HACKATHON 2026 • PROBLEM STATEMENT SIH26167
      </div>

      <h1 style={{ fontSize: '38px', fontWeight: '800', color: '#F0F4F8', marginBottom: '16px', lineHeight: '1.2' }}>
        SatQuery <span style={{ color: '#FF6B00' }}>AI</span>
      </h1>

      <p style={{ fontSize: '18px', color: '#A0AEC0', maxWidth: '750px', margin: '0 auto 30px', lineHeight: '1.6' }}>
        Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis.
        Conversational queries, automated 12-stage change detection, STSF-Net pseudo-change suppression,
        and calibrated ISRO satellite sensor support.
      </p>

      <div style={{ display: 'flex', justifyContent: 'center', gap: '16px', marginBottom: '50px' }}>
        <button
          onClick={onStartDemo}
          style={{
            padding: '14px 28px',
            background: '#FF6B00',
            color: '#FFFFFF',
            border: 'none',
            borderRadius: '6px',
            fontSize: '15px',
            fontWeight: 'bold',
            cursor: 'pointer'
          }}
        >
          Launch Mission Control 🛰️
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '20px', textAlign: 'left' }}>
        <div style={{ background: '#0F1C30', padding: '20px', borderRadius: '8px', border: '1px solid #1D2A42' }}>
          <div style={{ fontSize: '24px', marginBottom: '8px' }}>⚡</div>
          <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#F0F4F8', marginBottom: '6px' }}>
            12-Stage Change Engine
          </div>
          <div style={{ fontSize: '12px', color: '#7E8B9B', lineHeight: '1.5' }}>
            Sub-pixel SIFT co-registration, histogram normalization, and bimodal confidence scoring.
          </div>
        </div>

        <div style={{ background: '#0F1C30', padding: '20px', borderRadius: '8px', border: '1px solid #1D2A42' }}>
          <div style={{ fontSize: '24px', marginBottom: '8px' }}>🛡️</div>
          <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#00D4AA', marginBottom: '6px' }}>
            STSF-Net Filter
          </div>
          <div style={{ fontSize: '12px', color: '#7E8B9B', lineHeight: '1.5' }}>
            Spatiotemporal spectral fusion eliminates false alarms from seasonal phenology and lighting.
          </div>
        </div>

        <div style={{ background: '#0F1C30', padding: '20px', borderRadius: '8px', border: '1px solid #1D2A42' }}>
          <div style={{ fontSize: '24px', marginBottom: '8px' }}>🤖</div>
          <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#4D96FF', marginBottom: '6px' }}>
            GeoChat-7B VLM
          </div>
          <div style={{ fontSize: '12px', color: '#7E8B9B', lineHeight: '1.5' }}>
            Remote sensing instruction-tuned multimodal model generates conversational spatial reports.
          </div>
        </div>

        <div style={{ background: '#0F1C30', padding: '20px', borderRadius: '8px', border: '1px solid #1D2A42' }}>
          <div style={{ fontSize: '24px', marginBottom: '8px' }}>🛰️</div>
          <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#FFB800', marginBottom: '6px' }}>
            ISRO Sensors
          </div>
          <div style={{ fontSize: '12px', color: '#7E8B9B', lineHeight: '1.5' }}>
            Direct calibration for Cartosat-2S, RISAT-1C, ResourceSat-2A, EOS-04, and EOS-05 GISAT-1A.
          </div>
        </div>
      </div>
    </div>
  );
}
