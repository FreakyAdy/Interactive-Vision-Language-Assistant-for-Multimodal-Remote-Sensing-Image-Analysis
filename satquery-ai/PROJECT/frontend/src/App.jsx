import React, { useState } from 'react';
import Home from './pages/Home';
import Analyze from './pages/Analyze';
import About from './pages/About';

export default function App() {
  const [currentPage, setCurrentPage] = useState('analyze');

  return (
    <div style={{ minHeight: '100vh', background: '#0B1628', color: '#F0F4F8', fontFamily: 'Inter, system-ui, sans-serif' }}>
      {/* Top Navigation Bar */}
      <header style={{
        height: '60px',
        borderBottom: '1px solid #1D2A42',
        background: '#0B1628',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 24px',
        position: 'sticky',
        top: 0,
        zIndex: 100
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer' }} onClick={() => setCurrentPage('home')}>
          <div style={{
            width: '32px',
            height: '32px',
            borderRadius: '6px',
            background: 'linear-gradient(135deg, #FF6B00 0%, #FF8800 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontWeight: 'bold',
            fontSize: '18px'
          }}>
            🛰️
          </div>
          <div>
            <div style={{ fontSize: '16px', fontWeight: '800', letterSpacing: '0.5px' }}>
              SatQuery <span style={{ color: '#FF6B00' }}>AI</span>
            </div>
            <div style={{ fontSize: '10px', color: '#7E8B9B', marginTop: '-2px' }}>
              ISRO SAC SIH 2026 • SIH26167
            </div>
          </div>
        </div>

        <nav style={{ display: 'flex', gap: '8px' }}>
          {[
            { id: 'home', label: 'Overview' },
            { id: 'analyze', label: 'Mission Control' },
            { id: 'about', label: 'Documentation' }
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => setCurrentPage(item.id)}
              style={{
                padding: '6px 14px',
                borderRadius: '6px',
                border: 'none',
                background: currentPage === item.id ? '#1D2A42' : 'transparent',
                color: currentPage === item.id ? '#00D4AA' : '#A0AEC0',
                fontSize: '13px',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.15s'
              }}
            >
              {item.label}
            </button>
          ))}
        </nav>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{
            padding: '3px 8px',
            borderRadius: '4px',
            background: '#00D4AA22',
            border: '1px solid #00D4AA55',
            color: '#00D4AA',
            fontSize: '11px',
            fontWeight: '600',
            fontFamily: 'monospace'
          }}>
            ● SYSTEM ACTIVE (DEMO_MODE)
          </span>
        </div>
      </header>

      {/* Main Page Body */}
      <main>
        {currentPage === 'home' && <Home onStartDemo={() => setCurrentPage('analyze')} />}
        {currentPage === 'analyze' && <Analyze />}
        {currentPage === 'about' && <About />}
      </main>
    </div>
  );
}
