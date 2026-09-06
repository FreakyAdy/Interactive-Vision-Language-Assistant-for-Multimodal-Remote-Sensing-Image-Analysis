import React from 'react';

const SUGGESTIONS = [
  'Detect flood extent and submerged highway sections',
  'Quantify forest canopy loss between 2024 and 2026',
  'Measure urban built-up expansion in outer ring road',
  'Identify cargo ships and berths in harbor area',
  'Assess crop health and vegetation index anomalies'
];

export default function QueryInput({ query, setQuery, onSubmit, loading }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      <div style={{ display: 'flex', gap: '10px' }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !loading && onSubmit()}
          placeholder="Ask a question about this satellite scene (e.g. 'How much water spread over farmland?')..."
          style={{
            flex: 1,
            padding: '12px 16px',
            background: '#0B1628',
            border: '1px solid #2A3B5C',
            borderRadius: '6px',
            color: '#F0F4F8',
            fontSize: '14px',
            outline: 'none'
          }}
        />
        <button
          onClick={onSubmit}
          disabled={loading || !query.trim()}
          style={{
            padding: '12px 24px',
            background: loading ? '#4A5568' : '#FF6B00',
            color: '#FFFFFF',
            fontWeight: 'bold',
            border: 'none',
            borderRadius: '6px',
            cursor: loading ? 'not-allowed' : 'pointer',
            transition: 'background 0.2s'
          }}
        >
          {loading ? 'Analyzing...' : 'Execute Analysis 🚀'}
        </button>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', alignItems: 'center' }}>
        <span style={{ fontSize: '11px', color: '#7E8B9B' }}>Suggestions:</span>
        {SUGGESTIONS.map((text, idx) => (
          <button
            key={idx}
            onClick={() => setQuery(text)}
            style={{
              background: '#14223A',
              border: '1px solid #1D2A42',
              borderRadius: '12px',
              padding: '2px 8px',
              fontSize: '11px',
              color: '#4D96FF',
              cursor: 'pointer'
            }}
          >
            {text}
          </button>
        ))}
      </div>
    </div>
  );
}
