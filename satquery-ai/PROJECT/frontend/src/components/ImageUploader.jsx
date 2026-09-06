import React, { useRef, useState } from 'react';
import { readFileAsDataURL } from '../utils/imageUtils';

export default function ImageUploader({ label = 'Upload Satellite Image', onImageSelected, accept = '.tif,.tiff,.png,.jpg,.jpeg' }) {
  const [preview, setPreview] = useState(null);
  const [fileName, setFileName] = useState('');
  const fileInputRef = useRef(null);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setFileName(file.name);
    try {
      const dataUrl = await readFileAsDataURL(file);
      setPreview(dataUrl);
      if (onImageSelected) onImageSelected(file, dataUrl);
    } catch (err) {
      console.error('File read error:', err);
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setFileName(file.name);
      const dataUrl = await readFileAsDataURL(file);
      setPreview(dataUrl);
      if (onImageSelected) onImageSelected(file, dataUrl);
    }
  };

  return (
    <div
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
      style={{
        border: '2px dashed #2A3B5C',
        borderRadius: '8px',
        padding: '20px',
        textAlign: 'center',
        background: '#0B1628',
        cursor: 'pointer',
        transition: 'border-color 0.2s'
      }}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />
      {preview ? (
        <div>
          <img
            src={preview}
            alt="Preview"
            style={{ maxHeight: '140px', maxWidth: '100%', borderRadius: '6px', objectFit: 'contain' }}
          />
          <div style={{ fontSize: '11px', color: '#00D4AA', marginTop: '6px', fontFamily: 'monospace' }}>
            {fileName}
          </div>
        </div>
      ) : (
        <div>
          <div style={{ fontSize: '24px', marginBottom: '8px' }}>🛰️</div>
          <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#F0F4F8' }}>{label}</div>
          <div style={{ fontSize: '11px', color: '#7E8B9B', marginTop: '4px' }}>
            Drag & drop GeoTIFF or PNG/JPG
          </div>
        </div>
      )}
    </div>
  );
}
