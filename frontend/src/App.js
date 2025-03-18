import React, { useState } from 'react';
import './styles.css';

function App() {
  const [predictedTexture, setPredictedTexture] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleImageUpload = async (e) => {
    e.preventDefault();
    setIsLoading(true);

    const file = e.target.files[0];
    const formData = new FormData();
    formData.append('image', file);

    try {
      // Send the image to the Flask backend
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to predict soil texture');
      }

      const result = await response.json();
      setPredictedTexture(result.predicted_texture);
    } catch (error) {
      console.error(error);
      alert('An error occurred. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>AgriSense</h1>
      <p>Upload an image of soil to predict its texture.</p>
      <input
        type="file"
        id="image"
        name="image"
        accept="image/*"
        onChange={handleImageUpload}
        disabled={isLoading}
      />
      <div id="result">
        <h2>Prediction Result:</h2>
        {isLoading ? (
          <p>Loading...</p>
        ) : (
          <p id="predicted-texture">{predictedTexture || '-'}</p>
        )}
      </div>
    </div>
  );
}

export default App;