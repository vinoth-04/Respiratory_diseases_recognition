from flask import Flask, request, render_template, jsonify
import librosa
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Mock model (replace this with your actual loaded model later)
def mock_predict(features):
    """Mock prediction function - replace with your actual model"""
    # Return random prediction for testing
    classes = ['Bronchiectasis', 'COPD', 'Healthy', 'Other', 'Pneumonia', 'URTI', 'LRTI']
    import random
    pred = random.choice(classes)
    probs = np.random.dirichlet(np.ones(len(classes)))  # random probabilities that sum to 1
    return pred, probs

def extract_features(audio_path, sr=22050):
    """Extract MFCC features from audio file"""
    try:
        audio, sample_rate = librosa.load(audio_path, sr=sr)
        
        if len(audio) < sr * 4:
            audio = np.pad(audio, (0, sr * 4 - len(audio)), mode='constant')
        else:
            audio = audio[:sr * 4]
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        
        mean = np.mean(mfccs, axis=1)
        std = np.std(mfccs, axis=1)
        min_ = np.min(mfccs, axis=1)
        max_ = np.max(mfccs, axis=1)
        
        features = np.concatenate([mean, std, min_, max_])
        
        return features.reshape(1, -1)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Lung Condition Predictor</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .upload-box { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
            .result { background: #f0f8ff; padding: 20px; margin: 20px 0; border-radius: 5px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <h1>🫁 Lung Condition Predictor</h1>
        <p>Upload a respiratory sound recording (WAV, MP3) to predict lung condition.</p>
        
        <div class="upload-box">
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" name="audio" accept=".wav,.mp3,.m4a,.flac" required>
                <br><br>
                <button type="submit">Predict Condition</button>
            </form>
        </div>
        
        <div id="result" class="result" style="display:none;"></div>
        
        <script>
            document.getElementById('uploadForm').onsubmit = async function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                const fileInput = document.querySelector('input[type="file"]');
                formData.append('audio', fileInput.files[0]);
                
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '<p>Processing audio...</p>';
                resultDiv.style.display = 'block';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        resultDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
                    } else {
                        resultDiv.innerHTML = `
                            <h3>Prediction Result</h3>
                            <p><strong>Predicted Condition:</strong> ${data.prediction}</p>
                            <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
                            <p><strong>All Probabilities:</strong></p>
                            <ul>
                                ${Object.entries(data.probabilities).map(([condition, prob]) => 
                                    `<li>${condition}: ${(prob * 100).toFixed(2)}%</li>`
                                ).join('')}
                            </ul>
                        `;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                }
            };
        </script>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'})
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        # Save and process file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        features = extract_features(filepath)
        if features is None:
            return jsonify({'error': 'Could not process audio file'})
        
        # Use mock prediction for now - REPLACE WITH YOUR ACTUAL MODEL
        prediction, probabilities = mock_predict(features)
        
        class_names = ['Bronchiectasis', 'COPD', 'Healthy', 'Other', 'Pneumonia', 'URTI', 'LRTI']
        prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        confidence = float(np.max(probabilities))
        
        os.remove(filepath)
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': prob_dict
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("Starting Lung Condition Predictor Web App...")
    print("Visit http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)