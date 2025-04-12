import os
import numpy as np
import pandas as pd
import joblib
import gradio as gr
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from utils.advanced_features import extract_advanced_features
import warnings
warnings.filterwarnings('ignore')

def analyze_audio_file(audio_file):
    """Process audio file and extract visualizations and features"""
    y, sr = librosa.load(audio_file)
    os.makedirs('temp', exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.plot(np.linspace(0, len(y)/sr, len(y)), y)
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig('temp/waveform.png')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig('temp/spectrogram.png')
    plt.close()

    features = extract_advanced_features(y, sr)
    
    return y, sr, features

def predict_parkinsons(audio_path):
    """Process audio file without quality checks to ensure dataset files are analyzed"""
    y, sr = librosa.load(audio_path)
    

    
    try:
        if not os.path.exists('models/parkinson_model.pkl'):
            return "Error: Model not found. Please run the training script first."
        model = joblib.load('models/parkinson_model.pkl')
        y, sr, features = analyze_audio_file(audio_path)
        feature_names = []
        if os.path.exists('models/feature_names.txt'):
            with open('models/feature_names.txt', 'r') as f:
                feature_names = f.read().splitlines()
        if feature_names:
            feature_values = []
            for feature_name in feature_names:
                feature_values.append(features.get(feature_name, 0))
            feature_array = np.array([feature_values])
        else:
            features_df = pd.DataFrame([features])
            if 'filename' in features_df.columns:
                features_df = features_df.drop('filename', axis=1)
            if 'label' in features_df.columns:
                features_df = features_df.drop('label', axis=1)
            feature_array = features_df.values
        prediction = 'healthy'
        confidence = 0.5
        
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(feature_array)[0]
            parkinsons_probability = probability[1] if len(probability) > 1 else 0.5
            if parkinsons_probability > 0.7:
                prediction = 'parkinsons'
                confidence = parkinsons_probability
            else:
                prediction = 'healthy'
                confidence = 1 - parkinsons_probability
        else:
            raw_prediction = model.predict(feature_array)[0]
            prediction = raw_prediction
            confidence = 0.8  
        result = f"""
        <h2>Parkinson's Disease Analysis Results</h2>
        
        <div style="display: flex; flex-direction: column; gap: 20px;">
            <div>
                <h3>Audio Visualization</h3>
                <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <p><b>Waveform:</b></p>
                        <img src="file/temp/waveform.png" alt="Audio Waveform" style="max-width: 100%; height: auto;">
                    </div>
                    <div>
                        <p><b>Spectrogram:</b></p>
                        <img src="file/temp/spectrogram.png" alt="Audio Spectrogram" style="max-width: 100%; height: auto;">
                    </div>
                </div>
            </div>
            
            <div>
                <h3>Analysis Results</h3>
                <p><b>Prediction:</b> {prediction.capitalize()}</p>
                <p><b>Confidence:</b> {confidence:.2%}</p>
                
                <div style="background-color: {'#ffebee' if prediction == 'parkinsons' else '#e8f5e9'}; padding: 15px; border-radius: 5px; margin-top: 15px;">
                    <h4>{'Parkinson\'s Disease Indicators Detected' if prediction == 'parkinsons' else 'No Parkinson\'s Disease Indicators Detected'}</h4>
                    
                    <h4>Key Voice Characteristics:</h4>
                    <ul>
                        <li><b>Jitter (frequency variation):</b> {features.get('jitter_relative', 0):.5f} {'(Elevated)' if features.get('jitter_relative', 0) > 0.01 and prediction == 'parkinsons' else '(Normal)'}</li>
                        <li><b>Shimmer (amplitude variation):</b> {features.get('shimmer_relative', 0):.5f} {'(Elevated)' if features.get('shimmer_relative', 0) > 0.06 and prediction == 'parkinsons' else '(Normal)'}</li>
                        <li><b>Harmonic-to-Noise Ratio:</b> {features.get('hnr', 0):.2f} dB {'(Reduced)' if features.get('hnr', 0) < 10 and prediction == 'parkinsons' else '(Normal)'}</li>
                    </ul>
                    
                    <h4>Recommendations:</h4>
                    <ul>
                        {'<li>Consult with a neurologist for a comprehensive evaluation</li><li>Monitor symptoms and keep a daily log</li><li>Consider regular physical therapy and exercise</li><li>Join a Parkinson\'s support group</li>' 
                        if prediction == 'parkinsons' else 
                        '<li>Continue regular health check-ups</li><li>Maintain an active lifestyle</li><li>Consider periodic voice assessments every 6-12 months</li><li>Stay hydrated for vocal health</li>'}
                    </ul>
                </div>
            </div>
        </div>
        """
        return result
    except Exception as e:
        return f"Error analyzing audio: {str(e)}"
interface = gr.Interface(
    fn=predict_parkinsons,
    inputs=gr.Audio(type="filepath", label="Upload Voice Recording (.wav format)"),
    outputs=gr.HTML(),
    title="MediGuardian - Parkinson's Disease Detection",
    description="""Upload a voice recording to check for voice-based indicators of Parkinson's Disease.
    
    For accurate results, please follow these recording guidelines:
    • Record in a quiet environment with minimal background noise
    • Speak at a normal volume and pace
    • Record for at least 5 seconds of continuous speech
    • Sustained vowel sounds like "aaah" work best for analysis
    • Keep the microphone about 6 inches from your mouth
    
    Note: This is intended for educational purposes only and should not replace professional medical advice.
    Many factors can affect voice quality including fatigue, illness, and recording conditions.""",
    examples=[["data/HC_AH/HC_AH_1.wav"], ["data/PD_AH/PD_AH_1.wav"]] if (os.path.exists("data/HC_AH") and 
                                                                        os.listdir("data/HC_AH") and 
                                                                        os.path.exists("data/PD_AH") and 
                                                                        os.listdir("data/PD_AH")) else None,
    allow_flagging="never"
)
if __name__ == "__main__":
    interface.launch(share=True)
