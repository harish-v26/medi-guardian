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



if __name__ == "__main__":
    analyze_audio_file()
