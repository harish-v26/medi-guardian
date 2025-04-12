import os
import librosa
import numpy as np
import pandas as pd

def load_audio_files(directory):
    file_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            file_path = os.path.join(directory, filename)

            y, sr = librosa.load(file_path, sr=None)
            file_data.append({
                'filename': filename,
                'audio': y,
                'sample_rate': sr,
                'length': len(y) / sr  
            })
    return file_data

def extract_features(audio_data, sample_rate):
    features = {}
    
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0])
    
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0])
    
    features['rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0])
    
    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_data)[0])
    
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    for i, mfcc in enumerate(mfccs):
        features[f'mfcc_{i+1}'] = np.mean(mfcc)
    
    pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
    pitch_values = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:
            pitch_values.append(pitch)
    
    pitch_values = np.array(pitch_values)
    if len(pitch_values) > 0:
        features['pitch_mean'] = np.mean(pitch_values)
        features['pitch_std'] = np.std(pitch_values)
        
        if len(pitch_values) > 1:
            features['jitter'] = np.mean(np.abs(np.diff(pitch_values))) / np.mean(pitch_values)
        else:
            features['jitter'] = 0
    else:
        features['pitch_mean'] = 0
        features['pitch_std'] = 0
        features['jitter'] = 0
    
    harmonic = librosa.effects.harmonic(audio_data)
    percussive = librosa.effects.percussive(audio_data)
    features['harmonic_ratio'] = np.mean(harmonic**2) / np.mean(percussive**2) if np.mean(percussive**2) > 0 else 0
    
    return features

def prepare_dataset(files, label):
    dataset = []
    for file in files:
        features = extract_features(file['audio'], file['sample_rate'])
        features['filename'] = file['filename']
        features['label'] = label
        dataset.append(features)
    return dataset
