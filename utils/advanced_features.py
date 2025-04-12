import numpy as np
import librosa
import scipy
import scipy.stats as stats
import scipy.signal

def extract_advanced_features(audio, sr):
    """Extract a comprehensive set of features specifically tailored for Parkinson's detection."""
    features = {}
    
    audio = audio / (np.max(np.abs(audio)) + 1e-10)
    
    features['rms_energy'] = np.sqrt(np.mean(audio**2))
    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0])
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)[0])
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    for i, mfcc in enumerate(mfccs):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfcc)
        features[f'mfcc_{i+1}_std'] = np.std(mfcc)
        features[f'mfcc_{i+1}_skew'] = stats.skew(mfcc)
        features[f'mfcc_{i+1}_kurtosis'] = stats.kurtosis(mfcc)
    
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch_values = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:
            pitch_values.append(pitch)
    
    if len(pitch_values) > 0:
        features['f0_mean'] = np.mean(pitch_values)  
        features['f0_std'] = np.std(pitch_values)    
        features['f0_min'] = np.min(pitch_values)
        features['f0_max'] = np.max(pitch_values)
        features['f0_range'] = np.max(pitch_values) - np.min(pitch_values)
        
        if len(pitch_values) > 1:
            features['jitter_absolute'] = np.mean(np.abs(np.diff(pitch_values)))
            features['jitter_relative'] = features['jitter_absolute'] / features['f0_mean'] if features['f0_mean'] > 0 else 0
            
            if len(pitch_values) > 5:
                differences = []
                for i in range(2, len(pitch_values)-2):
                    avg = np.mean(pitch_values[i-2:i+3])
                    differences.append(abs(pitch_values[i] - avg))
                features['jitter_ppq5'] = np.mean(differences) / features['f0_mean'] if features['f0_mean'] > 0 else 0
    else:
        features['f0_mean'] = 0
        features['f0_std'] = 0
        features['f0_min'] = 0
        features['f0_max'] = 0
        features['f0_range'] = 0
        features['jitter_absolute'] = 0
        features['jitter_relative'] = 0
        features['jitter_ppq5'] = 0
    
    amplitude_envelope = np.abs(librosa.stft(audio))
    mean_amplitude = np.mean(amplitude_envelope, axis=0)
    
    if len(mean_amplitude) > 1:
        features['shimmer_absolute'] = np.mean(np.abs(np.diff(mean_amplitude)))
        features['shimmer_relative'] = features['shimmer_absolute'] / np.mean(mean_amplitude) if np.mean(mean_amplitude) > 0 else 0
    else:
        features['shimmer_absolute'] = 0
        features['shimmer_relative'] = 0
    
    try:
        harmonic = librosa.effects.harmonic(audio)
        percussive = librosa.effects.percussive(audio)
        harmonic_energy = np.sum(harmonic**2)
        percussive_energy = np.sum(percussive**2)
        features['hnr'] = 10 * np.log10(harmonic_energy / (percussive_energy + 1e-10))
    except:
        features['hnr'] = 0
    
    try:
        S = np.abs(librosa.stft(audio))
        formants = []
        for frame in range(S.shape[1]):
            spec = S[:, frame]
            peaks, _ = scipy.signal.find_peaks(spec, height=np.max(spec)/10)
            if len(peaks) > 0:
                formant_freqs = librosa.fft_frequencies(sr=sr)[peaks]
                formants.extend(formant_freqs)
        
        if formants:
            features['formant_mean'] = np.mean(formants)
            features['formant_std'] = np.std(formants)
        else:
            features['formant_mean'] = 0
            features['formant_std'] = 0
    except:
        features['formant_mean'] = 0
        features['formant_std'] = 0
    
    return features