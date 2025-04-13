import os
import numpy as np
import pandas as pd
import librosa
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils.advanced_features import extract_advanced_features

def train_optimized_model():
    """Train an optimized model for Parkinson's disease detection from voice samples"""
    all_features = []
    
    print("Loading and processing audio files...")
    healthy_dir = "data/HC_AH/"
    if os.path.exists(healthy_dir):
        for filename in os.listdir(healthy_dir):
            if filename.endswith('.wav'):
                try:
                    file_path = os.path.join(healthy_dir, filename)
                    y, sr = librosa.load(file_path)
                    features = extract_advanced_features(y, sr)
                    features['filename'] = filename
                    features['label'] = 'healthy'
                    all_features.append(features)
                    
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
    pd_dir = "data/PD_AH/"
    if os.path.exists(pd_dir):
        for filename in os.listdir(pd_dir):
            if filename.endswith('.wav'):
                try:
                    file_path = os.path.join(pd_dir, filename)
                    y, sr = librosa.load(file_path)
                    
                    features = extract_advanced_features(y, sr)
                    
                    features['filename'] = filename
                    features['label'] = 'parkinsons'
                    
                    all_features.append(features)
                    
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
    
    if not all_features:
        print("No features were successfully extracted from any files. Please check your data directories and feature extraction code.")
        return

    all_features = pd.DataFrame(all_features)
    healthy_count = len(all_features[all_features['label'] == 'healthy'])
    pd_count = len(all_features[all_features['label'] == 'parkinsons'])
    print(f"Processed {healthy_count} healthy samples and {pd_count} Parkinson's samples")
    
    if healthy_count == 0 or pd_count == 0:
        print("Not enough data for both classes. Please check your data directories.")
        return
    
    print(f"Dataset shape: {all_features.shape}")
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    all_features.to_csv('models/extracted_features.csv', index=False)
    X = all_features.drop(['filename', 'label'], axis=1)
    y = all_features['label']
    feature_names = X.columns.tolist()
    with open('models/feature_names.txt', 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    
    print("\nOptimizing model hyperparameters...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Parkinson\'s'],
                yticklabels=['Healthy', 'Parkinson\'s'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrix.png')
    if hasattr(best_model['classifier'], 'feature_importances_'):
        importances = best_model['classifier'].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        feature_names = X.columns
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance for Parkinson\'s Disease Detection')
        plt.bar(range(min(20, len(indices))), importances[indices[:20]], align='center')
        plt.xticks(range(min(20, len(indices))), [feature_names[i] for i in indices[:20]], rotation=90)
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance.png')
        
        print("\nTop 10 Most Important Features:")
        for i in range(min(10, len(indices))):
            print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    joblib.dump(best_model, 'models/parkinson_model.pkl')
    print("\nModel training complete. Model saved to models/parkinson_model.pkl")

def create_example_visualizations():
    """Create example visualizations for one healthy and one PD sample"""
    
    healthy_dir = "data/HC_AH/"
    pd_dir = "data/PD_AH/"
    
    if os.path.exists(healthy_dir) and os.path.exists(pd_dir):
        healthy_file = None
        for filename in os.listdir(healthy_dir):
            if filename.endswith('.wav'):
                healthy_file = os.path.join(healthy_dir, filename)
                break
                
        pd_file = None
        for filename in os.listdir(pd_dir):
            if filename.endswith('.wav'):
                pd_file = os.path.join(pd_dir, filename)
                break
        
        if healthy_file and pd_file:
            os.makedirs('visualizations', exist_ok=True)
            
            y_healthy, sr = librosa.load(healthy_file)
            plt.figure(figsize=(12, 4))
            plt.plot(np.linspace(0, len(y_healthy)/sr, len(y_healthy)), y_healthy)
            plt.title('Healthy Control - Audio Waveform')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.tight_layout()
            plt.savefig('visualizations/healthy_waveform.png')
            plt.close()
            
            plt.figure(figsize=(12, 6))
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y_healthy)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Healthy Control - Spectrogram')
            plt.tight_layout()
            plt.savefig('visualizations/healthy_spectrogram.png')
            plt.close()
            y_pd, sr = librosa.load(pd_file)
            plt.figure(figsize=(12, 4))
            plt.plot(np.linspace(0, len(y_pd)/sr, len(y_pd)), y_pd)
            plt.title('Parkinson\'s Disease - Audio Waveform')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.tight_layout()
            plt.savefig('visualizations/pd_waveform.png')
            plt.close()
            plt.figure(figsize=(12, 6))
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y_pd)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Parkinson\'s Disease - Spectrogram')
            plt.tight_layout()
            plt.savefig('visualizations/pd_spectrogram.png')
            plt.close()
            
            print("Example visualizations created in the visualizations directory.")

if __name__ == "__main__":
    train_optimized_model()
    create_example_visualizations()
