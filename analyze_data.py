import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
def analyze_features():
    """Analyze features to understand key differences between healthy and Parkinson's voices"""

    if not os.path.exists('models/extracted_features.csv'):
        print("Error: Feature data not found. Please run the training script first.")
        return
    os.makedirs('visualizations', exist_ok=True)
    df = pd.DataFrame(pd.read_csv('models/extracted_features.csv'))
    print(f"Loaded feature data with {df.shape[0]} samples and {df.shape[1]} features")
    print("Generating feature distribution visualizations...")
    key_features = [
        'jitter_relative', 'jitter_absolute', 'jitter_ppq5',
        'shimmer_relative', 'shimmer_absolute',
        'hnr', 'f0_mean', 'f0_std'
    ]
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(key_features):
        if feature in df.columns:
            plt.subplot(2, 4, i+1)
            sns.boxplot(x='label', y=feature, data=df)
            plt.title(f'{feature} by Group')
            plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/key_features_boxplot.png')
    plt.close()
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_heatmap.png')
    plt.close()

if _name_ == "__main__":
    analyze_features()
