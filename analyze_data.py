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
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    X = numeric_df.values
    y = df['label'].values
    X_std = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_std)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=[0 if label == 'healthy' else 1 for label in y], 
              cmap='coolwarm', alpha=0.8)
    plt.colorbar(scatter, label='Class (0=Healthy, 1=Parkinson\'s)')
    plt.title('t-SNE Visualization of Voice Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig('visualizations/tsne_visualization.png')
    plt.close()
    print("\nStatistical Analysis of Key Features:")
    for feature in key_features:
        if feature in df.columns:
            healthy_values = df[df['label'] == 'healthy'][feature]
            pd_values = df[df['label'] == 'parkinsons'][feature]
            print(f"\nFeature: {feature}")
            print(f"Healthy - Mean: {healthy_values.mean():.5f}, Std: {healthy_values.std():.5f}")
            print(f"Parkinson's - Mean: {pd_values.mean():.5f}, Std: {pd_values.std():.5f}")
            from scipy.stats import ttest_ind
            t_stat, p_value = ttest_ind(healthy_values, pd_values, equal_var=False)
            print(f"T-test p-value: {p_value:.5f} {'(Significant)' if p_value < 0.05 else ''}")
    print("\nAnalysis complete. Visualizations saved to visualizations/ directory.")
if __name__ == "__main__":
    analyze_features()
