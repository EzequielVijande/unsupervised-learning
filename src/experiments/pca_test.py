import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from src.networks.oja import OjaNetwork
from src.utils.feature_scaling import zscore

_COLORS = None

def _get_colors(num_colors):
    global _COLORS
    if _COLORS is None:
        cmap = plt.get_cmap("Set2")
        _COLORS = [cmap(i) for i in range(num_colors)]
    return _COLORS

def _feature_label_offset(feature, x_loading, y_loading):
    factors = {
        'Life.expect': (1.52, 1),
        'Pop.growth': (1.52, 1),
        'GDP': (1.25, 1.1),
        'Inflation': (1.1, 1.3),
        'Area': (1, 1.6),
        'Military': (1, 1.2),
        'Unemployment': (1.8, 1.2)
    }
    x_factor, y_factor = factors.get(feature, (1.1, 1.1))
    return x_loading * x_factor, y_loading * y_factor

def _country_label_offset(country):
    factors = {
        'Luxembourg': (-35, 10),
        'Switzerland': (-25, 10),
        'Norway': (10, 0),
        'Lithuania': (7, -5),
        'Hungary': (7, -3),
        'Poland': (7, 0),
        'Czech Republic': (-25, -15),
        'Finland': (7, -3),
        'Germany': (-25, -15),
        'Denmark': (8, -6),
        'Belgium': (-50, -1),
        'Sweden': (10, 5)
    }
    x_pos, y_pos = factors.get(country, (5, 5))
    return x_pos, y_pos


def load_and_process_dataset(path, plot_boxplots=False):
    df = pd.read_csv(DATASET_PATH)
    df.info()
    
    feature_cols = df.drop('Country', axis=1).columns.tolist()
    
    # 1. Get unscaled data first for the "before" plot
    country_factors = df[feature_cols].to_numpy(dtype=np.float64)
    
    # 2. Get scaled data using zscore
    scaled_factors, _, _ = zscore(df, feature_cols)
    
    x_ticks = feature_cols
    colors = _get_colors(len(x_ticks))
    
    if plot_boxplots:
        plt.figure(figsize=(16,12))
        plt.title('Country variables before standarization', fontsize=20)
        # Use unscaled 'country_factors' here
        bp = plt.boxplot(country_factors, tick_labels=x_ticks, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        plt.xticks(rotation=0, fontsize=15)
        plt.grid(True)
        plt.show()
        
        #Standarized factors
        plt.figure(figsize=(16,12))
        plt.title('Country variables after standarization', fontsize=20)
        # Use scaled 'scaled_factors' here
        bp = plt.boxplot(scaled_factors, tick_labels=x_ticks, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        plt.xticks(rotation=0, fontsize=15)
        plt.grid(True)
        plt.show()
        
    return {'Country':df['Country'].to_list(), 'Features':x_ticks, 'Data':scaled_factors}

def plot_biplot(dst, pca_model, pc1_scores, pc2_scores):
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.scatterplot(x=pc1_scores, y=pc2_scores, s=100, alpha=0.7, ax=ax)

    for i, country in enumerate(dst['Country']):
        country_label_x, country_label_y = _country_label_offset(country)
        ax.annotate(country, (pc1_scores[i], pc2_scores[i]),
                    xytext=(country_label_x, country_label_y), textcoords='offset points',
                    fontsize=11, alpha=0.8)
    colors = _get_colors(len(dst['Features']))
    loadings = pca_model.components_.T * np.sqrt(pca_model.explained_variance_)

    for i, feature in enumerate(dst['Features']):
        arrow_color = colors[i]
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                 head_width=0.05, head_length=0.05,
                 fc=arrow_color, ec=arrow_color, alpha=0.6, linewidth=2.5)
        feature_label_x, feature_label_y = _feature_label_offset(feature, loadings[i, 0], loadings[i, 1])
        ax.text(feature_label_x, feature_label_y,
                feature, color=arrow_color, fontsize=12,
                ha='center', va='center', weight='bold')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

DATASET_PATH = './datasets/europe.csv'

def main():
    dst = load_and_process_dataset(DATASET_PATH)

    pca_2 = PCA(2)
    pca_scores = pca_2.fit_transform(dst['Data'])
    plot_biplot(dst, pca_2, pca_scores[:, 0], pca_scores[:, 1])

    pca = PCA(1)
    pca.fit(dst['Data'])
    pca1 = pca.fit_transform(dst['Data']).flatten()
    # Get the components (loadings)
    pca1_contrib = pca.components_[0].flatten()

    print(f'\nPCA Results (sklearn):')
    print(f'Components shape = {pca1_contrib.shape}')
    print(f'Components = {pca1_contrib}')
    
    # Train Oja Network
    print(f'\nTraining Oja Network...')
    oja = OjaNetwork(n_features=dst['Data'].shape[1], seed=42)
    oja.train(dst['Data'], epochs=1000, learning_rate=0.01)
    oja_weights = oja.get_weights()
    
    print(f'\nOja Network Results:')
    print(f'Weights shape = {oja_weights.shape}')
    print(f'Weights = {oja_weights}')
    
    # Compare results
    print(f'\n=== COMPARISON: sklearn PCA vs Oja Network ===')
    print(f'Feature names: {dst["Features"]}')
    print(f'PCA components:  {pca1_contrib}')
    print(f'Oja weights:     {oja_weights}')
    print(f'Absolute difference: {np.abs(pca1_contrib - oja_weights)}')
    print(f'Mean absolute difference: {np.mean(np.abs(pca1_contrib - oja_weights)):.6f}')
    
    # Check if they are similar (considering sign ambiguity)
    similarity = np.abs(np.dot(pca1_contrib, oja_weights))
    print(f'Cosine similarity (absolute): {similarity:.6f}')

    #Plot featrues contributions
    plt.figure(figsize=(18,12))
    plt.title('Weight of each feature in PCA1 component')
    plt.bar(range(len(pca1_contrib)), pca1_contrib)
    plt.xticks(range(len(pca1_contrib)), dst['Features'], rotation=35, ha='right')
    plt.ylabel('Contribution')
    plt.grid(True)
    plt.show()
    #Plot PCA1 components
    plt.figure(figsize=(18,12))
    plt.title('PCA1 component for each country')
    plt.bar(range(len(pca1)), pca1)
    plt.xticks(range(len(pca1)), dst['Country'], rotation=35, ha='right')
    plt.ylabel('PCA 1')
    plt.grid(True)
    plt.show()
    

if __name__ == "__main__":
    main()  # This calls main() only when file is run directly