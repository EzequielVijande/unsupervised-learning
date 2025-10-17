from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

_COLORS = None

def _get_colors(num_colors):
    global _COLORS
    if _COLORS is None:
        cmap = plt.get_cmap("Set2")
        _COLORS = [cmap(i) for i in range(num_colors)]
    return _COLORS

def load_and_process_dataset(path, plot_boxplots=False):
    df = pd.read_csv(DATASET_PATH)
    df.info()
    country_factors = df.drop('Country', axis=1).to_numpy(dtype=np.float64)
    scaled_factors = (country_factors - country_factors.mean(axis=0))/country_factors.std(axis=0)
    x_ticks = df.drop('Country', axis=1).keys().to_list()
    colors = _get_colors(len(x_ticks))
    if plot_boxplots:
        plt.figure(figsize=(16,12))
        plt.title('Country variables before standarization', fontsize=20)
        bp = plt.boxplot(country_factors, tick_labels=x_ticks, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        plt.xticks(rotation=0, fontsize=15)
        plt.grid(True)
        plt.show()
        #Standarized factors
        plt.figure(figsize=(16,12))
        plt.title('Country variables after standarization', fontsize=20)
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
        ax.annotate(country, (pc1_scores[i], pc2_scores[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8)
    colors = _get_colors(len(dst['Features']))
    loadings = pca_model.components_.T * np.sqrt(pca_model.explained_variance_)

    for i, feature in enumerate(dst['Features']):
        arrow_color = colors[i]
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                 head_width=0.05, head_length=0.05,
                 fc=arrow_color, ec=arrow_color, alpha=0.6, linewidth=1.5)
        ax.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15,
                feature, color=arrow_color, fontsize=10,
                ha='center', va='center', weight='bold')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax.set_title('PCA Biplot: Countries and Feature Loadings', fontsize=14, weight='bold')
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

    print(f'Components shape = {pca1_contrib.shape}')
    print(f'Components = {pca1_contrib}')
    cov_mat = pca.get_covariance()
    # Weight of each feature in the component

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