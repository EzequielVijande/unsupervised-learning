import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
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

def plot_pca_vs_oja_comparison(features, pca_components, oja_weights):
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = _get_colors(len(features))
    x_positions = np.arange(len(features))
    # sklearn PCA components as dots
    for i, (feature, pca_val) in enumerate(zip(features, pca_components)):
        ax.scatter(i, pca_val, marker='s', s=200, color=colors[i],
                   label=f'{feature} (PCA)', edgecolors='black', linewidths=1.5, alpha=0.8, zorder=3)
    # oja weights as crosses
    for i, (feature, oja_val) in enumerate(zip(features, oja_weights)):
        ax.scatter(i, oja_val, marker='.', s=200, color=colors[i],
                   linewidths=3, alpha=0.9, zorder=4, edgecolors='black')
    # Add connecting lines between PCA and Oja for each feature
    for i in range(len(features)):
        ax.plot([i, i], [pca_components[i], oja_weights[i]],
                color=colors[i], linestyle='--', alpha=0.4, linewidth=1)

    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.3)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(features, ha='center', fontsize=14)
    ax.set_ylabel('Weight / Component Value', fontsize=14, fontweight='bold')
    ax.set_xlabel('Features', fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')


    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=10, label='sklearn PCA', markeredgecolor='black', markeredgewidth=1.5),
        Line2D([0], [0], marker='x', color='gray', markersize=10,
               label='Oja Network', linewidth=3, linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=11, framealpha=0.9)

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
    if np.dot(oja.get_weights(), pca.components_[0]) < 0:
        oja.weights *= -1
    oja.train(dst['Data'], pca.components_[0], epochs=1000, learning_rate=0.01, verbose = True, norm_each_epoch = True)
    oja_weights = oja.get_weights()
    oja_var_ratio = oja.explained_variance_ratio(dst["Data"])
    
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

    plt.rcParams.update({'font.size': 14})
    #plot convergence visualization
    plt.figure(figsize=(10,6))
    plt.plot(oja.angle_history)
    plt.xlabel("Epoch")
    plt.ylabel("Angle with PCA (°)")
    plt.title("Oja convergence toward first principal component")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Check if they are similar (considering sign ambiguity)
    similarity = np.abs(np.dot(pca1_contrib, oja_weights))
    print(f'Cosine similarity (absolute): {similarity:.6f}')

    plot_pca_vs_oja_comparison(dst['Features'], pca1_contrib, oja_weights)

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

    #oja vs pca country scores:
    w_pca = pca.components_[0]
    scores_pca = np.dot(dst['Data'], w_pca)
    scores_oja = np.dot(dst['Data'], oja_weights)

    if np.dot(scores_pca, scores_oja) < 0:
        scores_oja *= -1

    #dataframe
    scores_df = pd.DataFrame({
        "Country": dst["Country"],
        "PCA": scores_pca,
        "Oja": scores_oja
    })

    # Scatter (PCA vs Oja scores)
    plt.figure(figsize=(8, 6))

    # Scatter points: actual PCA vs Oja scores per country
    plt.scatter(scores_pca, scores_oja, color="royalblue", alpha=0.7, label="Country scores")

    # Red dashed line: perfect 1:1 relationship
    mn, mx = scores_pca.min(), scores_pca.max()
    plt.plot([mn, mx], [mn, mx],
             "r--", linewidth=2, label="Perfect match (PCA = Oja)")
    mn, mx = scores_pca.min(), scores_pca.max()
    #plt.plot([mn, mx], [mn, mx], "r--", label="y = x")
    plt.xlabel("PCA country scores (PC1)")
    plt.ylabel("Oja country scores (PC1)")
    plt.title("Country projections: PCA vs Oja (PC1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Bars by country (sorted)
    scores_df_sorted = scores_df.sort_values("PCA", ascending=True)
    plt.figure(figsize=(12, 6))
    plt.bar(scores_df_sorted["Country"], scores_df_sorted["PCA"], alpha=0.6, label="PCA")
    plt.bar(scores_df_sorted["Country"], scores_df_sorted["Oja"], alpha=0.6, label="Oja")
    plt.xticks(rotation=90)
    plt.ylabel("Component score (PC1)")
    plt.title("Country scores on PC1: PCA vs Oja")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # correlation
    corr = np.corrcoef(scores_pca, scores_oja)[0, 1]
    var_oja_pca = corr ** 2
    print(f"\nCorrelation between PCA and Oja country projections (PC1): {corr:.4f}")
    print(f"\nVariance between PCA and Oja country projections (PC1): {var_oja_pca:.4f}")
    #explained variance
    print("\n=== Explained Variance Comparison ===")
    pca_ratio = pca.explained_variance_ratio_[0]
    oja_ratio = oja.explained_variance_ratio(dst["Data"])
    print(f"PCA explained variance ratio (PC1): {pca_ratio:.4f}")
    print(f"Oja explained variance ratio (PC1): {oja_ratio:.4f}")
    print(f"Difference: {abs(pca_ratio - oja_ratio):.6f}")

    #weight evolution plot:
    W = np.array(oja.history)  # shape: (epochs, n_features)
    var_over_time = [np.var(np.dot(dst['Data'], w)) for w in W]
    total_var = np.var(dst['Data'], axis=0).sum()
    ratio_over_time = (np.array(var_over_time) / total_var)

    plt.figure(figsize=(10, 6))
    plt.plot(ratio_over_time, color="royalblue")
    plt.xlabel("Epoch")
    plt.ylabel("Explained variance ratio")
    plt.title("Variance captured by Oja weights over training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()  # This calls main() only when file is run directly