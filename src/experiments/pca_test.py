from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def load_and_process_dataset(path, plot_boxplots=False):
    df = pd.read_csv(DATASET_PATH)
    df.info()
    country_factors = df.drop('Country', axis=1).to_numpy(dtype=np.float64)
    scaled_factors = (country_factors - country_factors.mean(axis=0))/country_factors.std(axis=0)
    x_ticks = df.drop('Country', axis=1).keys().to_list()
    if plot_boxplots:
        plt.figure(figsize=(16,12))
        plt.title('Country variables before standarization')
        plt.boxplot(country_factors, tick_labels=x_ticks)
        plt.xticks(rotation=25)
        plt.grid(True)
        plt.show()
        #Standarized factors
        plt.figure(figsize=(16,12))
        plt.title('Country variables after standarization')
        plt.boxplot(scaled_factors, tick_labels=x_ticks)
        plt.xticks(rotation=25)
        plt.grid(True)
        plt.show()
    return {'Country':df['Country'].to_list(), 'Features':x_ticks, 'Data':scaled_factors}


DATASET_PATH = './datasets/europe.csv'

def main():
    dst = load_and_process_dataset(DATASET_PATH)
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