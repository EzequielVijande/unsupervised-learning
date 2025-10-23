import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Literal, Optional, Tuple
from src.utils.feature_scaling import zscore


def som_size_heuristic(n_samples: int) -> Tuple[int, int]:
    """
    Heurística común para el tamaño de mapa (KxK) en función de N muestras.
    """
    target = int(5 * math.sqrt(n_samples))
    rows = int(math.sqrt(target))
    cols = int(math.ceil(target / max(1, rows)))
    return max(5, rows), max(5, cols) # Asegura un tamaño mínimo de grilla

def square_euclidean_distance(a: NDArray[np.floating], b: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.sum((a - b)**2, axis=2)

# Self-Organizing Map (Kohonen)
class SOM:

    def __init__(self, rows: int, cols: int, dim: int, lr: float = 0.5, sigma: Optional[float] = None, seed: int = 0,
                 init_weights_type: Literal['random', 'sample'] = 'random',  # 'random' (N(0,1)) o 'sample' (muestras de X)
                 bmu_metric: Literal['euclidean', 'exponential'] = 'euclidean',  # 'euclidean' (distancia euclídea) o 'exponential'
                 lr_decay_type: Literal['exponential', 'linear', 'inverse', 'constant'] = 'exponential',  # 'exponential', 'linear', 'inverse' (1/i), o 'constant'
                 sigma_decay_type: Literal['exponential', 'constant'] = 'exponential',  # 'exponential' o 'constant'
                 neighborhood_type: Literal['gaussian', 'hard'] = 'gaussian') -> None:   # 'gaussian' (soft) o 'hard' (regla estricta)
        self.rows, self.cols, self.dim = rows, cols, dim
        self.initial_lr = float(lr)
        self.initial_sigma = float(sigma if sigma is not None else max(rows, cols) / 2.0)
        self.seed = seed
        self.init_weights_type = init_weights_type
        self.bmu_metric = bmu_metric
        self.lr_decay_type = lr_decay_type
        self.sigma_decay_type = sigma_decay_type
        self.neighborhood_type = neighborhood_type
        self.weights: Optional[NDArray[np.floating]] = None
        # neuron coords grid; (rows, cols, 2)
        row_indices, col_indices = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        self.grid: NDArray[np.floating] = np.stack([row_indices, col_indices], axis=-1).astype(float)

    def _initialize_weights(self, data: NDArray[np.floating]) -> None:
        rng = np.random.default_rng(self.seed)
        if self.init_weights_type == 'random': # N(0,1) scaled
            self.weights = rng.normal(0, 1, size=(self.rows, self.cols, self.dim)) / np.sqrt(self.dim)
        else:
            # 'sample' initialization from data to avoid 'dead neurons'
            n_samples = data.shape[0]
            sample_indices = rng.choice(n_samples, size=self.rows * self.cols, replace=True)
            flat_weights = data[sample_indices].copy()
            self.weights = flat_weights.reshape(self.rows, self.cols, self.dim)

    def _decays(self, it: int, iterations: int) -> Tuple[float, float]:
        """ Compute learning rate (lr) and neighborhood radius (sigma) at iteration 'it'."""
        lr = self.initial_lr * {
                'exponential': np.exp(-it / iterations),
                'linear': (1 - it / iterations), # η(i) = 1/i
                'inverse': 1 / max(1, it + 1),
                'constant': 1
        }[self.lr_decay_type]

        if self.sigma_decay_type == 'exponential':
            sigma = self.initial_sigma * np.exp(-it / iterations)
        else: # 'constant' (R(i) constant)
            sigma = self.initial_sigma

        return float(lr), float(sigma)

    def _bmu_index(self, sample: NDArray[np.floating]) -> Tuple[Tuple[int, int], float]:
        """ Best Matching Unit (BMU):
         finds the neuron whose weight vector is closest to 'sample'.
        """
        assert self.weights is not None, "Weights must be initialized before calling _bmu_index"
        squared_dist = square_euclidean_distance(self.weights, sample) # (rows, cols)

        if self.bmu_metric == 'euclidean':
            # Wk̂ = arg min {∥Xp − Wj ∥}
            metric_to_minimize = squared_dist
        else:
            # Wk̂ = arg min {e^(-||Xp − Wj||²)} (exponential)
            metric_to_minimize = np.exp(-squared_dist)

        bmu_idx_flat = int(np.argmin(metric_to_minimize))
        bmu_idx = np.unravel_index(bmu_idx_flat, (self.rows, self.cols))

        # returning squared distance for QE and dists metrics
        return (int(bmu_idx[0]), int(bmu_idx[1])), float(np.sqrt(squared_dist[bmu_idx]))

    def _compute_influence(self, bmu_pos: NDArray[np.floating], sigma: float) -> NDArray[np.floating]:
        grid_dist_squared = np.sum((self.grid - bmu_pos)**2, axis=2)
        sigma_squared = max(1e-9, sigma * sigma)

        if self.neighborhood_type == 'gaussian':
            return np.exp(-grid_dist_squared / (2.0 * sigma_squared))
        elif self.neighborhood_type == 'hard':
            return np.where(grid_dist_squared <= sigma_squared, 1.0, 0.0)
        else:
            raise ValueError(f"Neighborhood type '{self.neighborhood_type}' not supported")

    def _update_weights(self, sample: NDArray[np.floating], bmu_pos: NDArray[np.floating], lr: float, sigma: float) -> None:
        assert self.weights is not None, "Weights must be initialized before updating"
        influence = self._compute_influence(bmu_pos, sigma)
        self.weights += lr * influence[..., None] * (sample - self.weights)

    def train(self, data: NDArray[np.floating], epochs: int = 100, shuffle: bool = True, verbose: bool = False) -> None:
        self._initialize_weights(data)
        n_samples = data.shape[0]
        total_iterations = max(1, epochs)

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples) if shuffle else np.arange(n_samples)
            lr, sigma = self._decays(epoch, total_iterations)

            for sample_idx in indices:
                sample = data[sample_idx]
                (bmu_row, bmu_col), _ = self._bmu_index(sample)
                bmu_pos = np.array([bmu_row, bmu_col], dtype=float)
                self._update_weights(sample, bmu_pos, lr, sigma)

            if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
                qe = self.quantization_error(data)
                te = self.topological_error(data)
                print(f"Epoch {epoch+1:3d}/{epochs} | QE={qe:.4f} | TE={te:.4f} | LR={lr:.4f} | Sigma={sigma:.4f}")

    # --------------------------
    # Metrics and plots

    def bmu_for_each(self, data: NDArray[np.floating]) -> Tuple[NDArray[np.int_], NDArray[np.floating]]:
        """
        Retorna índices (i,j) del BMU para cada sample en data, y la distancia BMU.
        """
        n_samples = data.shape[0]
        bmu_coords = np.empty((n_samples, 2), dtype=int)
        bmu_dists = np.empty(n_samples, dtype=float)

        # Utiliza la implementación de BMU con la métrica configurada
        for idx in range(n_samples):
            (row, col), dist = self._bmu_index(data[idx])
            bmu_coords[idx] = (row, col)
            bmu_dists[idx] = dist

        return bmu_coords, bmu_dists

    def hits(self, data: NDArray[np.floating]) -> NDArray[np.int_]:
        """
        Conteo de asignaciones por neurona (heatmap de hits).
        """
        bmu_coords, _ = self.bmu_for_each(data)
        hit_map = np.zeros((self.rows, self.cols), dtype=int)
        for row, col in bmu_coords:
            hit_map[row, col] += 1
        return hit_map

    def u_matrix(self, neighbor_type: Literal[4, 8] = 4) -> NDArray[np.floating]:
        """
        U-Matrix: distancia promedio a neuronas vecinas. Permite 4 u 8 vecinos.
        """
        assert self.weights is not None, "Weights must be initialized before computing u-matrix"
        umatrix = np.zeros((self.rows, self.cols), dtype=float)

        if neighbor_type == 4:
            # 4-vecinos (arriba, abajo, izquierda, derecha)
            directions = [(-1,0),(1,0),(0,-1),(0,1)]
        elif neighbor_type == 8:
            # 8-vecinos (incluye diagonales)
            directions = [(-1,0),(1,0),(0,-1),(0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        else:
            raise ValueError("neighbor_type debe ser 4 u 8.")

        for row in range(self.rows):
            for col in range(self.cols):
                neighbor_dists = []
                for delta_row, delta_col in directions:
                    neighbor_row, neighbor_col = row + delta_row, col + delta_col
                    if 0 <= neighbor_row < self.rows and 0 <= neighbor_col < self.cols:
                        # Distancia Euclídea entre el peso de la neurona central y el vecino
                        neighbor_dists.append(float(np.linalg.norm(self.weights[row, col] - self.weights[neighbor_row, neighbor_col])))

                umatrix[row, col] = np.mean(neighbor_dists) if neighbor_dists else 0.0 # Promedio de la distancia
        return umatrix

    def quantization_error(self, data: NDArray[np.floating]) -> float:
        """
         Quantization Error (QE): average Euclidean distance to the BMU.
        """
        _, bmu_dists = self.bmu_for_each(data)
        return float(np.mean(bmu_dists))

    def topological_error(self, data: NDArray[np.floating]) -> float:
        """
         Topological Error (TE): measures how well the neighborhood of the data is preserved.
        """
        assert self.weights is not None, "Weights must be initialized before computing topological error"
        n_samples = data.shape[0]
        topological_errors = 0

        for idx in range(n_samples):
            sample = data[idx]

            # Calcular distancias a todas las neuronas
            squared_dist = np.sum((self.weights - sample)**2, axis=2)  # (rows, cols)

            if self.bmu_metric == 'euclidean':
                metric_values = squared_dist
            elif self.bmu_metric == 'exponential':
                metric_values = np.exp(-squared_dist)
            else:
                metric_values = squared_dist

            # Encontrar los dos mejores BMUs
            flat_metrics = metric_values.flatten()

            if self.bmu_metric == 'exponential':
                two_best_flat = np.argpartition(flat_metrics, -2)[-2:] # two max
                sorted_indices = two_best_flat[np.argsort(-flat_metrics[two_best_flat])]
            else:
                two_best_flat = np.argpartition(flat_metrics, 2)[:2] # two min
                sorted_indices = two_best_flat[np.argsort(flat_metrics[two_best_flat])]

            bmu1_idx = np.unravel_index(int(sorted_indices[0]), (self.rows, self.cols))
            bmu2_idx = np.unravel_index(int(sorted_indices[1]), (self.rows, self.cols))

            # Verificar si el segundo BMU es vecino directo del primer BMU (4-vecindad)
            row_diff = abs(bmu1_idx[0] - bmu2_idx[0])
            col_diff = abs(bmu1_idx[1] - bmu2_idx[1])

            # Vecinos directos: diferencia de 1 en una dimensión y 0 en la otra
            # (arriba, abajo, izquierda, derecha)
            is_neighbor = (row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1)

            if not is_neighbor:
                topological_errors += 1

        return float(topological_errors / n_samples)
# --------------------------
# Eg con europe.csv
if __name__ == "__main__":
    try:
        df = pd.read_csv("../../datasets/europe.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    features = ["Area","GDP","Inflation","Life.expect","Military","Pop.growth","Unemployment"]
    standardized_data, mean_vals, std_vals = zscore(df, features)
    map_rows, map_cols = som_size_heuristic(len(df))
    print(f"SOM: {map_rows} x {map_cols} neurons")

    som = SOM(map_rows, map_cols, dim=standardized_data.shape[1], lr=0.5, sigma=None, seed=42,
              init_weights_type='sample',
              bmu_metric='euclidean',
              lr_decay_type='inverse',
              sigma_decay_type='exponential',
              neighborhood_type='hard')

    som.train(standardized_data, epochs=100, shuffle=True, verbose=True)

    # (country -> neuron)
    bmu_coords, bmu_dists = som.bmu_for_each(standardized_data)
    assignments = pd.DataFrame({
        "Country": df["Country"],
        "BMU_i": bmu_coords[:,0],
        "BMU_j": bmu_coords[:,1],
        "BMU_dist": np.round(bmu_dists, 4)
    }).sort_values(["BMU_i","BMU_j","Country"]).reset_index(drop=True)

    print("\nAsignación de países a neuronas (primeros 15):")
    print(assignments.head(15).to_string(index=False))

    # Métricas finales
    final_qe = som.quantization_error(standardized_data)
    final_te = som.topological_error(standardized_data)
    print(f"  Quantization Error (QE): {final_qe:.4f}")
    print(f"  Topological Error (TE):  {final_te:.4f}")

    # 5) Heatmap de conteos (hits)
    hit_map = som.hits(standardized_data)

    # 6) U-Matrix (probando 8-vecinos, R5)
    umatrix = som.u_matrix(neighbor_type=8)

    # 7) Gráficos
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Gráfico 1: Hits (Asociaciones)
    img0 = axes[0].imshow(hit_map, origin="upper", cmap='viridis')
    axes[0].set_title("Asociaciones por neurona (hits)")
    axes[0].set_xlabel("j"); axes[0].set_ylabel("i")
    plt.colorbar(img0, ax=axes[0], fraction=0.046, pad=0.04)

    # Gráfico 2: U-Matrix (Distancias promedio)
    img1 = axes[1].imshow(umatrix, origin="upper", cmap='gray')
    axes[1].set_title("U-Matrix (distancia promedio a 8 vecinas)")
    axes[1].set_xlabel("j"); axes[1].set_ylabel("i")
    plt.colorbar(img1, ax=axes[1], fraction=0.046, pad=0.04)

    # Gráfico 3: Mapa de Etiquetas (Países asignados)
    axes[2].imshow(np.zeros_like(hit_map, dtype=float), origin="upper", cmap="Blues", vmin=0, vmax=1)
    axes[2].set_title("Países asignados (BMU)")
    axes[2].set_xlabel("j"); axes[2].set_ylabel("i")
    for country, (row, col) in zip(df["Country"], bmu_coords):
        axes[2].text(col, row, country, ha="center", va="center", fontsize=7)
    axes[2].set_xticks(range(map_cols)); axes[2].set_yticks(range(map_rows))

    plt.tight_layout()
    plt.show()
