import numpy as np
from numpy.typing import NDArray
from typing import Literal, Optional, Tuple

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
            sample_indices = rng.choice(n_samples, size=self.rows * self.cols, replace=False)
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

    def _find_two_best_bmus(self, metric_values: NDArray[np.floating]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """ Find indices of the first and second best BMUs."""
        flat_metrics = metric_values.flatten()
        if self.bmu_metric == 'exponential':
            two_best_flat = np.argpartition(flat_metrics, -2)[-2:]  # two max
            sorted_indices = two_best_flat[np.argsort(-flat_metrics[two_best_flat])]
        else:
            two_best_flat = np.argpartition(flat_metrics, 2)[:2]  # two min
            sorted_indices = two_best_flat[np.argsort(flat_metrics[two_best_flat])]

        bmu1_idx = np.unravel_index(int(sorted_indices[0]), (self.rows, self.cols))
        bmu2_idx = np.unravel_index(int(sorted_indices[1]), (self.rows, self.cols))

        return (int(bmu1_idx[0]), int(bmu1_idx[1])), (int(bmu2_idx[0]), int(bmu2_idx[1]))

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

    def train(self, data: NDArray[np.floating], epochs: int = 100, shuffle: bool = True) -> None:
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

    def bmu_coords_and_dists(self, data: NDArray[np.floating]) -> Tuple[NDArray[np.int_], NDArray[np.floating]]:
        """
         For each sample in 'data', returns the coordinates of its BMU and the distance to it.
        """
        n_samples = data.shape[0]
        bmu_coords = np.empty((n_samples, 2), dtype=int)
        bmu_dists = np.empty(n_samples, dtype=float)
        for idx in range(n_samples):
            (row, col), dist = self._bmu_index(data[idx])
            bmu_coords[idx] = (row, col)
            bmu_dists[idx] = dist

        return bmu_coords, bmu_dists

    def hits(self, data: NDArray[np.floating]) -> NDArray[np.int_]:
        """ Number of hits a neuron is BMU for the given data."""
        bmu_coords, _ = self.bmu_coords_and_dists(data)
        hit_map = np.zeros((self.rows, self.cols), dtype=int)
        for row, col in bmu_coords:
            hit_map[row, col] += 1
        return hit_map

    def u_matrix(self, neighbor_type: Literal[11, 12] = 4) -> NDArray[np.floating]:
        if neighbor_type == 4:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else: # neighbor_type == 8
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]

        umatrix = np.zeros((self.rows, self.cols), dtype=float)
        for row in range(self.rows):
            for col in range(self.cols):
                center_weight = self.weights[row, col]
                neighbor_dists = []
                for dr, dc in directions:
                    n_row, n_col = row + dr, col + dc
                    if 0 <= n_row < self.rows and 0 <= n_col < self.cols:
                        neighbor_weight = self.weights[n_row, n_col]
                        # euclídean distance
                        distance = np.linalg.norm(center_weight - neighbor_weight)
                        neighbor_dists.append(distance)

                umatrix[row, col] = np.mean(neighbor_dists) if neighbor_dists else 0.0

        return umatrix

    def quantization_error(self, data: NDArray[np.floating]) -> float:
        """
         Quantization Error (QE): average Euclidean distance to the BMU.
        """
        _, bmu_dists = self.bmu_coords_and_dists(data)
        return float(np.mean(bmu_dists))

    def topological_error(self, data: NDArray[np.floating]) -> float:
        """
         Topological Error (TE): measures how well the neighborhood of the data is preserved.
        """
        assert self.weights is not None, "Weights must be initialized before computing topological error"

        def is_neighbor(bmu1: Tuple[int, int], bmu2: Tuple[int, int]) -> bool:
            # check 4-neighborhood of bmu1 and bmu2
            row_diff = abs(bmu1[0] - bmu2[0])
            col_diff = abs(bmu1[1] - bmu2[1])
            return (row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1)

        n_samples = data.shape[0]
        topological_errors = 0
        for idx in range(n_samples):
            sample = data[idx]
            squared_dist = square_euclidean_distance(self.weights, sample)
            if self.bmu_metric == 'exponential':
                metric_values = np.exp(-squared_dist)
            else:
                metric_values = squared_dist

            bmu1_idx, bmu2_idx = self._find_two_best_bmus(metric_values)
            if not is_neighbor(bmu1_idx, bmu2_idx):
                topological_errors += 1

        return float(topological_errors / n_samples)
