import numpy as np


class OjaNetwork:
    def __init__(self, n_features, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        rng = np.random.default_rng(seed)
        self.n_features = n_features
        #initiatlizing random weights + normalize
        self.weights = rng.normal(size = n_features)
        self.weights /= np.linalg.norm(self.weights)
        self.history = []
        self.angle_history = []


    def train(self, data, comp_vec, epochs=100, learning_rate=0.01, tol = 1e-6, norm_each_epoch=True, verbose = False):
        X = np.asarray(data, dtype = float)
        w = self.weights.copy()
        
        for epoch in range(epochs):
            w_prev = w.copy()
            for x in X:
                y = np.dot(w, x)
                w += learning_rate *y *(x-y*w)
            # if norm_each_epoch:
            #    w /= np.linalg.norm(w) + 1e-12

            # Track convergence
            delta = np.linalg.norm(w -w_prev)
            self.history.append(w.copy())
            if comp_vec is not None:
                cosang = np.clip(np.dot(w, comp_vec) /
                        (np.linalg.norm(w)* np.linalg.norm(comp_vec) +1e-12), -1, 1)
                angle_deg = np.degrees(np.arccos(cosang))
                self.angle_history.append(angle_deg)

            if verbose:
                msg = f" epoch {epochs: 03d}, delta w = {delta:.2e}"
                if self.angle_history:
                    msg += f"angle = {self.angle_history[-1]:.4f}°"
                print(msg)

            if delta < tol:
                if verbose:
                    print(f"Converged at epoch {epoch}")
                break

        self.weights = w / np.linalg.norm(w)
                    # for sample in data:
            #     # Calcular la salida lineal
            #     output = np.dot(self.weights, sample)
            #
            #     # Calcular el cambio en los pesos usando la Regla de Oja:
            #     # Δw = η(O·x - O²·w)
            #
            #
            #     # Actualizar los pesos
            #     self.weights += delta_w
    
    def get_weights(self):
        norm = np.linalg.norm(self.weights)
        if norm == 0:
            return self.weights
        return self.weights / norm

    def explained_variance_ratio(self, X):
        # Projection of data on learned component
        z = np.dot(X, self.get_weights())
        # Variance captured along this direction
        var_captured = np.var(z, ddof=0)
        # Total variance in all features
        total_var = np.var(X, axis=0, ddof=0).sum()
        return var_captured / total_var