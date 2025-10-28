import numpy as np


class OjaNetwork:
    def __init__(self, n_features, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        rng = np.random.default_rng(seed)
        self.n_features = n_features
        self.weights = rng.random(n_features) - 0.5
    
    def train(self, data, epochs=100, learning_rate=0.01):
        data = np.array(data)
        
        for epoch in range(epochs):
            for sample in data:
                # Calcular la salida lineal
                output = np.dot(self.weights, sample)
                
                # Calcular el cambio en los pesos usando la Regla de Oja:
                # Δw = η(O·x - O²·w)
                delta_w = learning_rate * (output * sample - (output ** 2) * self.weights)
                
                # Actualizar los pesos
                self.weights += delta_w
    
    def get_weights(self):
        norm = np.linalg.norm(self.weights)
        if norm == 0:
            return self.weights
        return self.weights / norm