import numpy as np

class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))
    
    def train(self, patterns):
        """
        Train the network using Hebbian learning.
        Patterns should be a 2D array where each row is a pattern of length n_neurons.
        Values in patterns must be -1 or 1.
        """
        patterns = np.asarray(patterns)
        
        n_patterns = len(patterns)
        
        # Initialize weights
        self.weights = np.dot(patterns.T, patterns) / n_patterns
        
        # Remove self-connections
        np.fill_diagonal(self.weights, 0)
    
    def recall(self, pattern, max_iter=100, async_update=True):
        """
        Recall a stored pattern from a noisy input.
        If async_update is True, update neurons asynchronously (recommended).
        """
        results = []
        pattern = np.array(pattern, dtype=np.float32)
        assert len(pattern) == self.n_neurons, "Input pattern length must match network size"
        
        current_pattern = pattern.copy()
        
        if async_update:
            for _ in range(max_iter):
                results.append(current_pattern)
                # Random neuron order for asynchronous update
                neurons_order = np.random.permutation(self.n_neurons)
                new_pattern = current_pattern.copy()
                for neuron in neurons_order:
                    activation = np.dot(self.weights[neuron], new_pattern)
                    new_pattern[neuron] = 1 if activation >= 0 else -1
                if np.array_equal(new_pattern, current_pattern):
                    break
                current_pattern = new_pattern
        else:
            # Synchronous update (less stable)
            for _ in range(max_iter):
                results.append(current_pattern)
                activation = np.dot(self.weights, current_pattern)
                new_pattern = np.where(activation >= 0, 1, -1)
                if np.array_equal(new_pattern, current_pattern):
                    break
                current_pattern = new_pattern
        
        results.append(current_pattern)  
        return results
    
    def compute_energy(self, pattern):
        """Compute the energy of a given pattern."""
        return -0.5 * np.dot(pattern, np.dot(self.weights, pattern))
    
    def get_weights(self):
        """Return the weight matrix."""
        return self.weights