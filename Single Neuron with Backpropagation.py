import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    mse_values = []
    updated_weights = initial_weights
    updated_bias = initial_bias
    
    for _ in range(epochs):
      predictions = sigmoid(np.dot(features, updated_weights) + updated_bias)
      e = predictions - labels
      mse_values.append(round(np.mean(e ** 2), 4))
      
      d_weight = np.dot(features.T, e * predictions * (1 - predictions)) / len(labels)
      d_bias = np.sum(e * predictions * (1 - predictions)) / len(labels)
      
      updated_weights -= np.round(learning_rate * d_weight, 4)
      updated_bias -= np.round(learning_rate * d_bias, 4)
      
    return updated_weights, updated_bias, mse_values
