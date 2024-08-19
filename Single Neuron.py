import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
  z = [((np.array(weights) @ np.array(feature)) + bias) for feature in features]
  probabilities = [round(1/(1+math.exp(-val)), 4) for val in z]
  e = [(probabilities[i]-labels[i])**2 for i in range(len(probabilities))]
  mse = round(sum(e)/len(features), 4)
  return probabilities, mse
