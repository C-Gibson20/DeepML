import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)
	min = np.min(data, axis=0)
	max = np.max(data, axis=0)
	
	standardized_data = np.round(((data - mean) / std), 4).tolist()
	normalized_data = np.round(((data - min)/(max - min)), 4).tolist()
	return standardized_data, normalized_data
