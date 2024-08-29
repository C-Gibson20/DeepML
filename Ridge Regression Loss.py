import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	return np.mean((y_true - X @ w)**2) + alpha * np.sum(w**2)
