import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    # theta_j = theta_j - alpha * 1/m * sum(h_{\theta}(x)-y)*x_j
    
    m, n = X.shape  
    theta = np.zeros((n, 1))
    y = y.reshape(-1, 1)
    
    for _ in range(iterations):
      ypred = X @ theta # h_{\theta}(x)
      errors = ypred - y # h_{\theta}(x) - y
      updates = X.T @ errors / m # 1/m * sum(errors * x_j)
      theta -= alpha * updates
      
    return np.round(theta.flatten(), 4)
