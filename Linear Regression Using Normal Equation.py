import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    X_t = X.T
    
    theta = np.linalg.inv(X_t.dot(X)).dot(X_t).dot(y)
    return np.round(theta, 4).flatten().tolist()
