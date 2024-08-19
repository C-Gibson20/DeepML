def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    n_features = len(vectors)
    n_observations = len(vectors[0])
    covariance_matrix = [[0 for _ in range(n_features)] for _ in range(n_features)]
    
    feature_means = [sum(feature)/n_observations for feature in vectors]
    
    for i in range(n_features):
      for j in range(n_features):
        covariance = 0
        for k in range(n_observations):
        	covariance += (vectors[i][k] - feature_means[i]) * (vectors[j][k] - feature_means[j])
        covariance_matrix[i][j] = covariance_matrix[j][i] = (covariance / (n_observations - 1))
    
    return covariance_matrix
