import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
    # elements = [i for list in a for i in list]
    # reshaped_matrix = [elements[i:i+new_shape[1]] for i in range(0, len(elements), new_shape[1])]
    # return reshaped_matrix
    return np.array(a).reshape(new_shape).tolist()
