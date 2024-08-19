def matrix_dot_vector(a:list[list[int|float]],b:list[int|float])-> list[int|float]:
    if len(a[0]) != len(b):
        return -1
      
    dot_product = [0 for row in a]
    for row_idx, row in enumerate(a):
      for column_idx, column in enumerate(row):
        dot_product[row_idx] += column * b[column_idx]

    return dot_product
