"""
Matrix addition is a common operation in many machine learning problems. The function `matrix_sum` below takes two matrices as input and returns their sum.
Provide the implementation of `matrix_sum`.
"""

import numpy as np

a = np.array([
    [1, 4],
    [5, 6]
])
b = np.array([
    [8, 9],
    [10, 11]
])


def matrix_sum(a, b):
    assert a.shape == b.shape

    c = np.zeros(a.shape)

    for i in range(len(a)):
        c[i, 0] = a[i, 0] + b[i, 0]
        c[i, 1] = a[i, 1] + b[i, 1]

    return c


print(matrix_sum(a, b).tolist())
