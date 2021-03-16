"""
Given two matrices of appropriate dimensions, create a function that computes their matrix product.
"""
import numpy as np

MAT_1 = [[0.6583596987271446, 1.0128241391924433],
         [0.37783705753739877, 0.42421340135829255],
         [-0.6905233695318467, -0.498554227530507]]
MAT_2 = [[0.5045660075993441], [1.83029285141006]]


def multiply_mats(mat1, mat2):
    """
    Compute the matrix product of the given matrices (mat1 * mat2).
    :param mat1: Left matrix of multiplication represented as nested array
    :param mat2: Right matrix of multiplication represented as nested array
    :return: The matrix product
    """

    multiply = np.dot(mat1, mat2)

    return multiply


multiply_mats(np.array(MAT_1), np.array(MAT_2)).tolist()
