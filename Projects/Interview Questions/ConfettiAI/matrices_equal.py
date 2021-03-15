"""
Given two matrices, create a function that checks whether they are equal (i.e. all their corresponding entries are the same).
"""

import numpy as np

MAT_1 = [[0.6583596987271446, 1.0128241391924433],
         [0.37783705753739877, 0.42421340135829255],
         [-0.6905233695318467, -0.498554227530507]]
MAT_2 = [[0.6583596987271446, 1.0128241391924433],
         [0.37883705753739877, 0.42421340135829255],
         [-0.6905233695318467, -0.498554227530507]]


def are_equal(mat1, mat2):
    """
    Check that the provided matrices are equal, up to a tolerance.
    :param mat1: Matrix represented as numpy array of floats
    :param mat2: Matrix represented as numpy array of floats
    :return: Boolean indicating whether the provided matrices are equal
    """

    return np.allclose(mat1, mat2)


are_equal(np.array(MAT_1), np.array(MAT_2))
