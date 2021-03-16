"""
Given a matrix, subtract the column mean from each entry. This has the effect of zero-centering
the data and is often done in algorithms such as principal components analysis or when running computer
vision models.
"""

import numpy as np

MAT = [[0.6583596987271446, 1.0128241391924433],
       [0.37783705753739877, 0.42421340135829255],
       [-0.6905233695318467, -0.498554227530507]]


def normalize_col_mean(mat):
    """
    Subtract the mean of each column from each entry.
    :param mat: Matrix of floats represented as numpy array
    :return: The transformed matrix with the column mean subtracted from each entry
    """

    mat = mat - np.mean(mat, axis=0)

    return mat


normalize_col_mean(np.array(MAT)).tolist()
