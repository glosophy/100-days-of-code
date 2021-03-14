"""
Given a matrix, write a function to compute the average of the entries in the matrix.
"""

import numpy as np

MAT = [[1.27411064, 0.05188032, -1.27088046],
       [-0.78844599, -0.14775522, -0.28198009]]


def compute_average(mat):
    """
    Compute the average of the entries in the provided matrix
    :param mat: Matrix represented as a numpy array
    :return: The average of the entries
    """
    mean = np.average(mat)

    return mean


compute_average(MAT)
