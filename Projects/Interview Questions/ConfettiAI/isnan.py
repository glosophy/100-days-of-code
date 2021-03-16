"""
Given a matrix, write a function that checks whether or not there are any NaN values
in the matrix. This is a common thing to check when building deep learning models, as
instable training procedures can generate NaNs in your weights.
"""
import numpy as np

MAT = [[0.6583596987271446, 1.0128241391924433],
       [0.37783705753739877, float("nan")],
       [-0.6905233695318467, -0.498554227530507]]


def contains_nan(mat):
    """
    Check whether there are any NaN values in the provided matrix.
    :param mat: Matrix of floats represented as numpy array
    :return: Boolean value indicating whether there are any NaNs
    """

    return np.isnan(mat).any()


contains_nan(np.array(MAT))