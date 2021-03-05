"""
Given the input data of points [(1, 2), (3,4), (5,6), (8, 8), (7, -1)], fit a line of the form
Y = A * X + B. Return the values (A, B) as a tuple
"""

import numpy as np


def lin_interpolate(data):
    """
    :param data: List of points to linearly interpolate, each of which is provided as a tuple.
    :return: The optimal A and B of your fit line
    """
    # TODO: Fill in this function
    all_data = np.array(data)
    X = np.ones_like(all_data)
    X[:, 0] = all_data[:, 0]
    y = all_data[:, 1].tolist()

    coef = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return tuple(coef)


lin_interpolate([(1, 2), (3, 4), (5, 6), (8, 8), (7, -1)])
