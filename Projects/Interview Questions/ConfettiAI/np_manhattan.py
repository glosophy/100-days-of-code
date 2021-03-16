"""
Given two vectors of data, compute the Manhattan distance between them.

HINT: Try to implement this only using `numpy` functions.
"""

import numpy as np

VEC_1 = [-0.25560104, 0.06393334, -0.43760861, 0.35258494, -0.06174621]
VEC_2 = [0.16257878, -0.88344182, 1.14405499, 0.33765161, 1.206262]


def manhattan_distance(vec1, vec2):
    """
    Compute the Manhattan distance between two vectors.
    :param vec1: Vector represented as numpy array of floats
    :param vec2: Vector represented as numpy array of floats
    :return: The Manhattan distance between the vectors
    """

    manhattan = np.sum(np.abs(vec2 - vec1))

    return manhattan


manhattan_distance(np.array(VEC_1), np.array(VEC_2))
