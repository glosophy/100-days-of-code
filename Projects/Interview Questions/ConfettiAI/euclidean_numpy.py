"""
Given two vectors of data, compute the Euclidean distance between them.
"""

import numpy as np

VEC_1 = [-0.25560104, 0.06393334, -0.43760861, 0.35258494, -0.06174621]
VEC_2 = [0.16257878, -0.88344182, 1.14405499, 0.33765161, 1.206262]


def euclidean_distance(vec1, vec2):
    """
    Compute the Euclidean distance between two vectors.
    :param vec1: Vector represented as a numpy array
    :param vec2: Vector represented as a numpy array
    :return: The Euclidean distance between the vectors
    """
    euclidean = np.linalg.norm(vec1 - vec2)

    return euclidean


euclidean_distance(np.array(VEC_1), np.array(VEC_2))
